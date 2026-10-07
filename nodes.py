"""ComfyUI nodes for LoRA the Explorer (ported from Fizgig).

Node graph, and what it replaces in the original app:

    FizgigLoraExplorer  --- the brain
        reads the LoRA's safetensors header, derives the block set the way
        WorkbenchEngine.primary_block_ids did, and rolls the four variants the
        "Roll variants" button rolled. One JSON output per variant.

    FizgigExplorerPick  --- "Variant N selected as new baseline"
        takes a variant's JSON, stamps seed/resolution/prompt into it and hands
        it back for the next roll. Wire variant N -> Pick -> Explorer.roll_state.

    FizgigLoraBlockLoader --- per-block strengths, applied for real
        Fizgig did this in memory (FamilyLoRA.set_blocks -> per-block
        multiplier on the live adapter). We do the equivalent by scaling the
        LoRA's own weights per block before handing the dict to ComfyUI's
        standard loader, so no per-block loader node is required.

    FizgigCheckpointScan --- LoRA Royale's scanner (lora_royale/scan.py),
        for feeding a run's epoch checkpoints one at a time.

Everything the tkinter tab did around the edges -- the undo stack, the
thumbnail refs, the VRAM unload on tab switch, the thread handoff to Tk's
`after()` -- has no counterpart here and is not carried over.

Two things worth knowing before reading the code:

1. Key layouts differ per family, and ComfyUI normalises some of them on load.

    Klein 9B       lora_unet_double_blocks_0_lora_up.weight    -> double_0
                   lora_unet_single_blocks_23_lora_up.weight   -> single_23
    Klein 9B       diffusion_model.double_blocks.0.lora_B.weight -> double_0
    Krea 2         lora_unet_blocks_4_lora_up.weight            -> block_4
                   lora_unet_txtfusion_layerwise_blocks_1_...   -> txt_lw_1
    Qwen 2.1       transformer.transformer_blocks.4.attn.to_q.lora_B.weight
    MiniMax H3     lora_unet_blocks_7_lora_up.weight            -> h3blk_7
                   lora_unet_token_refiner_blocks_1_...         -> h3_rf_1

   Qwen, Krea 2 and H3 share the bare `blocks_<n>` shape, so the family is read
   from the file's metadata first and only guessed from shape as a fallback.

2. ComfyUI forbids cycles in a graph. Wiring a node's output back into a node
   that feeds it -- Explorer.variant_1 -> Pick -> Explorer -- fails validation
   before anything runs ("Failed to validate prompt for output ..."), whatever
   the types are. The Explorer's loop has to be broken, and there are two ways,
   both supported:

   - `state_source = "picked variant"` (default). One generation per run:
     roll, look, then edit the `roll_state` widget to the Pick node's
     `baseline` text, and re-queue. `last_pick_blocks` gets the Pick's
     `changed_blocks` the same way. Two paste operations per generation.
   - `state_source = "bootstrap from LoRA weights"`. In-graph and repeatable:
     the Explorer reads the per-block strengths straight out of a baked LoRA
     (`bootstrap_lora`, measured against `bootstrap_reference`), so a save-LoRA
     node closes the loop through the filesystem instead of through a wire.
     Bake, re-queue, and the next roll starts from what you baked. No pasting.

3. The loop inputs are widgets rather than dangling sockets on purpose: an
   unfilled widget is visible on the node, where an optional socket that never
   got connected is not. That distinction cost a user four identical renders
   once.

The loader prints its report to the ComfyUI console as well as the `report`
output, because a STRING output is not displayed anywhere by default.
"""

import json
import logging
import os
import uuid
from typing import Dict, Optional

import folder_paths

from .explorer_core import (
    ANCHORS,
    IDENTITY_BLOCKS,
    SliderState,
    block_id_from_key,
    block_sort_key,
    family_from_metadata,
    guess_family,
    read_safetensors_keys,
    roll_variants,
    scan_block_ids,
    scan_checkpoints,
    state_from_baked,
    summarise_key_shapes,
)

logger = logging.getLogger(__name__)

# `auto` reads the family from file metadata (falling back to key shape).
# The rest are Fizgig's workbench families; `qwen21` is Qwen Image 2.1.
FAMILIES = ["auto", "klein9b", "krea2", "qwen21", "h3"]

# How a roll decides where to start. See the module docstring: a wire back into
# this node would be a cycle, and ComfyUI refuses those outright.
STATE_PICKED = "picked variant"
STATE_BOOTSTRAP = "bootstrap from LoRA weights"
STATE_FRESH = "fresh state"
STATE_SOURCES = [STATE_PICKED, STATE_BOOTSTRAP, STATE_FRESH]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lora_path(name: str) -> str:
    path = folder_paths.get_full_path("loras", name)
    if not path or not isinstance(path, str):
        raise FileNotFoundError(f"LoRA not found: {name}")
    return path


def _resolve_family(family: str, path: str) -> str:
    if family != "auto":
        return family
    return guess_family(path)


def _anchor_for(family: str, block_ids) -> str:
    """The composition anchor, with a fallback for shapes we did not foresee."""
    anchor = ANCHORS.get(family, "double_0")
    if anchor in block_ids:
        return anchor
    for candidate in ("double_0", "block_0", "h3blk_0"):
        if candidate in block_ids:
            return candidate
    return next(iter(block_ids)) if block_ids else anchor


def _resolve_file(entry: str, lora_name: str) -> str:
    """A file the user named: an absolute path, or a name in either loras or the
    `fizgig_loras` folder this package registers (so a bake can write somewhere
    of its own)."""
    if not entry or entry == lora_name:
        return _lora_path(lora_name)
    if os.path.isabs(entry) and os.path.isfile(entry):
        return entry
    for kind in ("loras", "fizgig_loras"):
        try:
            got = folder_paths.get_full_path(kind, entry)
        except Exception:
            got = None
        if got and isinstance(got, str) and os.path.isfile(got):
            return got
    if os.path.isfile(entry):
        return os.path.abspath(entry)
    raise FileNotFoundError(
        f"{entry!r} not found (looked on disk and in the loras / fizgig_loras folders)")


def _state_from_baked_lora(baked_name, reference_name, active, family, primary_path):
    """Read per-block strengths out of a baked LoRA, against its reference file.

    This is the in-graph half of the loop: a save-LoRA node writes the picked
    variant, and the next roll reads it back. It has to go through the
    filesystem because a wire back into this node would be a cycle, and ComfyUI
    refuses those.

    A multiplier is a *ratio* of the baked factor against the file it was baked
    from (see core.strength_from_pair), so both files are needed. Either may be
    `lora_name`, which is the useful default: with no bake yet, baked ==
    reference == the LoRA, every ratio is 1.0, and the first roll starts fresh.
    """
    import comfy.utils

    baked_path = _resolve_file(baked_name, primary_path)
    ref_path = _resolve_file(reference_name, primary_path)
    if os.path.normcase(baked_path) == os.path.normcase(ref_path):
        print(f"[FizgigLoraExplorer] bootstrap: {os.path.basename(baked_path)} is its own "
              "reference -- reading every block as 1.0 (a fresh start)", flush=True)

    baked = comfy.utils.load_torch_file(baked_path, safe_load=True)
    if os.path.normcase(baked_path) == os.path.normcase(ref_path):
        reference = baked
    else:
        reference = comfy.utils.load_torch_file(ref_path, safe_load=True)

    state = state_from_baked(baked, reference, active, family=family)
    measured = {b: bs for b, bs in state.blocks.items()
                if bs.primary_enabled is False or bs.primary_strength != 1.0}
    print(f"[FizgigLoraExplorer] bootstrap from {os.path.basename(baked_path)} "
          f"vs {os.path.basename(ref_path)}: {len(measured)} block(s) differ from 1.0"
          + ("" if measured else " -- this reads as an unedited file"), flush=True)
    return state


def _no_blocks_error(lora_name: str, path: str, family: str) -> ValueError:
    """Say which layout the file actually uses, rather than only which ones we
    wanted. A LoRA that adapts nothing the Explorer can address is usually a
    text-encoder-only file, a LyCORIS/GLoRA one, or a layout from a family this
    port does not know yet -- the shapes tell you which."""
    keys, meta = read_safetensors_keys(path)
    shapes = summarise_key_shapes(keys)
    meta_bits = ", ".join(f"{k}={v}" for k, v in list(meta.items())[:4]) or "(none)"
    shown = "\n".join(f"    {s}" for s in shapes) or "    (the file has no tensor keys?)"
    return ValueError(
        f"{lora_name}: no per-block LoRA keys found (detected family: {family}).\n"
        f"  This file's key shapes:\n{shown}\n"
        f"  metadata: {meta_bits}\n"
        "The Explorer needs keys that name transformer blocks. Known layouts:\n"
        "    lora_unet_double_blocks_0_ / lora_unet_single_blocks_23_  (Klein 9B)\n"
        "    lora_unet_blocks_4_                                      (Krea 2)\n"
        "    transformer.transformer_blocks.4.attn.to_q.lora_B.weight (Qwen Image 2.1)\n"
        "    lora_unet_blocks_7_ / lora_unet_token_refiner_blocks_1_  (MiniMax H3)\n"
        "If this is one of those families, check the `family` widget -- on Qwen, "
        "Krea 2 and H3 the bare `blocks_N` shape is ambiguous and it decides "
        "block_N vs h3blk_N. Otherwise use active_blocks_override to name the "
        "blocks by hand."
    )


# ---------------------------------------------------------------------------
# 1. the brain
# ---------------------------------------------------------------------------

class FizgigLoraExplorer:
    """Roll four mutated variants of a LoRA's per-block strengths."""

    CATEGORY = "Fizgig/Explorer"
    FUNCTION = "roll"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("variant_1", "variant_2", "variant_3", "variant_4",
                    "baseline", "active_blocks")
    DESCRIPTION = (
        "LoRA the Explorer, ported from Fizgig. Rolling produces four mutated "
        "variants of the loaded LoRA's per-block strengths: 3 is pure random, 4 "
        "avoids the blocks your last pick touched, and 1-2 add the composition-"
        "anchor structural hit when structural_variants is on (off by default: "
        "it negates the anchor and inverts an edit). Send each variant into "
        "FizgigLoraBlockLoader, then wire the loader's MODEL/CLIP into its own "
        "KSampler. When you like one, feed it back through FizgigExplorerPick "
        "into roll_state and roll again."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "family": (FAMILIES, {"default": "auto"}),
                "mutations": ("INT", {"default": 3, "min": 1, "max": 64,
                                      "tooltip": "Blocks perturbed per variant. Fizgig's own "
                                                 "refinement profile used 2-3."}),
                "intensity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "0.0 = tiny nudges (+-0.2), 1.0 = bold moves "
                                                   "(+-3.0). The structural hit below only fires "
                                                   "above 0.3, so a first roll is safe at or under it."}),
                "structure": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "How hard variants 1 and 2 may hit the anchor. "
                                                   "1.0 NEGATES the composition block (-1.0), which on "
                                                   "an edit LoRA inverts the edit -- the warped, "
                                                   "mirror-like frame. 0.0 is a plain mutation."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "Drives the mutation RNG, not the sampler."}),
                # Widgets, not sockets: a link connected to a widget input replaces
                # its value, and an unfilled widget is at least visible. A dangling
                # optional socket is how a roll silently stops being a roll.
                #
                # The GUI's Freeze set and the previous pick's changed blocks,
                # as comma-separated block ids.
                "locked_blocks": ("STRING", {"default": "",
                                             "tooltip": "Comma-separated block ids to freeze. "
                                                        "On Qwen Image 2.1, block_10..block_14 carry the "
                                                        "identity -- lock them to keep the face while the "
                                                        "picture moves."}),
                "last_pick_blocks": ("STRING", {"default": "",
                                                "tooltip": "Variant 4 excludes these. Connect "
                                                           "FizgigExplorerPick.changed_blocks here."}),
                "state_source": (STATE_SOURCES, {"default": STATE_PICKED,
                                                 "tooltip": "Where this roll starts from.\n"
                                                            "'picked variant': the roll_state text (paste the "
                                                            "Pick node's baseline).\n"
                                                            "'bootstrap from LoRA weights': read the per-block "
                                                            "strengths out of a baked LoRA -- closes the loop "
                                                            "in-graph, no pasting.\n"
                                                            "'fresh state': every block at 1.0, every run."}),
                "roll_state": ("STRING", {"default": "",
                                          "tooltip": "Where the roll starts. With 'picked variant', paste the "
                                                     "Pick node's baseline here (a wire would make a cycle, which "
                                                     "ComfyUI refuses). Empty = fresh state, every block at 1.0."}),
                "bootstrap_lora": ("STRING", {"default": "",
                                              "tooltip": "Only for 'bootstrap from LoRA weights'. The baked LoRA to "
                                                         "read the strengths from. Empty = use lora_name (which "
                                                         "reads as all 1.0 -- the first generation)."}),
                "bootstrap_reference": ("STRING", {"default": "",
                                                   "tooltip": "The LoRA the baked one came from. A multiplier is a "
                                                              "ratio, so it needs a reference. Empty = use lora_name."}),
                "active_blocks_override": ("STRING", {"default": "",
                                                      "tooltip": "Override the detected block set, comma-separated."}),
            },
            "optional": {
                # Added after the first release, so it lives at the END and is
                # optional: an older workflow has no value for it, and a missing
                # required input fails validation before the graph runs
                # ("Required input is missing"). Absent = the safe default.
                "structural_variants": ("BOOLEAN", {"default": False,
                                                    "tooltip": "Off (default): variants 1 and 2 are plain "
                                                               "mutations like variant 3 -- no anchor invert. "
                                                               "On: they carry the anchor invert/extreme, which "
                                                               "is Fizgig's exploration profile and what inverts "
                                                               "an edit."}),
            },
        }

    def roll(self, lora_name, family, mutations, intensity, structure, seed,
             locked_blocks="", last_pick_blocks="", state_source=STATE_PICKED,
             roll_state="", bootstrap_lora="", bootstrap_reference="",
             active_blocks_override="", structural_variants=False):
        path = _lora_path(lora_name)
        fam = _resolve_family(family, path)

        if active_blocks_override.strip():
            active = [b.strip() for b in active_blocks_override.split(",") if b.strip()]
        else:
            active = scan_block_ids(path, family=fam)
        if not active:
            raise _no_blocks_error(lora_name, path, fam)

        locked = {b.strip() for b in locked_blocks.split(",") if b.strip()}
        last_pick = {b.strip() for b in last_pick_blocks.split(",") if b.strip()}
        anchor = _anchor_for(fam, active)

        rolling_from = state_source != STATE_FRESH
        if state_source == STATE_BOOTSTRAP:
            baseline = _state_from_baked_lora(bootstrap_lora.strip() or lora_name,
                                              bootstrap_reference.strip() or lora_name,
                                              active, fam, path)
        elif rolling_from and roll_state.strip():
            baseline = SliderState.from_json(json.loads(roll_state))
            # A map from another family is the one error that makes four renders
            # identical while nothing else looks wrong: the ids do not exist in
            # this file, so nothing is ever scaled. Say so where the cause is
            # visible, rather than in the loader.
            prev_fam = baseline.family
            if prev_fam and prev_fam != fam:
                sample = sorted(baseline.blocks, key=block_sort_key)[0]
                raise ValueError(
                    f"roll_state is a {prev_fam} map but {lora_name} is {fam}.\n"
                    f"  The map holds {len(baseline.blocks)} blocks in {prev_fam} naming "
                    f"(e.g. {sample}); this file has {', '.join(active[:2])}, ...\n"
                    f"  Nothing would be scaled and all four variants would render alike.\n"
                    f"  Clear the roll_state widget for a fresh {fam} baseline, or load "
                    f"the LoRA this map came from."
                )
            baseline.blocks.setdefault(anchor, baseline.blocks.get(anchor))
            baseline.blocks = {bid: bs for bid, bs in baseline.blocks.items() if bs is not None}
        else:
            # STATE_FRESH, or STATE_PICKED with nothing pasted yet -- the first
            # generation either way: every block at 1.0.
            baseline = SliderState.from_block_ids(active)

        baseline.family = fam
        baseline.seed = seed

        # The anchor invert/extreme is the one mutation that does not read as a
        # nudge: at structure 1.0 it drives the composition block to -1.0, and for
        # an edit LoRA that inverts the edit. It is opt-in, and state.py only lets
        # it fire above intensity 0.3.
        use_structure = structure if structural_variants else 0.0

        variants = roll_variants(
            baseline, active,
            num_mutations=mutations, intensity=intensity, structure=use_structure,
            anchor=anchor, last_pick_blocks=last_pick, seed=seed,
            locked_blocks=locked,
        )

        # A roll that changes nothing renders four identical images. Say so here
        # rather than leaving it to be discovered in the output.
        moved = [sorted(v.diff_blocks(baseline)) for v in variants]

        # How far the BASELINE sits from a plain 1.0 LoRA. A roll from "picked
        # variant" (or from a bootstrap bake) inherits every earlier generation's
        # edits, so each new variant starts from an already-moved file: "1 changed"
        # is still a big render change, and the delta list alone hides it. This is
        # the line that explains "even at zero settings the picture is distorted".
        drift = {bid: bs.primary_strength for bid, bs in baseline.blocks.items()
                 if bs.primary_strength != 1.0 or not bs.primary_enabled}
        drift_note = ""
        if drift:
            worst = sorted(drift.items(), key=lambda kv: -abs(kv[1] - 1.0))[:5]
            drift_note = ("\n    BASELINE IS NOT FRESH: %d of %d block(s) differ from 1.0"
                          " -- this roll starts from an already-edited state\n      worst: %s"
                          % (len(drift), len(baseline.blocks),
                             ", ".join("%s=%+.2f" % (b, v) for b, v in worst)))
        if not any(moved):
            logger.warning(
                "FizgigLoraExplorer: %s -- all four variants are identical to the baseline. "
                "Every active block is locked, or the lock set covers the whole LoRA.",
                lora_name)
        for i, blocks in enumerate(moved):
            logger.info("FizgigLoraExplorer: variant %d changed %d block(s): %s",
                        i + 1, len(blocks), ", ".join(blocks) or "(none)")
        print(
            f"[FizgigLoraExplorer] {lora_name}  family={fam}  anchor={anchor}  "
            f"blocks={len(active)}  from={state_source}\n"
            + drift_note + "\n"
            + "\n".join(
                f"    variant {i + 1}: {len(b)} changed"
                + (f" -- {', '.join(b[:6])}{' ...' if len(b) > 6 else ''}" if b else " -- NOTHING")
                + ("  [ANCHOR INVERTED: %s -> %+.2f]" % (anchor, variants[i].blocks[anchor].primary_strength)
                   if anchor in variants[i].blocks
                   and variants[i].blocks[anchor].primary_strength < -0.05 else "")
                for i, b in enumerate(moved)),
            flush=True,
        )
        if not structural_variants and structure > 0.0:
            logger.info("FizgigLoraExplorer: structure=%.2f but structural_variants is off -- "
                        "no anchor invert/extreme this roll", structure)

        payloads = [json.dumps(v.to_json()) for v in variants]
        meta = json.dumps({
            "family": fam, "anchor": anchor, "lora": lora_name,
            "active": active, "locked": sorted(locked),
            "identity_blocks": sorted(IDENTITY_BLOCKS.get(fam, ())),
            "seed": seed, "intensity": intensity, "structure": structure,
            "mutations": mutations, "state_source": state_source,
            "changed": moved,
        })
        return (*payloads, json.dumps(baseline.to_json()), meta)


# ---------------------------------------------------------------------------
# 2. pick a variant -> new baseline
# ---------------------------------------------------------------------------

class FizgigExplorerPick:
    """Take a variant as the new baseline. The graph equivalent of the pick
    button: the chosen variant becomes the state, and the blocks it changed
    are reported so variant 4 can route around them."""

    CATEGORY = "Fizgig/Explorer"
    FUNCTION = "pick"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("baseline", "changed_blocks", "readout")
    DESCRIPTION = ("Selects one variant as the new baseline. Wire `baseline` into "
                   "FizgigLoraExplorer.roll_state and `changed_blocks` into its "
                   "last_pick_blocks, then roll again.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": ("STRING", {"forceInput": True}),
                "previous_baseline": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF,
                                 "tooltip": "-1 keeps the variant's own seed."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "preview_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "preview_height": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "load_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0,
                                            "step": 0.05,
                                            "tooltip": "The strength the LoRA is used at; every block is relative to it."}),
            },
        }

    def pick(self, variant, previous_baseline="", seed=-1, prompt="",
             preview_width=0, preview_height=0, load_strength=1.0):
        state = SliderState.from_json(json.loads(variant))
        if seed >= 0:
            state.seed = seed
        if prompt:
            state.prompt = prompt
        if preview_width:
            state.preview_width = preview_width
        if preview_height:
            state.preview_height = preview_height
        state.primary_scale = load_strength

        changed = ""
        if previous_baseline.strip():
            try:
                prev = SliderState.from_json(json.loads(previous_baseline))
                changed = ",".join(state.diff_blocks(prev))
            except (ValueError, TypeError, KeyError):
                logger.warning("FizgigExplorerPick: previous_baseline was not a state; "
                               "variant 4 protection is off this round.")
        return (json.dumps(state.to_json()), changed,
                state.to_plain_text(family=_family_hint(state)))


def _family_hint(state: SliderState) -> str:
    """Which family's block naming a state uses -- only for the readout tags."""
    ids = list(state.blocks)
    if any(b.startswith("double_") or b.startswith("single_") for b in ids):
        return "klein9b"
    if any(b.startswith("h3blk_") or b.startswith("h3_rf_") for b in ids):
        return "h3"
    return "qwen21" if ids and IDENTITY_BLOCKS["qwen21"] & set(ids) else "krea2"


# ---------------------------------------------------------------------------
# 3. apply per-block strengths (live, no baked file)
# ---------------------------------------------------------------------------

# ComfyUI's loader derives a module's multiplier as alpha/rank. Fizgig's bake
# absorbs the slider into the "up" factor and sets alpha = rank so the file's
# own scale becomes 1.0 (bake.py:_bake_single_contribution). We do the same
# absorption, but only in memory -- nothing is written to disk.
#
# Every layout puts the output factor last, but the separators differ: kohya
# uses an underscore (lora_unet_blocks_0_lora_up.weight), Qwen and other
# diffusers-format files use a dot (transformer_blocks.0.attn.to_q.lora_B.weight).
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".lora_emb.weight",
                "_lora_up.weight", "_lora_B.weight", "_lora_emb.weight")
# LyCORIS. Both forms are linear in their first factor --
#   m * kron(w1, w2) == kron(m * w1, w2)          (LoKr)
#   (m * W1) # W2    == m * (W1 # W2)             (LoHa, # = Hadamard)
# -- so a multiplier absorbs into w1 exactly the way it absorbs into lora_up,
# which is what bake.py does to keep a LoKR in, a LoKR out. ComfyUI reads w1 as
# `lokr_w1_a @ lokr_w1_b` when the split form is present, else `lokr_w1`, so the
# factor to scale is whichever one ComfyUI reads as the first.
#
# Without this a LyCORIS file has no tensor a slider can reach: nothing is
# scaled and every variant renders identically. That "LoKR in, LoKR out" line in
# Fizgig's release notes is this.
_LYCORIS_FIRST_FACTORS = (".lokr_w1_a", ".lokr_w1", ".hada_w1_a", ".hada_w1",
                          "_lokr_w1_a", "_lokr_w1", "_hada_w1_a", "_hada_w1")
# Kept uniform: GLoRA's 4-matrix form and the Tucker/CP variants, which Fizgig's
# bake refuses too.
_LYCORIS_UNSUPPORTED = ("glora", "lora_tucker", "lora_cp")


def _lycoris_first_factor(key: str) -> Optional[str]:
    """The suffix to multiply if this is a LyCORIS key we can scale, else None.

    A full tensor key and a bare suffix both work, because the loader sees
    `transformer.transformer_blocks.3.attn.to_q.lokr_w1` while the bake sees the
    grouped suffix `lokr_w1`.
    """
    if any(m in key for m in _LYCORIS_UNSUPPORTED):
        return None
    bare = str(key).lstrip("._")
    for suffix in _LYCORIS_FIRST_FACTORS:
        sfx = str(suffix).lstrip("._")
        if bare == sfx or bare.endswith("." + sfx) or bare.endswith("_" + sfx):
            return suffix
    return None


class FizgigLoraBlockLoader:
    """Load a LoRA with a per-block multiplier map from the Explorer.

    Kohya, diffusers and LyCORIS (LoKR/LoHa) keys are all scaled -- a LyCORIS
    file is linear in its w1, so the multiplier absorbs there the same way it
    absorbs into lora_up, which is what Fizgig's bake does to keep a LoKR in, a
    LoKR out. GLoRA/Tucker keys are left uniform, as in Fizgig's bake, and the
    report says how many.
    """

    CATEGORY = "Fizgig/Explorer"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    DESCRIPTION = ("Loads a LoRA with per-block multipliers taken from a Fizgig "
                   "Explorer variant. Equivalent to Fizgig's live set_blocks(), "
                   "except the scaling happens on the LoRA's weights before "
                   "ComfyUI's standard loader sees them. The report is printed "
                   "to the console.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "block_map": ("STRING", {"default": "", "forceInput": True,
                                         "tooltip": "A variant JSON from FizgigLoraExplorer."}),
            },
            "optional": {
                "family": (FAMILIES, {"default": "auto",
                                      "tooltip": "Disambiguates block_N vs h3blk_N."}),
                "unlisted_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0,
                                                "step": 0.05,
                                                "tooltip": "Applied to keys outside the block map (text encoder, io)."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
                "scale_unlisted": ("BOOLEAN", {"default": True,
                                               "tooltip": "On: keys outside the block map are scaled by "
                                                          "unlisted_strength. Off: left untouched."}),
                "verbose": ("BOOLEAN", {"default": False,
                                        "tooltip": "Print every block's multiplier in the report. "
                                                   "Run the four branches with this on and compare the "
                                                   "reports -- if they match, the graph is sending the "
                                                   "same map to all four."}),
            },
        }

    def load(self, model, clip, lora_name, block_map, family="auto",
             unlisted_strength=1.0, strength_model=1.0, strength_clip=1.0,
             scale_unlisted=True, verbose=False):
        import comfy.sd
        import comfy.utils

        path = _lora_path(lora_name)
        try:
            state = SliderState.from_json(json.loads(block_map)) if block_map.strip() else None
        except (ValueError, TypeError) as e:
            raise ValueError(f"block_map is not an Explorer variant JSON: {e}") from e

        # The map knows which family's block naming it carries, and that is
        # what decides how its ids are matched against this file. An explicit
        # `family` widget still wins -- it is the user's word against the map's.
        # Past that, a disagreement is the failure that looks like nothing at
        # all: the ids are simply never found, nothing is scaled, and all four
        # branches render identically.
        detected = _resolve_family("auto", path)
        widget_fam = family if family != "auto" else ""
        map_fam = state.family if state is not None else ""
        fam = widget_fam or map_fam or detected

        # A map whose ids belong to another family does not "scale nothing" --
        # it scales the WRONG blocks, because two families can share a block
        # count (Qwen and Krea 2 both have 32). Every module gets a multiplier
        # meant for a different block, the counts all look healthy, and the
        # result is a distorted but still recognisable picture. That is strictly
        # worse than refusing, so refuse.
        if map_fam and detected and map_fam != detected and not widget_fam:
            raise ValueError(
                f"family mismatch: the block map is {map_fam} but {lora_name} reads as "
                f"{detected}.\n"
                f"  The map holds {len(state.blocks)} blocks in {map_fam} naming "
                f"(e.g. {sorted(state.blocks)[0]}); applying them would scale the wrong "
                f"blocks -- the picture distorts while every counter reports success.\n"
                f"  Clear roll_state for a fresh {detected} baseline, or roll this LoRA "
                f"from its own Explorer node."
            )

        lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
        report, warnings = _scale_lora_weights(lora_sd, state, path, fam,
                                              unlisted_strength, scale_unlisted, verbose)
        if widget_fam and map_fam and widget_fam != map_fam:
            raise ValueError(
                f"family mismatch: the block map is {map_fam} but the family widget says "
                f"{widget_fam} for {lora_name}.\n"
                f"  Two families can share a block count, so this would silently scale "
                f"the wrong blocks. Set family to 'auto' or to {map_fam}."
            )
        elif widget_fam and map_fam and widget_fam != map_fam:
            warnings.append(
                f"family mismatch: the map is {map_fam} but the family widget says "
                f"{widget_fam} -- the map's blocks do not line up with this file")

        try:
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model, clip, lora_sd, strength_model, strength_clip)
        except Exception as e:
            raise RuntimeError(
                f"{lora_name}: the scaled weights could not be applied to this model.\n"
                f"  {type(e).__name__}: {e}\n{report}"
            ) from e
        if model_out is None:
            model_out = model
        if clip_out is None:
            clip_out = clip

        # "Four identical renders" is almost always answered right here: the
        # map is fine and the variant differs, but the LoRA matched none of the
        # loaded model's modules, so no patch exists for the sliders to scale.
        matched_mods, matched_blocks, err = _match_report(model, clip, lora_sd, fam)
        if err:
            report += f"\n{err}"
        elif matched_mods == 0:
            warnings.append(
                "MODEL was NOT patched: this LoRA adapts none of the loaded model's modules. "
                "Every variant will render identically. Check that lora_name matches the "
                "architecture you are sampling (a Qwen LoRA on a Klein model patches nothing)")
        else:
            report += (f"\nmodules patchable by this LoRA: {matched_mods}"
                       f"  |  blocks among them: {matched_blocks}")
            in_lora = _blocks_in_lora(lora_sd, fam)
            if state is not None and in_lora and not (in_lora & set(state.blocks)):
                warnings.append(
                    "the LoRA's blocks do not line up with the block map -- the map is for "
                    "another family or another LoRA")
        if clip_out is None or clip_out is clip:
            report += "\n(CLIP was not patched -- normal for a block-only LoRA.)"

        report = "\n".join(warnings + [report]) if warnings else report
        print(f"[FizgigLoraBlockLoader] {lora_name}\n{report}", flush=True)
        for w in warnings:
            logger.warning("FizgigLoraBlockLoader: %s -- %s", lora_name, w)
        return (model_out, clip_out, report)


def _blocks_in_lora(lora_sd, family: str):
    """The block ids a loaded LoRA's own keys address."""
    out = set()
    for k in lora_sd:
        bid = block_id_from_key(k, family=family)
        if bid:
            out.add(bid)
    return out


def _match_report(model, clip, lora_sd, family):
    """How many of the loaded model's modules this LoRA can actually patch.

    Returns (module_count, block_count, error_text). ComfyUI's own
    `load_lora_for_models` clones the model regardless, so object identity
    cannot answer this; we ask the same key-mapping functions it uses. These
    internals move between ComfyUI releases, so every failure is reported as
    text instead of raised -- diagnostics must never break a load.
    """
    try:
        import comfy.lora
    except Exception as e:                                            # pragma: no cover
        return 0, 0, f"(comfy.lora not importable, match check skipped: {e})"
    try:
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        try:
            loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
        except TypeError:
            loaded = comfy.lora.load_lora(lora_sd, key_map)
    except Exception as e:
        return 0, 0, f"(could not measure how many modules this LoRA patches: {type(e).__name__}: {e})"

    blocks = set()
    for patch_keys in loaded.keys():
        for k in (patch_keys if isinstance(patch_keys, (tuple, list)) else [patch_keys]):
            bid = block_id_from_key(str(k), family=family)
            if bid:
                blocks.add(bid)
    return len(loaded), len(blocks), ""


def _scale_lora_weights(lora_sd, state, path, family, unlisted_strength, scale_unlisted,
                        verbose=False):
    """Multiply each module's first factor by its block's slider value.

    One factor per module is all the maths needs: LoRA is linear in
    lora_up/lora_B, LyCORIS in its w1. Returns (report_text, warnings)."""
    import torch

    if state is None:
        return ("No block map supplied -- loaded at a uniform strength. "
                "Every one of the four branches had no map to apply, so they will "
                "all render identically.", [])

    keys = list(lora_sd.keys())
    fam = family
    if not fam or fam == "auto":
        _, meta = read_safetensors_keys(path)
        fam = family_from_metadata(meta) or guess_family(path)

    touched, outside, scaled = set(), 0, 0
    lycoris_scaled, lycoris_uniform = 0, 0

    for key in keys:
        is_lycoris = _lycoris_first_factor(key)
        is_plain = key.endswith(_UP_SUFFIXES) or any((s + ".") in key for s in _UP_SUFFIXES)
        if not is_lycoris and not is_plain:
            continue
        bid = block_id_from_key(key, family=fam)
        if bid is None:
            if not scale_unlisted:
                continue
            mult = float(unlisted_strength)
            outside += 1
        else:
            bs = state.blocks.get(bid)
            if bs is None:
                mult = float(unlisted_strength)
                outside += 1
            else:
                mult = float(bs.primary_strength) if bs.primary_enabled else 0.0
                touched.add(bid)

        try:
            lora_sd[key] = (lora_sd[key].to(torch.float32) * mult).to(lora_sd[key].dtype)
            scaled += 1
            if is_lycoris:
                lycoris_scaled += 1
        except Exception as e:            # pragma: no cover - defensive
            logger.warning("FizgigLoraBlockLoader: could not scale %s (%s)", key, e)

    off = [b for b, bs in state.blocks.items() if not bs.primary_enabled]
    lines = [
        f"family: {fam}",
        f"blocks in map: {len(state.blocks)}  |  touched by this LoRA: {len(touched)}",
        f"tensors scaled: {scaled}  |  outside the map: {outside}  |  off: {len(off)}",
    ]
    if state.primary_scale != 1.0:
        lines.append(f"load strength {state.primary_scale:g} -- every block is relative to it")
    missing = sorted(set(state.blocks) - touched, key=block_sort_key)
    if missing:
        lines.append("in the map but absent from this file: "
                     + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if lycoris_scaled:
        lines.append(f"LyCORIS first factors scaled: {lycoris_scaled} "
                     "(multiplier into w1 -- LoKR/LoHa are linear in it)")
    if lycoris_uniform:
        lines.append(f"GLoRA/Tucker keys left uniform: {lycoris_uniform} "
                     "(Fizgig's bake refuses these too)")
    if verbose:
        moved = sorted(touched, key=block_sort_key)
        lines.append("multipliers: "
                     + ", ".join(f"{b}={'off' if not state.blocks[b].primary_enabled else format(state.blocks[b].primary_strength, '.3f')}"
                                 for b in moved))

    warnings = []
    if scaled == 0:
        warnings.append(
            "WARNING: no tensors were scaled -- no variant can differ from any other. "
            "This LoRA's keys do not name blocks this port recognises")
    if scaled and not touched:
        warnings.append(
            "WARNING: tensors were scaled but not one of them belonged to a block in the map")
    return ("\n".join(lines), warnings)


# ---------------------------------------------------------------------------
# 4. bake the chosen variant into a file
# ---------------------------------------------------------------------------

# Every way a factor key ends, longest first so `lora_up.weight` is not eaten by
# `lora_up`. The separator is part of the suffix: kohya uses an underscore
# (`lora_unet_blocks_0_lora_up.weight`), Qwen/diffusers a dot
# (`...to_q.lora_B.weight`). Getting this wrong is silent -- the module keeps a
# `weight` suffix, no factor is found, and the module is copied through unbaked.
_FACTOR_ENDINGS = tuple(sorted((
    "_lora_up.weight", "_lora_down.weight", "_lora_emb.weight",
    "_lora_up", "_lora_down", "_lora_emb",
    "_lora_A.weight", "_lora_B.weight", "_lora_A", "_lora_B",
    ".lora_up.weight", ".lora_down.weight", ".lora_emb.weight",
    ".lora_up", ".lora_down", ".lora_emb",
    ".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B",
    ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_w1", ".lokr_w2",
    "_lokr_w1_a", "_lokr_w1_b", "_lokr_w2_a", "_lokr_w2_b", "_lokr_w1", "_lokr_w2",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
    ".hada_w1", ".hada_w2",
    "_hada_w1_a", "_hada_w1_b", "_hada_w2_a", "_hada_w2_b", "_hada_w1", "_hada_w2",
    ".glora_a", ".glora_b", ".glora_alpha",
    ".lokr_alpha", "_lokr_alpha", ".hada_alpha", "_hada_alpha",
    ".lora_tucker", ".lora_cp", ".dora_scale",
    ".alpha", "_alpha",
), key=len, reverse=True))


# The factor names themselves, without any separator. A key is split by finding
# one of these as its trailing segment, no matter how short the module name is:
# `lokr_w1` alone must split (module "", factor "lokr_w1"), and
# `lora_unet_blocks_3.lokr_alpha` must split after `blocks_3`, not after `lokr`.
_FACTOR_NAMES = tuple(sorted({
    "lora_up.weight", "lora_down.weight", "lora_emb.weight",
    "lora_up", "lora_down", "lora_emb",
    "lora_A.weight", "lora_B.weight", "lora_A", "lora_B",
    "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_w1", "lokr_w2",
    "lokr_alpha", "lokr_t2",
    "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b",
    "hada_t1", "hada_t2", "hada_w1", "hada_w2", "hada_alpha",
    "glora_a", "glora_b", "glora_alpha",
    "lora_tucker", "lora_cp", "dora_scale",
    "alpha",
}, key=len, reverse=True))


def _split_factor(key: str):
    """(module_name, factor) for one tensor key, or None if it names no factor.

    `transformer.transformer_blocks.0.attn.to_q.lora_B.weight`
        -> ("transformer.transformer_blocks.0.attn.to_q", "lora_B.weight")
    `lora_unet_double_blocks_0_lora_up.weight`
        -> ("lora_unet_double_blocks_0", "lora_up.weight")
    `lokr_w1` -> ("", "lokr_w1")
    `lora_unet_blocks_3.lokr_alpha` -> ("lora_unet_blocks_3", "lokr_alpha")
    """
    for name in _FACTOR_NAMES:
        if key == name:
            return "", name
        for sep in (".", "_"):
            tail = sep + name
            if key.endswith(tail):
                module = key[: -len(tail)]
                if module:
                    return module, name
    return None


def _group_by_module(sd: Dict) -> Dict[str, Dict]:
    """{module_name: {suffix: tensor}} -- one entry per LoRA module, so the
    bake can reason per module rather than per tensor. Keys that name no
    factor (rare, and never a block) keep their own name as a module of one."""
    out: Dict[str, Dict] = {}
    for key, tensor in sd.items():
        split = _split_factor(key)
        if split is None:
            head, sep, tail = key.rpartition(".")
            module, suffix = (head, tail) if sep else (key, "")
        else:
            module, suffix = split
        out.setdefault(module, {})[suffix] = tensor
    return out


def _alpha_key(mod_keys: Dict) -> Optional[str]:
    for k in mod_keys:
        if k == "alpha" or k.endswith(".alpha") or k.endswith("_alpha"):
            return k
    return None


def _rank_of(mod_keys: Dict, up_key: str) -> int:
    """The rank ComfyUI's loader divides alpha by. Both kohya `lora_up` and
    diffusers `lora_B` are [out, rank], so the last axis is the rank in every
    layout this port handles."""
    try:
        dims = list(getattr(mod_keys[up_key], "shape", []) or [])
        return int(dims[-1]) if dims else 0
    except Exception:
        return 0


# The suffix check has to accept both spellings: _group_by_module hands over
# `lora_B.weight` (no leading dot), while a full key reads
# `transformer.transformer_blocks.0.attn.to_q.lora_B.weight`. Writing the
# constants with a leading dot and matching only full keys is how this node
# silently copied every module through unbaked once.
def _is_factor_suffix(name: str, suffixes) -> bool:
    """True when `name` is one of `suffixes`. Both sides are stripped of their
    separators first: the constants carry a leading dot, _split_factor hands over
    bare names like `lora_B.weight`, and an underscore-separated kohya name
    (`lora_unet_blocks_0_lora_up.weight` -> `lora_up.weight`) must match too."""
    bare = str(name).lstrip("._")
    for sfx in suffixes:
        sfx_bare = str(sfx).lstrip("._")
        if bare == sfx_bare or bare.endswith("." + sfx_bare) or bare.endswith("_" + sfx_bare):
            return True
    return False


def _factor_sample(mod_keys: Dict) -> str:
    """The key that decides what kind of module this is.

    NOT `next(iter(mod_keys))`: a dict's first key is insertion order, and a
    LoKR module comes out of _group_by_module as
    `{"lokr_alpha": ..., "lokr_w1": ..., "lokr_w2": ...}` -- alpha first, which
    carries no information about the form. Pick a factor, not a scalar.
    """
    for key in mod_keys:
        if key.lstrip("._").endswith("alpha"):
            continue
        if _lycoris_first_factor(key) or _is_factor_suffix(key, _UP_SUFFIXES):
            return key
    return next(iter(mod_keys), "")


def _bake_module(mod_keys: Dict, multiplier: float, fuse_lycoris: bool,
                 alpha_sentinel: float = 1e6, sentinel: bool = False) -> Optional[Dict]:
    """Absorb `multiplier` into one module's first factor.

    Mirrors bake.py:_bake_single_contribution (kohya/diffusers) and
    _bake_single_lycoris_contribution (LoKR/LoHa). The multiplier is linear in
    the first factor, so it absorbs there and the module keeps its shape -- a
    LoKR stays a LoKR, no SVD, no loss. GLoRA/Tucker returns None, as in bake.py.

    A multiplier of exactly 1.0 returns the input untouched, original alpha
    included: the no-op guarantee that keeps blocks you never moved byte-identical.
    """
    import torch

    # Docstring heading. (The real one is above the function.)
    sample = _factor_sample(mod_keys)
    if any(m in sample for m in _LYCORIS_UNSUPPORTED):
        return None
    out = dict(mod_keys)

    def _set_alpha(value):
        k = _alpha_key(out)
        if k is not None:
            out[k] = torch.tensor(float(value), dtype=out[k].dtype)

    lycoris_suffix = _lycoris_first_factor(sample)
    if lycoris_suffix:
        if not fuse_lycoris:
            return None
        if abs(multiplier - 1.0) < 1e-12:
            return dict(mod_keys)
        hit = False
        for k in list(out):
            if _is_factor_suffix(k, (lycoris_suffix,)):
                out[k] = (out[k].to(torch.float32) * multiplier).to(out[k].dtype)
                hit = True
        if not hit:
            return dict(mod_keys)
        # Two ways to say it, and the choice is not cosmetic.
        #
        # sentinel=True writes Fizgig's >=1e6 marker, meaning "the scale is
        # already inside w1, do not apply alpha again". Fizgig's own loader reads
        # it that way. A plain LyCORIS loader that multiplies by alpha instead
        # would apply 1e6 -- a destroyed model with every slider still reading
        # 1.0, which is exactly the reported "distortion at zero settings".
        #
        # sentinel=False (the default) divides the multiplier back out of the
        # would-be alpha instead, so the file keeps a normal small alpha and every
        # loader agrees. Nothing is lost: w1 already carries the multiplier.
        if sentinel:
            _set_alpha(alpha_sentinel)
        else:
            current = _alpha_key(out)
            if current is not None:
                try:
                    base = float(out[current].reshape(-1)[0])
                except Exception:
                    base = 1.0
                # alpha = 1 keeps the LyCORIS scale at 1, which is what the
                # absorbed multiplier wants.
                out[current] = out[current] * 0.0 + 1.0
        return out

    up_key = None
    for k in out:
        if _is_factor_suffix(k, _UP_SUFFIXES):
            up_key = k
            break
    if up_key is None:
        return dict(out)                       # not a factor we know: pass through
    if abs(multiplier - 1.0) < 1e-12:
        return dict(mod_keys)

    dtype = out[up_key].dtype
    for k in list(out):
        if any(sfx in k for sfx in ("lora_down", "lora_A", "lora_up", "lora_B",
                                    "lora_down", "lora_emb")):
            out[k] = out[k].to(dtype)
    out[up_key] = (out[up_key].to(torch.float32) * multiplier).to(dtype)

    # alpha = rank makes the file's own scale 1.0 (bake.py), so it loads at
    # strength 1.0 and the sliders mean what the preview showed.
    rank = _rank_of(out, up_key)
    if rank:
        _set_alpha(rank)
    return out


class FizgigLoraSave:
    """Bake a chosen Explorer variant into a .safetensors file.

    This is Fizgig's Repair Studio save. The slider state becomes a real file,
    loadable by ComfyUI's stock LoraLoader at strength 1.0, and the original is
    never rewritten -- the bake goes to the `fizgig_loras` folder this package
    registers, which FizgigLoraExplorer can then bootstrap from.

    Primary only. Fizgig's donor blending (rank-concatenating a second LoRA into
    the same blocks) is deliberately not carried over: it is a different feature
    from "save the variant I just picked", and it needs the SVD +
    LyCORIS-materialize path.
    """

    CATEGORY = "Fizgig/Explorer"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Writes the per-block multipliers of a chosen Explorer variant "
                   "into a new .safetensors. Load the result with a normal LoRA "
                   "loader at strength 1.0, or point the Explorer's bootstrap_lora "
                   "at it to keep rolling from where you left off.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                # forceInput: a map you paste by hand is a map you pasted wrong.
                "block_map": ("STRING", {"default": "", "forceInput": True,
                                         "tooltip": "variant_N from the Explorer -- the one you picked."}),
                "filename": ("STRING", {"default": "",
                                        "tooltip": "Empty = <lora>_explorer_seed<seed>.safetensors"}),
                "overwrite": ("BOOLEAN", {"default": False,
                                          "tooltip": "Off: a name collision gets a short suffix instead "
                                                     "of overwriting."}),
            },
            "optional": {
                "family": (FAMILIES, {"default": "auto"}),
                "fuse_lycoris": ("BOOLEAN", {"default": True,
                                             "tooltip": "Bake the multiplier into a LoKR/LoHa's w1 "
                                                        "(native format, no SVD). Off = refuse those."}),
                # Off by default: the sentinel is Fizgig's own convention and a
                # loader that does not read it as "already applied" multiplies by
                # 1e6. A plain alpha = 1 is read the same way everywhere.
                "sentinel_alpha": ("BOOLEAN", {"default": False,
                                               "tooltip": "LoKR/LoHa only. On: write Fizgig's >=1e6 alpha "
                                                          "marker ('scale already inside w1'). Off (default): "
                                                          "write a normal alpha = 1, which every LyCORIS loader "
                                                          "reads identically. Turn on only if your loader "
                                                          "expects the sentinel."}),
                "bake_strength": ("BOOLEAN", {"default": False,
                                              "tooltip": "Also fold the map's load strength into the file. "
                                                         "Off is Fizgig's behaviour: the file keeps its own "
                                                         "scale and is used at that strength."}),
            },
        }

    def save(self, lora_name, block_map, filename, overwrite,
             family="auto", fuse_lycoris=True, bake_strength=False,
             sentinel_alpha=False):
        from safetensors.torch import load_file, save_file

        path = _lora_path(lora_name)
        if not block_map.strip():
            raise ValueError("block_map is empty -- connect the variant you picked")
        state = SliderState.from_json(json.loads(block_map))

        detected = _resolve_family("auto", path)
        fam = (family if family != "auto" else "") or state.family or detected
        # Baking a foreign map writes a file whose blocks this LoRA does not
        # have: plausible-looking, and it does nothing. Refuse.
        if state.family and detected and state.family != detected:
            raise ValueError(
                f"the block map is {state.family} but {lora_name} reads as {detected} -- "
                "baking would write blocks this LoRA does not have")
        if state.family and fam != state.family:
            raise ValueError(
                f"the family widget says {fam} but the map is {state.family}; "
                "bake with the family the map was rolled for")

        sd = load_file(path)
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
        except Exception:
            metadata = {}

        modules = _group_by_module(sd)
        lm = float(state.primary_scale) if bake_strength else 1.0

        out: Dict = {}
        dropped, refused = set(), set()
        moved, untouched, passthrough = 0, 0, 0

        for mod_name, mod_keys in modules.items():
            bid = block_id_from_key(mod_name, family=fam)
            if bid is None or bid not in state.blocks:
                # Not a block this map knows (text encoder, `io`): pass through.
                passthrough += 1
                for suffix, tensor in mod_keys.items():
                    out[f"{mod_name}.{suffix}"] = tensor
                continue
            bs = state.blocks[bid]
            if not bs.primary_enabled or abs(float(bs.primary_strength)) < 1e-9:
                dropped.add(bid)
                continue
            mult = float(bs.primary_strength) * lm
            baked = _bake_module(mod_keys, mult, fuse_lycoris, sentinel=sentinel_alpha)
            if baked is None:
                refused.add(bid)
                for suffix, tensor in mod_keys.items():
                    out[f"{mod_name}.{suffix}"] = tensor
                continue
            if abs(mult - 1.0) < 1e-12:
                untouched += 1
            else:
                moved += 1
            for suffix, tensor in baked.items():
                out[f"{mod_name}.{suffix}"] = tensor

        if not out:
            raise ValueError("nothing to write: every block in this map was dropped or refused")

        metadata.update({
            "fizgig.source_lora": os.path.basename(path),
            "fizgig.explorer_family": fam,
            "fizgig.explorer_blocks_moved": str(len([b for b, bs in state.blocks.items()
                                                     if bs.primary_enabled
                                                     and abs(bs.primary_strength - 1.0) > 1e-9])),
            "fizgig.explorer_blocks_off": str(len(dropped)),
            "fizgig.explorer_seed": str(state.seed),
            "fizgig.load_strength_baked": "1" if bake_strength else "0",
        })

        if not filename.strip():
            filename = (f"{os.path.splitext(os.path.basename(path))[0]}"
                        f"_explorer_seed{state.seed}.safetensors")
        if not filename.lower().endswith(".safetensors"):
            filename += ".safetensors"
        out_dir = _bake_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path) and not overwrite:
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"

        save_file(out, out_path, metadata=metadata)

        report = (f"saved: {out_path}\n"
                  f"  source: {os.path.basename(path)}  |  family {fam}\n"
                  f"  tensors written: {len(out)}  |  modules moved: {moved}"
                  f"  |  untouched (x1.0): {untouched}"
                  f"  |  passed through: {passthrough}\n"
                  f"  blocks off (dropped): {len(dropped)}"
                  + (f"  |  REFUSED (GLoRA/Tucker): {', '.join(sorted(refused)[:6])}"
                     if refused else "")
                  + (f"\n  load strength {lm:g} baked in" if bake_strength and lm != 1.0
                     else "\n  load this file at strength 1.0")
                  + f"\n  roll from it: state_source '{STATE_BOOTSTRAP}' + bootstrap_lora")
        print(f"[FizgigLoraSave] {lora_name}\n{report}", flush=True)
        if refused:
            logger.warning("FizgigLoraSave: %s -- %d block(s) refused; written at their "
                           "original weights", lora_name, len(refused))
        return (out_path,)


def _bake_dir() -> str:
    """Where bakes go: the `fizgig_loras` folder __init__ registered, so the
    Explorer sees them by name. Falls back to output/fizgig."""
    try:
        paths = folder_paths.get_folder_paths("fizgig_loras")
        if paths:
            return paths[0]
    except Exception:
        pass
    base = (folder_paths.get_output_directory()
            if hasattr(folder_paths, "get_output_directory") else ".")
    return os.path.join(base, "fizgig")


# ---------------------------------------------------------------------------
# 5. LoRA Royale's checkpoint scanner
# ---------------------------------------------------------------------------

class FizgigCheckpointScan:
    """List LoRA checkpoints in a folder, the way LoRA Royale does."""

    CATEGORY = "Fizgig/Explorer"
    FUNCTION = "scan"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("paths", "labels", "count")
    DESCRIPTION = ("Ported from lora_royale/scan.py. One clean epoch run gets "
                   "integer epoch labels; anything else is labelled by filename.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": "", "tooltip": "Absolute path to scan."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
        }}

    def scan(self, folder, index):
        found = scan_checkpoints(folder)
        if not found:
            return ("", "", 0)
        paths = "\n".join(p for _, p in found)
        labels = "\n".join(str(l) for l, _ in found)
        return (paths, labels, len(found))


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FizgigLoraExplorer": FizgigLoraExplorer,
    "FizgigExplorerPick": FizgigExplorerPick,
    "FizgigLoraBlockLoader": FizgigLoraBlockLoader,
    "FizgigLoraSave": FizgigLoraSave,
    "FizgigCheckpointScan": FizgigCheckpointScan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FizgigLoraExplorer": "LoRA the Explorer (Fizgig)",
    "FizgigExplorerPick": "LoRA the Explorer - Pick Variant",
    "FizgigLoraBlockLoader": "LoRA Block Loader (Explorer)",
    "FizgigLoraSave": "LoRA Save (Explorer)",
    "FizgigCheckpointScan": "Checkpoint Scan (LoRA Royale)",
}
