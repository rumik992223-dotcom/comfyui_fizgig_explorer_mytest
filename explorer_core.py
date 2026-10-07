"""LoRA the Explorer -- evolutionary per-block LoRA discovery.

Ported from Fizgig (https://github.com/shootthesound/Fizgig), Apache-2.0.
Source of truth in the original:

  src/fizgig/repair_studio/state.py      BlockState / SliderState.mutate / diff_blocks
  src/fizgig/lora_royale/scan.py         checkpoint discovery
  lora_trainer_gui.py  ~L15133          _explorer_generate_baseline_and_roll (the 4-variant recipe)
  src/fizgig/repair_studio/bake.py       _block_id_from_key (model-specific key layouts)
  src/fizgig/families/qwen_image.py      Qwen Image 2.1 key template + block ids

What was dropped, and why: the tkinter tab (20k+ lines of widgets), the render
pipeline (`families/workbench.py` builds its own DiT/VAE/TE pipeline) and the
bake (`bake.save_repaired_lora` writes a .safetensors). In ComfyUI those three
roles belong to KSampler, the standard LoRA loaders and the LoraSave node --
and four variants are four branches of one graph, not four threads.

What is kept is the part that is actually Fizgig's: the mutation recipe, the
per-family block map, the anchor/protection rules and the checkpoint scanner.
This module is stdlib-only and side-effect free, so it is trivially testable.
"""

import json
import os
import random
import re
import struct
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Block model (mirrors repair_studio/state.py)
# ---------------------------------------------------------------------------

# Fizgig clamps every per-block slider to [-3, 3] (state.py, SliderState.mutate).
STRENGTH_MIN = -3.0
STRENGTH_MAX = 3.0

# The anchor is never disabled -- only inverted or pushed to an extreme.
# Klein 9B: double_0 (state.py mutate() docstring). Krea 2: block_0. Qwen
# Image 2.1 shares Krea 2's `block_N` namespace; its Composition presets leave
# the identity blocks alone, so block_0 is the anchor there too.
ANCHORS = {
    "klein9b": "double_0",
    "krea2": "block_0",
    "qwen21": "block_0",
    "h3": "h3blk_0",
}

# Qwen Image 2.1 carries identity in a contiguous run of blocks. From
# families/qwen_image.py (measured 2 Oct 2026, Profiler group + single-block
# switch-offs on two character LoRAs, ArcFace vs the subjects' photos): blocks
# 10-14 alone give 67%/89% of the likeness; leaving them out removes 82%/84%,
# block 12 the strongest. Reported by the node so you can lock them if you want
# the picture to move without the face mutating -- Fizgig itself does not treat
# them specially in the Explorer, and neither does this port.
IDENTITY_BLOCKS = {
    "qwen21": frozenset(f"block_{i}" for i in range(10, 15)),
}


@dataclass
class BlockState:
    """One per-block slider pair. Donor fields are carried for parity with
    Fizgig's Repair Studio; the Explorer itself only touches primary_*."""

    primary_enabled: bool = True
    primary_strength: float = 1.0
    donor_enabled: bool = True
    donor_strength: float = 0.0


@dataclass
class SliderState:
    """The complete configuration at a point in time -- the sole input to a render.

    `blocks` is keyed by block id (double_0, single_23, block_7, h3blk_7, ...).
    The remaining fields are the preview parameters; keyframes/references/
    tensors from the original are deliberately not carried over (presets never
    serialise them there either)."""

    blocks: Dict[str, BlockState] = field(default_factory=dict)
    seed: int = 42
    prompt: str = ""
    preview_width: int = 512
    preview_height: int = 512
    primary_scale: float = 1.0
    donor_scale: float = 1.0
    # Which family's block naming this state uses (klein9b / krea2 / qwen21 /
    # h3). The block ids ARE the family's namespace, so the state has to carry
    # it: `block_4` means Krea 2 or Qwen, `h3blk_4` means MiniMax H3, and a map
    # applied under the wrong one scales nothing at all. Empty on states written
    # by an older version of this port.
    family: str = ""

    # -- construction -------------------------------------------------------

    @classmethod
    def from_block_ids(cls, block_ids, **kw) -> "SliderState":
        return cls(blocks={b: BlockState() for b in block_ids}, **kw)

    def copy(self) -> "SliderState":
        """Fast deep copy without JSON serialisation (state.py:119)."""
        return SliderState(
            blocks={bid: BlockState(bs.primary_enabled, bs.primary_strength,
                                    bs.donor_enabled, bs.donor_strength)
                    for bid, bs in self.blocks.items()},
            seed=self.seed, prompt=self.prompt,
            preview_width=self.preview_width, preview_height=self.preview_height,
            primary_scale=self.primary_scale, donor_scale=self.donor_scale,
            family=self.family,
        )

    # -- the Explorer's one real operation ----------------------------------

    def mutate(self, active_blocks, num_mutations: int = 3,
               intensity: float = 0.5, structure: float = 1.0,
               anchor: str = "double_0", rng: Optional[random.Random] = None,
               protected_blocks=()) -> "SliderState":
        """Perturb `num_mutations` blocks. Verbatim port of state.py:134.

        active_blocks    : block ids the loaded LoRA touches (others are skipped)
        intensity        : 0.0 = no change at all, 1.0 = bold moves (+-3.0).
                           Fizgig's floor of +-0.2 at 0.0 is dropped on purpose,
                           so zero can be used to check the wiring.
        structure        : 0.0 = no structural change, 1.0 = full (invert/extreme)
        anchor           : composition block; gets the structural change first
                           and is never disabled
        protected_blocks : variant 4's exclusion set (last pick's changed blocks)

        `rng` is the only addition to the original -- Fizgig seeds the module
        global, a node has to stay reproducible across re-executions.
        """
        rng = rng or random.Random()
        new = self.copy()
        candidates = [bid for bid in new.blocks
                      if bid in active_blocks and bid not in protected_blocks]
        if not candidates:
            candidates = [bid for bid in new.blocks if bid in active_blocks]
        if not candidates:
            return new

        n = min(num_mutations, len(candidates))
        if structure > 0.05 and anchor in candidates:
            others = [bid for bid in candidates if bid != anchor]
            chosen = [anchor] + rng.sample(others, min(max(n - 1, 0), len(others)))
        else:
            chosen = rng.sample(candidates, n)

        for i, bid in enumerate(chosen):
            bs = new.blocks[bid]
            # Fizgig's floor is 0.2, so its intensity 0 still moves every chosen
            # block by +-0.2. That is fine with a live preview next to a button,
            # but in a node it means "no change" is impossible -- so a user cannot
            # tell a wiring problem from a mutation. Zero means zero here; the
            # 0.2 floor is kept for any positive intensity, leaving the recipe
            # unchanged everywhere it mattered in the GUI (Fizgig's own refine
            # profile is 0.25, and its exploration default 0.964).
            magnitude = 0.0 if intensity <= 0.0 else 0.2 + intensity * 2.8
            if i == 0 and intensity > 0.3 and structure > 0.05:
                structural = rng.choice(["invert", "extreme"])
                if structural == "invert":
                    bs.primary_strength = _clamp(
                        bs.primary_strength * (1.0 - 2.0 * structure))
                else:
                    target = rng.choice([STRENGTH_MIN, STRENGTH_MAX])
                    bs.primary_strength = _clamp(
                        bs.primary_strength + (target - bs.primary_strength) * structure)
            else:
                if magnitude <= 0.0:
                    continue            # intensity 0: leave the block exactly as it is
                delta = rng.uniform(-magnitude, magnitude)
                bs.primary_strength = _clamp(bs.primary_strength + delta)
                if rng.random() < 0.35 * intensity:
                    bs.primary_enabled = not bs.primary_enabled
        return new

    def diff_blocks(self, other: "SliderState") -> List[str]:
        """Blocks whose state differs -- feeds variant 4's protection set."""
        changed = []
        for bid, bs in self.blocks.items():
            ob = other.blocks.get(bid)
            if ob is None or bs != ob:
                changed.append(bid)
        return changed

    # -- serialisation ------------------------------------------------------

    def to_json(self) -> Dict:
        return {"blocks": {b: asdict(bs) for b, bs in self.blocks.items()},
                "seed": self.seed, "prompt": self.prompt,
                "preview_width": self.preview_width,
                "preview_height": self.preview_height,
                "primary_scale": self.primary_scale, "donor_scale": self.donor_scale,
                "family": self.family}

    @classmethod
    def from_json(cls, d: Dict) -> "SliderState":
        raw = d.get("blocks") or {}
        blocks = {bid: BlockState(**bs) for bid, bs in raw.items()}
        return cls(blocks=blocks, seed=int(d.get("seed", 42)),
                   prompt=str(d.get("prompt", "")),
                   preview_width=int(d.get("preview_width", 512)),
                   preview_height=int(d.get("preview_height", 512)),
                   primary_scale=float(d.get("primary_scale", 1.0)),
                   donor_scale=float(d.get("donor_scale", 1.0)),
                   family=str(d.get("family", "")))

    def to_block_set(self) -> Dict[str, float]:
        """The Explorer's output in the form ComfyUI's per-block LoRA samplers
        expect: {block_id: multiplier}, disabled blocks omitted (0.0)."""
        return {bid: (float(bs.primary_strength) if bs.primary_enabled else 0.0)
                for bid, bs in self.blocks.items()}

    def to_plain_text(self, family: str = "") -> str:
        """Read-only readout, same shape as the original's state textbox.
        On Qwen Image 2.1 the identity blocks are tagged -- see IDENTITY_BLOCKS."""
        identity = IDENTITY_BLOCKS.get(family, frozenset())
        lines = []
        for bid in sorted(self.blocks, key=block_sort_key):
            bs = self.blocks[bid]
            if not bs.primary_enabled or bs.primary_strength != 1.0:
                en = "ON" if bs.primary_enabled else "OFF"
                tag = "  [identity]" if bid in identity else ""
                lines.append(f"{bid}: {en} @ {bs.primary_strength:+.2f}{tag}")
        if not lines:
            lines = ["All blocks at default (1.0)"]
        if abs(self.primary_scale - 1.0) > 1e-9:
            lines.insert(0, f"Load strength {self.primary_scale:g} "
                            "(every block relative to it; the file keeps its scale)")
        return "\n".join(lines)


def _clamp(v: float) -> float:
    return max(STRENGTH_MIN, min(STRENGTH_MAX, v))


def block_sort_key(b) -> Tuple[str, int]:
    """Stable ordering across families. Klein ids are <prefix>_<n> (double_0),
    Krea 2 / Qwen ids are block_<n>, H3 adds h3blk_<n> and h3_rf_<n>, and Krea 2
    also has <prefix>_<prefix>_<n> (txt_lw_0) plus the odd non-numeric `io` --
    a naive int(b.split('_')[1]) crashes on txt_lw_0. Same fix as the GUI's
    _explorer_block_sort_key."""
    parts = str(b).split("_")
    try:
        return ("_".join(parts[:-1]), int(parts[-1]))
    except ValueError:
        return (str(b), 0)


# ---------------------------------------------------------------------------
# The four-variant recipe (lora_trainer_gui.py ~L15133)
# ---------------------------------------------------------------------------

# (uses_structural, protects_last_pick). Variants 1-2 get the structural
# treatment, 3 is pure random, 4 avoids the blocks the last pick touched.
VARIANT_RECIPE = ((True, False), (True, False), (False, False), (False, True))


def roll_variants(baseline: SliderState, active_blocks, *, num_mutations: int = 5,
                  intensity: float = 0.5, structure: float = 1.0,
                  anchor: str = "double_0", last_pick_blocks=(),
                  seed: int = 0, locked_blocks=()) -> List[SliderState]:
    """The Explorer's `Roll variants` button, as a pure function.

    Returns exactly four SliderStates rolled from `baseline`. The anchor is
    force-added to the active set (never disabled) unless explicitly frozen,
    which is what the GUI does before calling mutate()."""
    active = set(active_blocks) - set(locked_blocks)
    if anchor not in set(locked_blocks):
        active.add(anchor)
    protected = set(last_pick_blocks) & active
    if len(active - protected) < 2:
        protected = set()

    out = []
    for vi, (use_structure, use_protect) in enumerate(VARIANT_RECIPE):
        rng = random.Random((int(seed) & 0xFFFFFFFF) * 4 + vi)
        out.append(baseline.mutate(
            active,
            num_mutations=num_mutations,
            intensity=intensity,
            structure=(structure if use_structure else 0.0),
            anchor=anchor,
            rng=rng,
            protected_blocks=(protected if use_protect else ()),
        ))
    return out


# ---------------------------------------------------------------------------
# Block discovery from a LoRA file's header (replaces engine.adapter_blocks)
# ---------------------------------------------------------------------------

# Every layout Fizgig's bake.py:_block_id_from_key knows, plus Qwen Image 2.1's
# native keys (families/qwen_image.py: kohya=False, so its files keep
# `transformer.transformer_blocks.<n>.<module>.lora_A/B.weight`).
#
# Order matters: double/single and the txtfusion/token_refiner namespaces are
# checked before the bare `blocks_<n>` shape, which would otherwise swallow them
# (in `double_blocks.0.` the substring `blocks.0.` matches on its own).
_BLOCK_KEY_RE = re.compile(r"(?:lora_unet_)?(double_blocks|single_blocks)[._](\d+)[._]")
_KREA2_TXT_KEY_RE = re.compile(r"txtfusion_(layerwise|refiner)_blocks[._](\d+)[._]")
_H3_REFINER_KEY_RE = re.compile(r"token_refiner_blocks[._](\d+)[._]")
# Covers lora_unet_blocks_0_ (Krea 2 / H3 / Qwen after ComfyUI normalises them)
# and transformer.transformer_blocks.0. (Qwen as Fizgig writes it).
_MAIN_BLOCKS_KEY_RE = re.compile(r"(?:lora_unet_|transformer_)?blocks[._](\d+)[._]")

# A run of >= 40 main blocks is MiniMax H3 (50 main blocks); Krea 2 and Qwen
# Image 2.1 both have 32.
_H3_BLOCK_HINT_MIN = 40


def read_safetensors_keys(path: str) -> Tuple[List[str], Dict[str, str]]:
    """Header-only read: key names + metadata, no tensors touched and no torch.

    safetensors layout is [8-byte LE header length][JSON header][bulk data].
    Fine for a 500 MB LoRA -- we read the first few KB and stop."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return [], {}
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 100_000_000:
            return [], {}
        header = json.loads(f.read(n).decode("utf-8"))
    keys = [k for k in header if k != "__metadata__"]
    meta = {str(k): str(v) for k, v in (header.get("__metadata__") or {}).items()}
    return keys, meta


def family_from_metadata(meta: Dict[str, str]) -> Optional[str]:
    """Fizgig names the base model in the LoRA's metadata. Read that before
    guessing from key shapes -- Qwen, Krea 2 and H3 are indistinguishable by
    shape alone once ComfyUI has normalised their keys."""
    blob = " ".join(f"{k}={v}" for k, v in (meta or {}).items()).lower()
    if not blob:
        return None
    # Explicit model names first, and H3 only on its own token. "h3" is two
    # characters: as a bare substring it matched a hash, a filename or a trainer's
    # comment ("run on 8x h3 nodes") and labelled a Qwen file as H3 -- and because
    # it was tested BEFORE "qwen", an explicit Qwen name could not correct it.
    # A real H3 file says "MiniMax H3"; a real Qwen file says "qwen".
    if "qwen" in blob:
        return "qwen21"
    if "krea" in blob:
        return "krea2"
    if "klein" in blob or "flux" in blob:
        return "klein9b"
    if "minimax" in blob or re.search(r"(?<![a-z0-9])h3(?![a-z0-9])", blob):
        return "h3"
    return None


def block_id_from_key(key: str, family: str = "") -> Optional[str]:
    """Map one LoRA tensor key to a Fizgig block id, or None.

    `family` disambiguates the bare `blocks_<n>` shape, which Krea 2, Qwen
    Image 2.1 and MiniMax H3 all produce (block_N vs h3blk_N)."""
    m = _BLOCK_KEY_RE.search(key)
    if m:
        return f"{m.group(1).replace('_blocks', '')}_{int(m.group(2))}"
    t = _KREA2_TXT_KEY_RE.search(key)
    if t:
        return f"txt_{'lw' if t.group(1) == 'layerwise' else 'rf'}_{int(t.group(2))}"
    r = _H3_REFINER_KEY_RE.search(key)
    if r:
        return f"h3_rf_{int(r.group(1))}"
    m = _MAIN_BLOCKS_KEY_RE.search(key)
    if m:
        n = int(m.group(1))
        return f"h3blk_{n}" if family == "h3" else f"block_{n}"
    return None


def scan_block_ids(path: str, family: Optional[str] = None) -> List[str]:
    """The block ids a LoRA actually adapts -- the equivalent of
    WorkbenchEngine.primary_block_ids, derived from the file itself.

    This is what makes the port self-contained: Fizgig gets the set from a live
    engine it already built (workbench.py:205); here we read the header.

    Family resolution order: explicit argument, file metadata, then the H3
    block-count heuristic. Getting this wrong mislabels Qwen/Krea 2 blocks as
    H3 ones, so it is worth the lookup."""
    keys, meta = read_safetensors_keys(path)
    fam = family or family_from_metadata(meta) or ""
    if not fam:
        highest = max((int(m.group(1)) for m in
                       (_MAIN_BLOCKS_KEY_RE.search(k) for k in keys) if m),
                      default=-1)
        if highest >= _H3_BLOCK_HINT_MIN:
            fam = "h3"
    ids = set()
    for k in keys:
        bid = block_id_from_key(k, family=fam)
        if bid:
            ids.add(bid)
    return sorted(ids, key=block_sort_key)


def guess_family(path: str) -> str:
    """Best-effort family, from the file's own key shapes.

    This must NOT go through scan_block_ids(): that function applies the H3
    block-count heuristic and namespaces bare `blocks_<n>` keys as `h3blk_<n>`,
    so asking it and then testing for `h3blk_` prefix confirms the guess it just
    made. That circularity is how a 32-block Qwen file was labelled H3, and an
    H3 map then silently applied to it -- the sliders scale the wrong blocks and
    the picture distorts while every counter still reports success.

    So: classify by the raw keys, before any namespace is assigned.
    """
    keys, meta = read_safetensors_keys(path)
    fam = family_from_metadata(meta)
    if fam:
        return fam

    blob = "\n".join(keys)
    # Unambiguous namespaces first.
    if "double_blocks" in blob or "single_blocks" in blob:
        return "klein9b"
    if "token_refiner_blocks" in blob:
        return "h3"
    if "txtfusion_" in blob:
        return "krea2"
    # Qwen writes diffusers-style keys with lora_A/lora_B and transformer_blocks.
    if "transformer_blocks." in blob and re.search(r"[._](lora_A|lora_B)[._]", blob):
        return "qwen21"
    # Bare `lora_unet_blocks_<n>_` is shared by Krea 2, Qwen and H3: only the
    # block count can separate them, and 32 is Krea 2 or Qwen while 50 is H3.
    nums = [int(m.group(1)) for m in
            (_MAIN_BLOCKS_KEY_RE.search(k) for k in keys) if m]
    if nums:
        return "h3" if max(nums) >= _H3_BLOCK_HINT_MIN else "krea2"
    return "krea2"          # nothing block-shaped: a caller will report it


def key_shape(key: str) -> str:
    """A tensor key with its digits stripped -- for error messages. Turning
    `transformer.transformer_blocks.0.attn.to_q.lora_B.weight` into
    `transformer.transformer_blocks.N.attn.to_q.lora_B.weight` says which
    layout a file uses without dumping a hundred keys."""
    return re.sub(r"\d+", "N", str(key))


def summarise_key_shapes(keys, limit: int = 4) -> List[str]:
    """The distinct key shapes in a file, most common first."""
    counts: Dict[str, int] = {}
    for k in keys:
        s = key_shape(k)
        counts[s] = counts.get(s, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in ranked[:limit]]


# ---------------------------------------------------------------------------
# Reading a baked LoRA back into a state
# ---------------------------------------------------------------------------

# A bake (Fizgig's save_repaired_lora, or ComfyUI's save-LoRA node) absorbs each
# block's multiplier into the module's first factor and sets alpha so the file's
# own scale is 1.0. The multiplier is therefore a RATIO of that factor against
# the file it was baked from -- a raw norm is a magnitude, and says nothing on
# its own. That is why a reference is required here rather than optional.
_FACTOR_SUFFIXES = ("lora_up.weight", "lora_B.weight", "lora_emb.weight",
                    "lokr_w1_a", "lokr_w1", "hada_w1_a", "hada_w1")


def _fac_norm(tensor) -> Optional[float]:
    """Frobenius norm of one tensor, on the CPU, never kept."""
    try:
        t = tensor
        if hasattr(t, "float"):
            t = t.float()
        if hasattr(t, "cpu"):
            t = t.cpu()
        n = t.norm() if hasattr(t, "norm") else None
        if n is None:                                            # e.g. numpy
            import numpy as _np
            return float(_np.linalg.norm(_np.asarray(t)))
        return float(n.item()) if hasattr(n, "item") else float(n)
    except Exception:                                            # pragma: no cover
        return None


def factor_norms(sd, block_ids, family: str = "") -> Dict[str, List[float]]:
    """{block_id: [norm of each first factor]} for the blocks in `block_ids`."""
    out: Dict[str, List[float]] = {}
    for key, tensor in sd.items():
        if not any(key.endswith(sep + suf) for suf in _FACTOR_SUFFIXES
                   for sep in (".", "_")):
            continue
        bid = block_id_from_key(key, family=family)
        if bid is None or bid not in block_ids:
            continue
        n = _fac_norm(tensor)
        if n is not None:
            out.setdefault(bid, []).append(n)
    for v in out.values():
        v.sort()
    return out


def _median(values) -> Optional[float]:
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid] if len(values) % 2
                 else 0.5 * (values[mid - 1] + values[mid]))


def strength_from_pair(baked, reference, block_ids, family: str = "") -> Dict[str, float]:
    """Recover per-block multipliers from a baked LoRA, against its source.

    `baked` and `reference` are {key: tensor}; only the first factor of each
    module is measured. Skips a block whose factor count differs between the two
    files -- a shape change means the comparison would not be honest.
    """
    ref = factor_norms(reference, block_ids, family=family)
    bak = factor_norms(baked, block_ids, family=family)
    out: Dict[str, float] = {}
    for bid, bak_norms in bak.items():
        ref_norms = ref.get(bid)
        if not ref_norms or len(ref_norms) != len(bak_norms):
            continue
        ratios = sorted(b / r for b, r in zip(bak_norms, ref_norms) if r > 1e-12)
        m = _median(ratios)
        if m is not None:
            out[bid] = round(m, 6)
    return out


def state_from_baked(baked, reference, block_ids, *, family: str = "",
                     **state_kw) -> "SliderState":
    """A SliderState rebuilt from a baked LoRA plus the file it came from.

    A block the bake dropped (disabled blocks are dropped, not zeroed --
    bake.py) comes back at 1.0 with primary_enabled False, which is what it was.
    """
    strengths = strength_from_pair(baked, reference, set(block_ids), family=family)
    ref_blocks = set(factor_norms(reference, set(block_ids), family=family))
    blocks = {}
    for bid in block_ids:
        if bid in strengths:
            value = strengths[bid]
            if abs(value) <= 1e-9:
                blocks[bid] = BlockState(primary_enabled=False)
            else:
                blocks[bid] = BlockState(primary_strength=value)
        elif bid in ref_blocks:
            blocks[bid] = BlockState(primary_enabled=False)      # dropped by the bake
        else:
            blocks[bid] = BlockState()
    return SliderState(blocks=blocks, family=family, **state_kw)


# ---------------------------------------------------------------------------
# Checkpoint discovery (ported from lora_royale/scan.py)
# ---------------------------------------------------------------------------

# Trainer epoch checkpoints: "<name>-000005.safetensors"
_EPOCH_RE = re.compile(r"^(?P<name>.+)-(?P<epoch>\d{6})\.safetensors$")


def scan_checkpoints(folder: str) -> List[Tuple]:
    """Find LoRA checkpoints in a directory, sorted ascending.

    Verbatim behaviour of LoRA Royale's scanner:
      - one clean epoch run and nothing else -> labels are the epoch integers
      - anything else -> every .safetensors, labelled by filename stem
    State dirs (<name>-NNNNNN-state/) and non-safetensors are ignored."""
    if not folder or not os.path.isdir(folder):
        return []
    by_run, others = {}, []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".safetensors"):
            continue
        full = os.path.join(folder, fn)
        if not os.path.isfile(full):
            continue
        m = _EPOCH_RE.match(fn)
        if m:
            by_run.setdefault(m.group("name"), []).append((int(m.group("epoch")), full))
        else:
            others.append(full)

    if len(by_run) == 1 and not others:
        return sorted(next(iter(by_run.values())), key=lambda t: t[0])

    all_paths = [p for run in by_run.values() for _, p in run] + others
    all_paths = sorted(set(all_paths), key=lambda p: os.path.basename(p).lower())
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in all_paths]
