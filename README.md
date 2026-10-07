# comfyui_fizgig_explorer

**LoRA the Explorer**, ported from [Fizgig](https://github.com/shootthesound/Fizgig)
(Apache-2.0) to ComfyUI. Evolutionary per-block LoRA discovery: roll four mutated
variants, render them, pick the one you like, roll again.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> comfyui_fizgig_explorer
# no dependencies beyond ComfyUI itself
```

Restart ComfyUI. Nodes appear under **Fizgig/Explorer**.

## Supported families and key layouts

| Family | Blocks | Key layout the Explorer reads | Block ids |
|---|---|---|---|
| Klein 9B | 8 double + 24 single | `lora_unet_double_blocks_0_…`, `lora_unet_single_blocks_23_…` | `double_0`, `single_23` |
| Klein 9B (diffusers) | as above | `diffusion_model.double_blocks.0.…` | as above |
| Krea 2 | 32 + text fusion | `lora_unet_blocks_4_…`, `lora_unet_txtfusion_layerwise_blocks_1_…` | `block_4`, `txt_lw_1` |
| **Qwen Image 2.1** | 32 | `transformer.transformer_blocks.4.attn.to_q.lora_B.weight` | `block_4` |
| MiniMax H3 | 50 main + 2 refiner | `lora_unet_blocks_7_…`, `lora_unet_token_refiner_blocks_1_…` | `h3blk_7`, `h3_rf_1` |

Qwen, Krea 2 and H3 all collapse to the same bare `blocks_<n>` shape once ComfyUI
normalises a file. The family is therefore read from the LoRA's **metadata** first
(`ss_sd_model_name` and friends), then classified from the raw key shapes — never
from the block ids, because naming those ids is exactly the decision being made.

### Two families can share a block count

Qwen Image 2.1 and Krea 2 both have 32. So a block map from the wrong family does
**not** scale nothing — it scales the *wrong blocks*: every module gets a multiplier
meant for a different one, all the counters look healthy, and the render is distorted
while staying recognisable. That is worse than a clean failure, so it is a **hard
error**, not a warning:

```
family mismatch: the block map is h3 but razd-qwen21-V5.safetensors reads as qwen21.
  The map holds 32 blocks in h3 naming (e.g. h3blk_0); applying them would scale the
  wrong blocks -- the picture distorts while every counter reports success.
  Clear roll_state for a fresh qwen21 baseline, or roll this LoRA from its own Explorer node.
```

`roll_state` is a widget, so it persists in the workflow: a map left over from a
previous session is the usual cause. The `family` widget is a deliberate override —
set it only when you know the file better than its keys do, and get the mismatch
error if you are wrong.

## What was ported, and what wasn't

Fizgig is a tkinter app that owns its own DiT/VAE/text-encoder pipeline. Three of
its jobs have no business being reimplemented inside a node:

| Fizgig | here |
|---|---|
| `families/workbench.py` `generate_preview()` — builds the pipeline, attaches the LoRA, samples | `KSampler` + the standard loaders |
| `repair_studio/bake.py` `save_repaired_lora()` — writes a .safetensors | the per-block weights are applied live, in memory |
| `lora_trainer_gui.py` L14597+ — the tab, undo stack, 4 previews, Tk `after()` handoff | four branches of one graph |

What *was* ported is the part that is actually the Explorer:

- `repair_studio/state.py` → `BlockState`, `SliderState.mutate()`, `diff_blocks()`, `copy()`, and the `[-3, 3]` clamp
- the four-variant recipe from `_explorer_generate_baseline_and_roll()`: variants 1–2 structural, 3 pure random, 4 excluding the last pick's changed blocks
- the anchor rule (Klein `double_0`, Krea 2 / Qwen `block_0`, H3 `h3blk_0`; never disabled — only inverted or pushed to an extreme)
- the Freeze set (blocks that never move)
- `lora_royale/scan.py` checkpoint discovery, verbatim
- the block-id key layouts from `bake.py:_block_id_from_key` plus `families/qwen_image.py`

Fizgig gets its block list from a live engine (`WorkbenchEngine.primary_block_ids`);
a node can't, so the port reads the LoRA's safetensors header instead — key names
and metadata only, no tensors, no torch. For a 500 MB LoRA that's the first few KB.

The only functional addition to `mutate()` is an injected `rng`. The original seeds
the module-global `random`; a node has to stay reproducible across re-executions.

### Qwen Image 2.1 and identity blocks

Blocks `10–14` carry the likeness on Qwen — from `families/qwen_image.py`, measured
with the Profiler on two character LoRAs: those five blocks alone give 67–89 % of the
likeness, and leaving them out removes 82–84 %. **The Explorer does not treat them
specially**, here or in Fizgig — a roll can hit them. If you want the picture to move
under a stable face, lock them:

```
locked_blocks = block_10,block_11,block_12,block_13,block_14
```

The `active_blocks` output lists them (`identity_blocks`) and the Pick readout tags
them, so you can see when a roll has touched one.

## Four variants look identical

The four variant states are always different — the tests assert the four JSON
payloads differ from each other and from the baseline. So identical images mean one
of these:

**1. Only one branch is rendering.** If each loader feeds its own KSampler, they need
distinct node ids, and your viewer must show all four (one PreviewImage each, or one
batch). These are four separate KSamplers — they do not take four inputs each.

**2. The MODEL reaching the sampler is not the loader's.** Every `KSampler` must have
its `model` wire coming out of *its own* `FizgigLoraBlockLoader`. Feeding them all
from one CheckpointLoader makes the loader a dead end: it still runs, prints its
report, and changes nothing.

**3. The LoRA patches nothing.** Look at the ComfyUI console after a run. If you get
`MODEL was NOT patched`, the LoRA adapts none of the loaded model's modules and *no*
strength can show. A Qwen LoRA on a Klein checkpoint is the usual case. Fix the
model/LoRA pairing, not the graph.

**4. The block map is for another family.** `roll_state` is a widget, so it persists
in the workflow. The loader now **refuses** the pair outright (see below) rather than
rendering something wrong.

**5. The LoRA is LyCORIS.** LoKR/LoHa files were skipped by earlier versions of this
port — you would see `tensors scaled: 0` and `LyCORIS keys left uniform: 392`. They
are scaled now (see below). Update the files.

**6. `roll_state` is wired but the loop was never actually run.** Leave `roll_state`
empty for the first roll. After that, wire `Pick.baseline` into it.

The console gets one line per roll, which is the other half of the answer:

```
[FizgigLoraExplorer] bonnie_qwen.safetensors  family=qwen21  anchor=block_0  blocks=32  rolling_from=fresh state
    variant 1: 5 changed -- block_0, block_16, block_20, block_5, block_6
    variant 2: 5 changed -- block_0, block_13, block_23, block_9, block_30
    variant 3: 5 changed -- block_2, block_14, block_20, block_21, block_27
    variant 4: 5 changed -- block_16, block_28, block_3, block_8, block_11
```

Four different block sets there plus four identical renders means the graph, not the
Explorer. Set `verbose` on the loader to print every block's multiplier and compare
the four branches' reports — identical reports mean identical maps, which means the
wiring is wrong.

## "Missing node type" / "Unknown pack" for every Fizgig node

That is not a workflow problem — it means the pack did not load, so ComfyUI
registered none of its nodes and the UI lists every one it finds in the file as
unknown.

The cause is almost always `nodes.py` and `explorer_core.py` being out of sync.
`nodes.py` imports helpers from `explorer_core.py`:

```python
from .explorer_core import (..., state_from_baked, strength_from_pair, ...)
```

Update `nodes.py` against an older `explorer_core.py` and that import raises, the
package's `__init__` fails, and every node disappears. As of 0.5.4 the package says
so instead of leaving you with a bare traceback:

```
comfyui_fizgig_explorer: nodes.py could not be imported, so NONE of this pack's
nodes are registered -- the UI will report every Fizgig node in your workflow as
a missing node type.
  ModuleNotFoundError: cannot import name 'state_from_baked' from ...
```

**Copy every `.py` from this package together**, not just the file that changed:

```
__init__.py  explorer_core.py  nodes.py  torch_stub.py  (the tests are optional)
```

Then restart ComfyUI fully — a browser refresh does not re-import a Python
module — and check the startup console for a traceback mentioning
`comfyui_fizgig_explorer`.

There is a test guarding this: it walks `nodes.py`'s imports, asserts each name
exists in `explorer_core.py`, imports the package with `folder_paths` stubbed and
checks all five classes register with a callable `FUNCTION`. Run it before copying
anything — it fails locally, which is far cheaper than debugging it in the UI:

```bash
python test_nodes.py
```

### A node id with a colon (`30:16`)

```
[WARNING] invalid prompt: {'type': 'missing_node_type',
 'details': "Node ID '#30:16'", 'extra_info': {'node_id': '30:16', 'class_type': None}}
```

An id of the form `30:16` is a node *inside a group node* (ComfyUI's newer
frontend nests subgraph definitions that way). `class_type: None` means the entry
in the prompt has no `type` at all — it is not a node this pack or any other can
resolve. It usually appears after a workflow has been saved while the pack was
failing to load: the frontend serialised a group whose members it could no longer
identify.

Recovery: load the workflow, **delete that group node**, and add
`LoRA the Explorer (Fizgig)` plus the four `LoRA Block Loader (Explorer)` nodes
fresh from the node menu. Re-wire them and save under a new name — do not
overwrite the broken file until the new graph queues successfully.

## Distorted even at zero intensity

Two separate causes, and the console line now names both.

### 1. The baseline is not fresh

`state_source = "picked variant"` starts from the `roll_state` text, and that text
carries every earlier generation's edits. The console reports the **delta** against
the baseline, so `1 changed` can still be a badly distorted render — the drift is in
the baseline, not the delta, and it was invisible. It is now printed:

```
[FizgigLoraExplorer] ... from=picked variant
    BASELINE IS NOT FRESH: 3 of 32 block(s) differ from 1.0 (the roll starts from an
      already-edited state)
      worst: block_2=+2.40, block_27=+2.20, block_16=+0.10
    variant 1: 1 changed -- block_9
```

If that block appears, the distortion is inherited. Set `state_source` to
**fresh state**, or clear `roll_state`, to start from every block at 1.0.

### 2. `intensity` 0 does not mean "no change"

Fizgig's magnitude is `0.2 + intensity * 2.8`, so **its** `intensity = 0` still moves
every chosen block by ±0.2 — there was no setting that left the file alone. That makes
"is the graph wired correctly?" impossible to answer by looking, and it is part of why
a "zeroed" roll still distorted.

In this port `0.0` means `0.0`: nothing moves, no block is toggled, and the four
variants come back byte-identical to the baseline. The 0.2 floor is kept for any
positive intensity, so the recipe is unchanged at the values the GUI actually used
(its refine profile is 0.25, its exploration default 0.964).

Note the floor is the **minimum span**, not a cap: at `intensity = 0.1` the span is
`0.2 + 0.1 × 2.8 = ±0.48`. Reading it as a cap is the mistake that made "intensity 0
is safe" look true.

### 3. The family is read as H3 for a Qwen file

```
[FizgigLoraExplorer] qwen2.1\razd-qwen21-V5.safetensors  family=h3
[INFO] Requested to load QwenImage21
```

The model is Qwen; the LoRA is being read as MiniMax H3. Both the map and the file
say `h3`, so the mismatch guard has nothing to compare and stays quiet — while every
block id is `h3blk_N` instead of `block_N`.

The family comes from the file's **metadata**, then from its key shapes. Two things
make the metadata path fail on a file that is plainly Qwen:

- `"h3"` was matched as a **bare substring**, so it fired on a hash, a filename or a
  trainer's comment (`"run on 8x h3 nodes"`) — and H3 was tested *before* `qwen`, so
  an explicit Qwen name could not correct it.
- A file saved without any model name in its metadata falls back to the block-count
  rule, where a run of ≥ 40 `blocks_N` means H3.

Both are fixed: explicit names (`qwen`, `krea`, `klein`/`flux`) are read first, and
`h3` only matches on its own token or on `minimax`. Check what this package reads:

```bash
python inspect_lora.py path/to/razd-qwen21-V5.safetensors
```

It prints the metadata, the naming style, tensors per block, the family this
package reads, and the ids it would use — plus what the ids would be under every
other family. Run it on the file, then set the `family` widget on the Explorer
**and** the loader to `qwen21` explicitly. Both must agree; the loader refuses a
disagreement between the widget and the map.

## Upgrading an existing workflow

Widgets are matched to a saved workflow **by position** and every `required` input must
be present. So a widget added after your workflow was saved must go at the **end** of
the inputs **and be optional** — otherwise ComfyUI fails before anything runs:

```
[ERROR] FizgigLoraExplorer 30:78:
[ERROR]   - Required input is missing: structural_variants
```

Putting it anywhere but the end is worse than an error: a positional shift feeds your
`seed` into the new widget and `locked_blocks` into `state_source`, so the node runs
happily with the wrong settings.

`structural_variants` therefore sits last, in `optional`, and defaults to off. An older
workflow loads unchanged and keeps its 13 original widget values. There are tests
asserting the original order, the optionality and the default, so this cannot regress
silently.

If a node still shows the error, it is a stale copy: restart ComfyUI so the new
`nodes.py` is read, then reload the workflow.

## Distortion with every slider at 1.0 (or at zero)

Two causes produce distortion that no slider value explains, and they are
distinguishable in one run.

### 1. The baked file's alpha (LoKR/LoHa only)

Fizgig marks an already-scaled LyCORIS module by writing `>= 1e6` into its alpha —
"the scale is already inside w1, do not apply alpha again". A loader that does not
read that convention multiplies by about a million. The pictures collapse and every
slider still reads 1.0, so it looks like nothing caused it.

`FizgigLoraSave` no longer writes the sentinel by default. It writes a normal
`alpha = 1`, which every LyCORIS loader reads identically — nothing is lost, `w1`
already carries the multiplier. Turn `sentinel_alpha` on only if your loader
demands the marker.

If you already saved LoRAs with the older build, re-save them: any file whose
`lokr_alpha` is `>= 1e6` is suspect. Check with:

```python
from safetensors import safe_open
with safe_open("your_bake.safetensors", framework="pt") as f:
    for k in f.keys():
        if k.endswith("alpha"):
            print(k, float(f.get_tensor(k).reshape(-1)[0]))
```

### 2. A baseline that has drifted

`state_source = bootstrap from LoRA weights` reads the multipliers back out of a
file as a *ratio* against `bootstrap_reference`. If the reference is itself a bake,
every round multiplies the previous multipliers in and the drift compounds — the
file gets stronger each generation while the widgets never move.

`bootstrap_reference` must be the **original** LoRA, and `bootstrap_lora` the bake.
If both point at the same file the read-back is all 1.0 (a fresh start) and the
loader says so on the console.

### The decisive test

Do this before anything else — it takes one run and separates the two:

1. Roll with every block **locked** (`locked_blocks` listing all of them) so the map
   is all 1.0 and nothing can move.
2. Render through the Explorer's loader. If that is already wrong, the mutation is
   irrelevant: the fault is in the scaling or in a previous bake, not in the recipe.
3. Then render the same LoRA through ComfyUI's stock `LoraLoader` at strength 1.0
   and compare against a render with no LoRA.

A flat map modifies **no tensor at all** — there is a test asserting it, byte for
byte, including the text-encoder keys. So a flat map that still looks wrong points
at the file or at a second loader in the graph, not at this node.

## Images distort badly ("mirror" warping)

A mutation stack is *supposed* to move the picture, but `invert`/`extreme` on the
anchor plus `±3.0` sliders is far past any LoRA's working range, and a Qwen
`block_0` inverted is a different composition, not a nudge. In Fizgig that is fine —
a button and a preview — but it is a bad first setting for four full renders.

For a usable first roll:

### The one mutation that is not a nudge

`state.py`'s structural branch is only reachable above `intensity > 0.3`, and at
`structure = 1.0` it drives the anchor block to `-1.0`:

```python
if i == 0 and intensity > 0.3 and structure > 0.05:
    bs.primary_strength = bs.primary_strength * (1.0 - 2.0 * structure)   # 1.0 -> -1.0
```

Negating the composition block **inverts the edit** on an edit LoRA — the warped,
mirror-like frame. The defaults no longer do this:

| widget | now | was | why |
|---|---|---|---|
| `intensity` | **0.25** | 0.5 | at or below the 0.3 gate, so a first roll cannot hit the anchor |
| `structure` | **0.0** | 1.0 | a plain mutation, no invert/extreme |
| `structural_variants` | **off** | — | the invert/extreme is opt-in, not the starting point |
| `mutations` | **3** | 5 | Fizgig's own refinement profile |

These match what Fizgig's own **Refine this baseline in Repair Studio** did —
`intensity 0.25`, `structure 0.15` — which is why the GUI never showed you the
warp: in that profile the `intensity > 0.3` gate never opens. The tab's *defaults*
were the exploration profile (`intensity 0.964`, `structure 1.0`, `mutations 8`),
and carrying those into a node that starts from a finished edit is what went wrong.

Turn `structural_variants` on to get them back deliberately; the console line then
marks the damage so you can see it before rendering:

```
    variant 1: 3 changed -- block_0, block_7, block_20  [ANCHOR INVERTED: block_0 -> -1.00]
```

With it off, all four variants are plain random mutations (different RNG streams),
and `structure` is ignored — the log says so if you left it raised.

### If all four branches are warped identically

Then it is not the mutation. The four branches share one upstream node, so a single
value reaching all four produces four copies of the same wrong image. Check, in order:

1. **Is the same block map on all four loaders?** Wires are easy to cross and one
   `variant_N` plugged into four loaders gives four identical renders.
2. **Is `load_strength` on the loader 1.0?** A non-1.0 value scales the whole file.
3. **Did the roll actually change anything?** The console prints each variant's
   changed-block list; if four lines read `-- NOTHING`, everything is locked.
4. **The family mismatch above** — it now raises instead of warning, precisely
   because it produces a picture that is wrong but not obviously broken.

The loader's report repeats the map's own contents, so the two places can be
compared:

```
[FizgigLoraBlockLoader] qwen2.1\Bonnie.safetensors
family: qwen21
blocks in map: 32  |  touched by this LoRA: 32   <- touched 0 means the two disagree
```

## Why the loader needs the LoRA as well

ComfyUI's `MODEL` socket carries only finished patches, not instructions — there is no
way to tell a KSampler "apply block 12 at 2.3". The stock `LoraLoader` applies one
strength to the whole file. So the node where a LoRA's weights and a block map meet
has to hold both:

- the **Explorer** reads the file's *header* — which blocks exist at all;
- the **loader** reads the *tensors* — to multiply the right ones.

Hence `lora_name` on both, and **the same file on both**. If the map and the file
disagree, the ids are never found, nothing is scaled, and all four branches render
identically while every node still reports success. The report shows both sides:

```
[FizgigLoraBlockLoader] qwen2.1\BonnieWright.safetensors
family: qwen21
blocks in map: 32  |  touched by this LoRA: 32
tensors scaled: 96  |  outside the map: 0  |  off: 0
LyCORIS first factors scaled: 96 (multiplier into w1 -- LoKR/LoHa are linear in it)
modules patchable by this LoRA: 196  |  blocks among them: 28
```

## Seeds: two different ones, and only one belongs here

The Explorer has a `seed` widget and **no** `control_after_generate`, deliberately:

- The Explorer's `seed` drives **the mutation RNG only** — which blocks move and how
  far. It does not touch sampler noise. Fizgig had the same split (the tab kept its
  own preview seed).
- It should stay **fixed** while you compare variants. If it advanced after every run
  you would be looking at fresh mutations each time and comparing nothing.
- `control_after_generate` belongs to the KSampler. The four branches must use **one
  and the same sampler seed** — otherwise you are comparing noise, not blocks. Wire
  all four from a single `PrimitiveInt` to make that impossible to get wrong. A test
  asserts the four roll states share a seed; the sampler's is yours to set.

Once a variant is chosen, advance the sampler seed to check the pick survives
different noise.

## The graph

```
FizgigLoraExplorer ──variant_1──> FizgigLoraBlockLoader ──MODEL──> KSampler ──> PreviewImage
        ▲          ──variant_2──>      (4 branches:
        │          ──variant_3──>       same lora_name,
        │          ──variant_4──>       different block_map)
        │
        │  roll_state        last_pick_blocks
        │       │                    ▲
        │       │                    │
        └── FizgigExplorerPick ──────┘
                 ▲
            (wire the variant you liked into `variant`,
             and the previous baseline into `previous_baseline`)
```

### Closing the loop

There is **no wire** from the Explorer back to itself — that is a cycle and ComfyUI
refuses it (`Output will be ignored`, with no useful detail). Instead the loop closes
through the file:

```
round 1   state_source = fresh state             four variants from every block at 1.0
          FizgigLoraSave on the variant you liked  -> fizgig_loras/<name>.safetensors
round 2   state_source = bootstrap from LoRA weights
          bootstrap_lora   = <name>.safetensors     four new variants from where you stopped
```

`bootstrap_reference` is the file the bake came from (default: `lora_name`). The core
reads the strengths as a **ratio** of factor norms between the baked file and its
reference, so the round trip is exact — `test_save.py` asserts a baked 2.5 and 0.4
read back as 2.5 and 0.4. Because the state lives in a file rather than in memory, it
survives a server restart, which a state-in-the-process node would not.

**First round:** leave `roll_state` empty — the Explorer builds a fresh state with
every block at 1.0. **After that:** pick one variant, wire `Pick.baseline` back into
`roll_state` and `Pick.changed_blocks` into `last_pick_blocks`, then re-queue.
Each round is one generation of the loop the GUI ran on button presses.

Four KSamplers at 512×512 with a handful of steps is the cheap way to use this; once
you have a winner, bake the strengths into a file with a normal save-LoRA node, or
just leave the loader in place in your production graph.

### Nodes

**FizgigLoraExplorer** — rolls four variants. Outputs `variant_1..4`, `baseline`
(pre-pick state, for `Pick.previous_baseline`) and `active_blocks` (family, anchor,
block list, identity blocks and what each variant changed, as JSON).

**FizgigExplorerPick** — the pick button. Takes a variant, stamps prompt / resolution /
seed / load strength onto it, returns the new baseline plus the list of blocks that
moved (which is what variant 4 routes around next round).

**FizgigLoraBlockLoader** — attaches the LoRA with per-block multipliers. The
multiplier goes on the module's **first factor**, which is exactly what Fizgig's bake
does, so the α/rank scale stays 1.0 and the sliders mean what they said in the GUI:

| Format | Factor scaled | Reason |
|---|---|---|
| kohya | `lora_up` | the loader computes `α/rank × up @ down`; linear in `up` |
| diffusers | `lora_B` | same |
| LoKR | `lokr_w1_a` / `lokr_w1` | `m·kron(w1,w2) == kron(m·w1,w2)` |
| LoHa | `hada_w1_a` / `hada_w1` | `(m·W1)#W2 == m·(W1#W2)` |

The second factor is never touched. LoKR and LoHa in, LoKR and LoHa out — no SVD, no
loss, which is what the "LoKR in, LoKR out" line in Fizgig's release notes means.
GLoRA and the Tucker/CP variants are left uniform, as in Fizgig's bake, and counted
in the report.

**FizgigLoraSave** — bakes a chosen variant into a real `.safetensors` (Fizgig's
Repair Studio save). Wire the *same* `variant_N` you sent to the loader, queue, and
the file lands in the `fizgig_loras` folder this package registers. It is an
`OUTPUT_NODE`, so it runs even with its output unconnected — use it as the button.

The multiplier is absorbed into the module's **first factor** and the alpha is set so
the file's own scale becomes 1.0, exactly as `bake.py` does:

| Format | Factor scaled | Alpha written |
|---|---|---|
| kohya | `lora_up` | `= rank` |
| diffusers | `lora_B` | `= rank` |
| LoKR / LoHa | `lokr_w1` / `hada_w1_a` | `1e6` sentinel, "the scale is already inside w1" |
| GLoRA / Tucker | — | refused, written at original weights |

A block whose multiplier is exactly 1.0 is copied byte for byte, original alpha
included, so blocks you never moved cannot drift. A disabled block is **dropped**,
not zeroed.

Two refusals worth knowing: a block map from another family, and an empty map. Both
would otherwise write a plausible-looking file that does nothing.

**FizgigCheckpointScan** — LoRA Royale's scanner. One clean epoch run gets integer
epoch labels; anything else is labelled by filename stem, sorted.

## If the Explorer says it found no blocks

The error names the shapes the file actually uses, its metadata, and the layouts this
port knows. Common causes:

- **A text-encoder-only LoRA** — nothing to explore, and the message shows that.
- **A layout from a family not listed above.** Check the printed key shape; if the
  blocks are addressable under a different name, `active_blocks_override` lets you
  name them by hand and the mutation will work as long as the loader's key mapper
  recognises the file too.
- **A GLoRA file** — not supported, as in Fizgig's bake.

## Known limits

- **GLoRA and the Tucker/CP variants are not scaled.** LoKR and LoHa are (above);
  GLoRA's 4-matrix form is refused by Fizgig's bake too.
- **`strength_model` / `strength_clip` apply on top of the per-block scaling**, so a
  second scaling pass. Keep them at 1.0 unless that is what you want.
- The Explorer needs a LoRA whose keys name transformer blocks. See above.

## Tests

```bash
python test_core.py /path/to/Fizgig     # ported core vs the real Fizgig modules
python test_nodes.py                    # node classes, with ComfyUI stubbed
python test_save.py                     # the save node, on real safetensors files
```

`test_core.py` imports Fizgig's `repair_studio/state.py` and `lora_royale/scan.py`
directly (they are stdlib-only) and checks `mutate()` bit-for-bit across seeds and
parameter combinations, plus block-id mapping against `bake.py`'s mapper and the
Qwen key layout from `families/qwen_image.py`. `test_nodes.py` builds synthetic
LoRAs in every layout above — kohya, diffusers, Qwen, LyCORIS — and exercises the
nodes end to end, including the family-mismatch and unmet-LoRA diagnostics.
`test_save.py` needs `safetensors` and numpy but not torch: a shim covers the tensor
API, and the bake is checked through to reading the strengths back with the core's
own `strength_from_pair`.

## Provenance

Upstream is Fizgig by [@shootthesound](https://github.com/shootthesound), Apache-2.0.
This port is a derivative work; keep the upstream licence and attribution intact.
