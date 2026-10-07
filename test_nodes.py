"""End-to-end check of the node classes with ComfyUI stubbed out.

Builds synthetic LoRAs in every supported layout (header only, no tensors) and
runs the Explorer node, the Pick node, the block loader's scaling maths and the
scanner over them.
"""

import json
import os
import struct
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # so `comfyui_fizgig_explorer` imports as a package

TMP = tempfile.mkdtemp()


def write_lora(path, keys, meta=None):
    header = {k: None for k in keys}
    if meta:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
    return set(keys)


def klein_keys(doubles=8, singles=22, rank=16, kohya=True):
    keys = []
    for kind, n in (("double", doubles), ("single", singles)):
        for i in range(n):
            stem = (f"lora_unet_{kind}_blocks_{i}" if kohya
                    else f"diffusion_model.{kind}_blocks.{i}")
            keys += [f"{stem}_lora_down.weight", f"{stem}_lora_up.weight", f"{stem}.alpha"]
    keys.append("lora_te1_text_model_encoder_layers_0_mlp_fc1_lora_up.weight")
    return keys


def qwen_keys(n=32):
    keys = []
    for i in range(n):
        for mod in ("attn.to_q", "attn.to_k", "img_mlp.proj"):
            keys += [f"transformer.transformer_blocks.{i}.{mod}.lora_A.weight",
                     f"transformer.transformer_blocks.{i}.{mod}.lora_B.weight"]
    keys.append("transformer.transformer_blocks.0.attn.to_q.alpha")
    return keys


LORA_DIR = os.path.join(TMP, "loras")
os.makedirs(LORA_DIR, exist_ok=True)
write_lora(os.path.join(LORA_DIR, "klein_kohya.safetensors"), klein_keys(),
           {"ss_network_dim": "16", "ss_sd_model_name": "flux2_klein_9b"})
write_lora(os.path.join(LORA_DIR, "klein_diffusers.safetensors"),
           klein_keys(kohya=False), {"ss_sd_model_name": "flux2_klein_9b"})
write_lora(os.path.join(LORA_DIR, "bonnie_qwen.safetensors"), qwen_keys(),
           {"ss_sd_model_name": "Qwen Image 2.1", "ss_network_dim": "8"})
write_lora(os.path.join(LORA_DIR, "te_only.safetensors"),
           ["lora_te1_text_model_encoder_layers_0_mlp_fc1_lora_up.weight"])

fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda kind: sorted(f for f in os.listdir(LORA_DIR)
                                           if f.endswith(".safetensors")) if kind == "loras" else []
fp.get_full_path = lambda kind, name: os.path.join(LORA_DIR, name)
sys.modules["folder_paths"] = fp

import torch_stub  # noqa: E402
sys.modules["torch"] = torch_stub

from comfyui_fizgig_explorer import explorer_core as core  # noqa: E402
from comfyui_fizgig_explorer import nodes as N             # noqa: E402

fails = []


def check(label, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(label)


def val(t):
    t = t.data if hasattr(t, "data") else t
    if hasattr(t, "ravel"):
        return float(t.ravel()[0])
    return float(t[0])


print("A. block discovery per layout")
p_single, p_diff = (os.path.join(LORA_DIR, n) for n in
                    ("klein_kohya.safetensors", "klein_diffusers.safetensors"))
p_qwen, p_te = (os.path.join(LORA_DIR, n) for n in
                ("bonnie_qwen.safetensors", "te_only.safetensors"))

ids = core.scan_block_ids(p_single, family="klein9b")
check("klein kohya: 8 double + 22 single", len(ids) == 30, len(ids))
check("klein kohya ordering", ids[:3] == ["double_0", "double_1", "double_2"], ids[:3])
check("single_10 sorts after single_9", ids.index("single_9") < ids.index("single_10"))
check("text-encoder keys are not blocks", all("te1" not in i for i in ids))

ids_d = core.scan_block_ids(p_diff, family="klein9b")
check("klein diffusers: same block set", ids_d == ids, (len(ids_d), len(ids)))

ids_q = core.scan_block_ids(p_qwen, family="qwen21")
check("qwen: 32 blocks", len(ids_q) == 32, len(ids_q))
check("qwen ids are block_N", ids_q[:2] == ["block_0", "block_1"], ids_q[:2])
check("qwen family from metadata", core.guess_family(p_qwen) == "qwen21",
      core.guess_family(p_qwen))
check("qwen auto-detect matches explicit",
      core.scan_block_ids(p_qwen) == ids_q)
check("klein family from metadata", core.guess_family(p_single) == "klein9b")

print("\nB. FizgigLoraExplorer.roll -- Qwen Image 2.1")
node = N.FizgigLoraExplorer()
v1, v2, v3, v4, baseline, meta = node.roll("bonnie_qwen.safetensors", "auto", 4, 0.6, 1.0, 123, structural_variants=True)
m = json.loads(meta)
check("family resolved to qwen21", m["family"] == "qwen21", m["family"])
check("anchor is block_0", m["anchor"] == "block_0", m["anchor"])
check("30+ blocks active", len(m["active"]) == 32, len(m["active"]))
check("identity blocks reported", m["identity_blocks"] == [f"block_{i}" for i in range(10, 15)],
      m["identity_blocks"])
check("variants are valid states",
      all(len(core.SliderState.from_json(json.loads(v)).blocks) == 32
          for v in (v1, v2, v3, v4)))
check("reproducible", node.roll("bonnie_qwen.safetensors", "auto", 4, 0.6, 1.0, 123, structural_variants=True)[0] == v1)
check("seeded reroll differs",
      node.roll("bonnie_qwen.safetensors", "auto", 4, 0.6, 1.0, 124, structural_variants=True)[0] != v1)

print("\n   locking the Qwen identity blocks")
identity = {f"block_{i}" for i in range(10, 15)}
q_locked = node.roll("bonnie_qwen.safetensors", "auto", 6, 1.0, 1.0, 77, structural_variants=True,
                     locked_blocks=",".join(sorted(identity)))
base_q = core.SliderState.from_json(json.loads(q_locked[4]))
moved = {b for v in q_locked[:4]
         for b in core.SliderState.from_json(json.loads(v)).diff_blocks(base_q)}
check("no identity block ever moved", not (moved & identity), sorted(moved))
check("everything else still moved", len(moved) > 0)
# and without the lock they do move (the recipe itself has no identity awareness)
q_free = node.roll("bonnie_qwen.safetensors", "auto", 12, 1.0, 1.0, 77, structural_variants=True)
base_free = core.SliderState.from_json(json.loads(q_free[4]))
moved_free = {b for v in q_free[:4]
              for b in core.SliderState.from_json(json.loads(v)).diff_blocks(base_free)}
check("unlocked, identity blocks are in play", bool(moved_free & identity), sorted(moved_free))

print("\nB2. the map carries its family, and the Explorer refuses a foreign one")
check("a fresh roll stamps the family on every variant",
      [core.SliderState.from_json(json.loads(v)).family for v in (v1, v2, v3, v4)]
      == ["qwen21"] * 4,
      [core.SliderState.from_json(json.loads(v)).family for v in (v1, v2, v3, v4)])
check("the baseline carries it too",
      core.SliderState.from_json(json.loads(baseline)).family == "qwen21")
check("the pick out of a variant keeps it",
      core.SliderState.from_json(json.loads(
          N.FizgigExplorerPick().pick(v1, baseline)[0])).family == "qwen21")
# an H3 map pointed at the Qwen file must be refused, not silently do nothing
h3_map = json.dumps({"blocks": {f"h3blk_{i}": {"primary_enabled": True,
                                              "primary_strength": 1.0,
                                              "donor_enabled": True,
                                              "donor_strength": 0.0} for i in range(28)},
                     "seed": 1, "family": "h3"})
try:
    node.roll("bonnie_qwen.safetensors", "auto", 4, 0.5, 1.0, 1, structural_variants=True, roll_state=h3_map)
    check("a foreign map is refused by the Explorer", False)
except ValueError as e:
    check("a foreign map is refused by the Explorer",
          "h3 map but" in str(e) and "qwen21" in str(e), str(e))
    check("and the message shows both namings", "h3blk_0" in str(e), str(e))
# a map with no family (written by an older version) must still be accepted
legacy_map = json.dumps({"blocks": {f"block_{i}": {"primary_enabled": True,
                                                   "primary_strength": 1.0,
                                                   "donor_enabled": True,
                                                   "donor_strength": 0.0}
                                    for i in range(32)}, "seed": 1})
out_legacy = node.roll("bonnie_qwen.safetensors", "auto", 4, 0.5, 1.0, 1, structural_variants=True,
                       roll_state=legacy_map)
check("a family-less map is accepted and re-stamped",
      core.SliderState.from_json(json.loads(out_legacy[4])).family == "qwen21")

print("\nC. the ambiguous bare `blocks_N` shape")
bare_dir = os.path.join(TMP, "bare")
os.makedirs(bare_dir, exist_ok=True)
write_lora(os.path.join(bare_dir, "qwen_bare.safetensors"),
           [f"lora_unet_blocks_{i}_lora_up.weight" for i in range(32)]
           + [f"lora_unet_blocks_{i}_lora_down.weight" for i in range(32)],
           {"ss_sd_model_name": "Qwen Image 2.1"})
write_lora(os.path.join(bare_dir, "h3_bare.safetensors"),
           [f"lora_unet_blocks_{i}_lora_up.weight" for i in range(50)]
           + [f"lora_unet_token_refiner_blocks_{i}_lora_up.weight" for i in range(2)],
           {})
check("qwen bare -> block_N (metadata wins)",
      core.scan_block_ids(os.path.join(bare_dir, "qwen_bare.safetensors"))[:2],
      ["block_0", "block_1"])
h3_ids = core.scan_block_ids(os.path.join(bare_dir, "h3_bare.safetensors"))
check("h3 bare -> 50 h3blk_N + 2 h3_rf_N",
      len([i for i in h3_ids if i.startswith("h3blk_")]) == 50
      and len([i for i in h3_ids if i.startswith("h3_rf_")]) == 2, h3_ids[:3])
check("h3 namespace, not block_N", "block_0" not in h3_ids)
check("h3 family guessed from block count",
      core.guess_family(os.path.join(bare_dir, "h3_bare.safetensors")) == "h3")

print("\nD. the error message names the real layout")
try:
    node.roll("te_only.safetensors", "auto", 4, 0.5, 1.0, 1, structural_variants=True)
    check("raises on a LoRA with no blocks", False)
except ValueError as e:
    msg = str(e)
    check("raises on a LoRA with no blocks", True)
    check("shows the file's actual key shape",
          "lora_teN_text_model_encoder_layers_N_mlp_fcN_lora_up.weight" in msg, msg)
    check("lists the known layouts", "transformer.transformer_blocks.4" in msg, msg)
    check("reports the metadata slot even when empty because of digit stripping",
          "metadata:" in msg and "(none)" in msg, msg)

print("\nE. Pick closes the loop")
pick = N.FizgigExplorerPick()
p_baseline, changed, readout = pick.pick(v2, baseline, seed=-1, prompt="a cat",
                                         preview_width=768, preview_height=768,
                                         load_strength=0.8)
ps = core.SliderState.from_json(json.loads(p_baseline))
check("picked state carries the variant's blocks",
      ps.to_json()["blocks"] == core.SliderState.from_json(json.loads(v2)).to_json()["blocks"])
check("prompt stamped", ps.prompt == "a cat")
check("load strength stamped", ps.primary_scale == 0.8)
check("changed_blocks matches the diff",
      set(changed.split(",")) == set(ps.diff_blocks(core.SliderState.from_json(json.loads(baseline)))))
out2 = node.roll("bonnie_qwen.safetensors", "auto", 4, 0.6, 1.0, 321, structural_variants=True,
                 roll_state=p_baseline, last_pick_blocks=changed)
v4b = core.SliderState.from_json(json.loads(out2[3]))
check("generation 2 rolled from the pick", v4b.primary_scale == 0.8)
check("variant 4 protected the last pick's blocks",
      not (set(v4b.diff_blocks(ps)) & set(changed.split(","))))

print("\nF. block loader scaling maths")
state = core.SliderState.from_json(json.loads(v2))
state.blocks["block_0"] = core.BlockState(primary_enabled=False, primary_strength=1.0)
sd = {"transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch_stub._T([1.0, 1.0]),
      "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch_stub._T([1.0, 1.0]),
      "transformer.transformer_blocks.12.attn.to_q.lora_B.weight": torch_stub._T([1.0, 1.0])}
report, warns = N._scale_lora_weights(sd, state, p_qwen, "qwen21", 1.0, True)
check("disabled block zeroed",
      val(sd["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"]) == 0.0)
check("lora_A is NOT scaled",
      val(sd["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]) == 1.0)
check("neighbouring block uses its own slider",
      abs(val(sd["transformer.transformer_blocks.12.attn.to_q.lora_B.weight"])
          - state.blocks["block_12"].primary_strength) < 1e-6)
check("report names the family", "family: qwen21" in report, report)
check("report counts the off block", "off: 1" in report, report)

klein_state = core.SliderState.from_block_ids(
    core.scan_block_ids(p_single, family="klein9b"))
sd_k = {"lora_unet_double_blocks_0_lora_up.weight": torch_stub._T([1.0]),
        "lora_te1_x_lora_up.weight": torch_stub._T([1.0])}
N._scale_lora_weights(sd_k, klein_state, p_single, "klein9b", 0.25, True)
check("kohya underscore suffix is scaled",
      abs(val(sd_k["lora_unet_double_blocks_0_lora_up.weight"])
          - klein_state.blocks["double_0"].primary_strength) < 1e-6)
check("unlisted_strength applies to the text encoder",
      abs(val(sd_k["lora_te1_x_lora_up.weight"]) - 0.25) < 1e-6)
sd_k2 = {"lora_te1_x_lora_up.weight": torch_stub._T([1.0])}
N._scale_lora_weights(sd_k2, klein_state, p_single, "klein9b", 0.25, False)
check("scale_unlisted=False leaves them untouched", val(sd_k2["lora_te1_x_lora_up.weight"]) == 1.0)
sd_l = {"lora_unet_single_blocks_3_lokr_w1": torch_stub._T([1.0])}
rep_l, warns_l = N._scale_lora_weights(sd_l, klein_state, p_single, "klein9b", 1.0, True)
check("LyCORIS key left alone", val(sd_l["lora_unet_single_blocks_3_lokr_w1"]) == 1.0)
check("report names the LyCORIS skip", "LyCORIS" in rep_l, rep_l)
check("report lists map blocks missing from the file", "absent from this file" in rep_l, rep_l)

print("\nF2. LyCORIS (LoKR / LoHa) scaling")
lokr_dir = os.path.join(TMP, "lokr")
os.makedirs(lokr_dir, exist_ok=True)
lokr_keys = []
for i in range(4):
    for mod in ("attn.to_q", "img_mlp.proj"):
        stem = f"transformer.transformer_blocks.{i}.{mod}"
        lokr_keys += [f"{stem}.lokr_w1", f"{stem}.lokr_w2",
                      f"{stem}.lokr_w1_a", f"{stem}.lokr_w1_b"]
        # LoHa as well, to prove both forms reach the same factor
        lokr_keys += [f"{stem}.hada_w1_a", f"{stem}.hada_w1_b", f"{stem}.hada_w2"]
lokr_keys.append("transformer.transformer_blocks.0.attn.to_q.lokr_alpha")
write_lora(os.path.join(lokr_dir, "bonnie_lokr.safetensors"), lokr_keys,
           {"ss_sd_model_name": "Qwen Image 2.1"})
p_lokr = os.path.join(lokr_dir, "bonnie_lokr.safetensors")

lokr_ids = core.scan_block_ids(p_lokr, family="qwen21")
check("LoKR keys yield block ids", lokr_ids == ["block_0", "block_1", "block_2", "block_3"], lokr_ids)
lokr_state = core.SliderState.from_block_ids(lokr_ids)
lokr_state.family = "qwen21"
lokr_state.blocks["block_0"] = core.BlockState(primary_enabled=False, primary_strength=1.0)
lokr_state.blocks["block_2"] = core.BlockState(primary_strength=2.5)
sd_lokr = {k: torch_stub._T([1.0]) for k in lokr_keys}
rep_lokr, warns_lokr = N._scale_lora_weights(sd_lokr, lokr_state, p_lokr, "qwen21", 1.0, True)
check("lokr_w1 scaled to the block's slider",
      abs(val(sd_lokr["transformer.transformer_blocks.2.attn.to_q.lokr_w1"]) - 2.5) < 1e-6,
      val(sd_lokr["transformer.transformer_blocks.2.attn.to_q.lokr_w1"]))
check("lokr_w1_b NOT scaled (second factor stays put)",
      val(sd_lokr["transformer.transformer_blocks.2.attn.to_q.lokr_w1_b"]) == 1.0)
check("hada_w1_a scaled too",
      abs(val(sd_lokr["transformer.transformer_blocks.2.attn.to_q.hada_w1_a"]) - 2.5) < 1e-6)
check("hada_w2 NOT scaled",
      val(sd_lokr["transformer.transformer_blocks.2.attn.to_q.hada_w2"]) == 1.0)
check("a disabled block zeroes its first factor",
      val(sd_lokr["transformer.transformer_blocks.0.attn.to_q.lokr_w1"]) == 0.0)
check("lokr_w2 untouched where block 0 is off",
      val(sd_lokr["transformer.transformer_blocks.0.attn.to_q.lokr_w2"]) == 1.0)
check("no WARNING when a LoKR is scaled", not warns_lokr, warns_lokr)
check("report says the first factors were scaled",
      "LyCORIS first factors scaled" in rep_lokr, rep_lokr)
check("lokr_alpha is left alone",
      val(sd_lokr["transformer.transformer_blocks.0.attn.to_q.lokr_alpha"]) == 1.0)

print("\nF4. the anchor invert is opt-in (the 'mirror warp' regression)")
q_active = core.scan_block_ids(p_qwen, family="qwen21")
base_q2 = core.SliderState.from_block_ids(q_active)
base_q2.family = "qwen21"

# structural_variants=False, whatever structure says: the anchor may still be
# mutated as an ordinary candidate, but it can never be negated by the recipe.
off = core.roll_variants(base_q2, q_active, num_mutations=3, intensity=1.0,
                         structure=0.0, anchor="block_0", seed=5)
check("structural_variants=False leaves no negated anchor",
      all(v.blocks["block_0"].primary_strength >= -0.05 for v in off),
      [v.blocks["block_0"].primary_strength for v in off])

# and the module's own rule: below 0.3 the structural branch cannot fire at all
low = core.roll_variants(base_q2, q_active, num_mutations=3, intensity=0.25,
                         structure=1.0, anchor="block_0", seed=5)
check("intensity 0.25 cannot invert the anchor (state.py gate)",
      all(v.blocks["block_0"].primary_strength >= -0.05 for v in low),
      [v.blocks["block_0"].primary_strength for v in low])

# with it on, the inversion is real -- which is why it is off by default
on = core.roll_variants(base_q2, q_active, num_mutations=3, intensity=1.0,
                        structure=1.0, anchor="block_0", seed=5)
# state.py picks invert OR extreme at random, so variant 2 may land on +3.0
# rather than on -1.0. What must hold is that variant 1 -- always first, always
# structural -- is the one that carries it, and that it is the recipe's own
# choice, not an accident of sampling.
v1_anchor = on[0].blocks["block_0"].primary_strength
check("structural_variants=True makes variant 1 the structural one",
      abs(v1_anchor) > 0.5, v1_anchor)
negated = [i + 1 for i, v in enumerate(on) if v.blocks["block_0"].primary_strength < -0.05]
check("and the invert form negates it outright",
      -1.0 in [round(on[i].blocks["block_0"].primary_strength, 4) for i in range(2)]
      or negated, [round(on[i].blocks["block_0"].primary_strength, 4) for i in range(2)])

# the node's own default must be the gentle one
spec = N.FizgigLoraExplorer.INPUT_TYPES()
req, opt = spec["required"], spec.get("optional", {})
check("node default intensity is gentle", req["intensity"][1]["default"] <= 0.3,
      req["intensity"][1]["default"])
check("node default structure is 0.0", req["structure"][1]["default"] == 0.0)
check("node default mutations is small", req["mutations"][1]["default"] <= 3)
check("structural_variants defaults to off", opt["structural_variants"][1]["default"] is False)

# A widget added after a release must be OPTIONAL and LAST. ComfyUI validates
# that every required input is present, and a saved workflow has no value for a
# widget it never saw -- "Required input is missing", failing before any node
# runs. It also matches a workflow's widget values POSITIONALLY, so inserting
# anywhere but the end shifts every later value onto the wrong widget: silent
# misconfiguration rather than an error.
check("structural_variants is not a required input", "structural_variants" not in req)
check("structural_variants is the last optional input",
      list(opt)[-1] == "structural_variants", list(opt))
check("the original widgets keep their order",
      list(req) == ["lora_name", "family", "mutations", "intensity", "structure", "seed",
                    "locked_blocks", "last_pick_blocks", "state_source", "roll_state",
                    "bootstrap_lora", "bootstrap_reference", "active_blocks_override"],
      list(req))
# and the call must work with the widget absent, which is how an old graph calls it
import inspect as _inspect
_sig = _inspect.signature(N.FizgigLoraExplorer.roll)
check("roll() defaults structural_variants", _sig.parameters["structural_variants"].default is False)
check("it sits after every defaulted parameter",
      list(_sig.parameters)[-1] == "structural_variants", list(_sig.parameters))

# and rolling with the node's defaults must not negate anything
g = N.FizgigLoraExplorer().roll("bonnie_qwen.safetensors", "auto", 3, 0.25, 0.0, 7)
gb = core.SliderState.from_json(json.loads(g[4]))
check("a roll on the defaults never negates the anchor",
      all(core.SliderState.from_json(json.loads(v)).blocks["block_0"].primary_strength >= -0.05
          for v in g[:4]),
      [core.SliderState.from_json(json.loads(v)).blocks["block_0"].primary_strength for v in g[:4]])

print("\nF5. the file set is self-consistent (the 'missing node type' guard)")
# A pack whose __init__ raises registers NO nodes: ComfyUI then reports every
# Fizgig node in a saved workflow as "Missing node type" and lists them under
# "Unknown pack". The usual cause is nodes.py and explorer_core.py being out of
# sync -- so assert every helper nodes.py imports actually exists.
import ast as _ast

_requested, _from = set(), None
for _node in _ast.walk(_ast.parse(open("nodes.py").read())):
    if isinstance(_node, _ast.ImportFrom) and _node.module == "explorer_core":
        _requested |= {a.name for a in _node.names}
        _from = _node.module
check("nodes.py imports from explorer_core", _from == "explorer_core", _from)
_absent = sorted(n for n in _requested if not hasattr(core, n))
check("every helper nodes.py imports exists in explorer_core", not _absent, _absent)
check("the bake-bootstrap helpers are present",
      all(hasattr(core, n) for n in ("state_from_baked", "strength_from_pair", "factor_norms")))

# and the package must import cleanly with ComfyUI's folder_paths stubbed, which
# is the direct test of "do the nodes register at all"
import importlib as _il
import types as _types
_prev_fp = sys.modules.get("folder_paths")
_stub = _types.ModuleType("folder_paths")
_stub.get_filename_list = lambda kind: []
_stub.get_full_path = lambda k, n: n
_stub.get_folder_paths = lambda kind: []
_stub.get_output_directory = lambda: "/tmp"
_stub.add_model_folder_path = lambda k, p: None
sys.modules["folder_paths"] = _stub
for _m in [k for k in list(sys.modules) if k.startswith("comfyui_fizgig_explorer")]:
    del sys.modules[_m]
try:
    _pkg = _il.import_module("comfyui_fizgig_explorer")
    check("the package imports and registers its nodes",
          len(_pkg.NODE_CLASS_MAPPINGS) == 5, sorted(_pkg.NODE_CLASS_MAPPINGS))
    check("every registered class is a real class",
          all(isinstance(v, type) for v in _pkg.NODE_CLASS_MAPPINGS.values()))
    check("every class has a display name",
          set(_pkg.NODE_DISPLAY_NAME_MAPPINGS) == set(_pkg.NODE_CLASS_MAPPINGS))
    check("each node declares INPUT_TYPES and FUNCTION",
          all(hasattr(v, "INPUT_TYPES") and hasattr(v, "FUNCTION")
              for v in _pkg.NODE_CLASS_MAPPINGS.values()))
    check("each node's FUNCTION names a real method",
          all(callable(getattr(v, v.FUNCTION, None))
              for v in _pkg.NODE_CLASS_MAPPINGS.values()),
          [v.__name__ for v in _pkg.NODE_CLASS_MAPPINGS.values()
           if not callable(getattr(v, v.FUNCTION, None))])
finally:
    if _prev_fp is not None:
        sys.modules["folder_paths"] = _prev_fp

print("\nF6. a flat map must change NOTHING (the 'distortion at zero settings' guard)")
# The decisive split: if a block map with every block at exactly 1.0 and nothing
# disabled still alters the weights, the distortion is in the scaling, not in the
# mutation -- and no slider value can explain it. One tensor per module, compared
# byte for byte, is the whole test.
import numpy as _np

_qwen_keys = {}
for _i in range(32):
    for _mod in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                 "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"):
        _stem = f"transformer.transformer_blocks.{_i}.{_mod}"
        _qwen_keys[f"{_stem}.lora_A.weight"] = _np.full((8, 16), 0.5, _np.float32)
        _qwen_keys[f"{_stem}.lora_B.weight"] = _np.full((8, 16), 0.5, _np.float32)
        _qwen_keys[f"{_stem}.alpha"] = _np.array(16.0, _np.float32)
# keys that are NOT blocks: a flat map must leave these alone too
_qwen_keys["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"] = _np.full((4, 4), 0.5, _np.float32)
_qwen_keys["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = _np.full((4, 4), 0.5, _np.float32)

_flat = core.SliderState.from_block_ids([f"block_{i}" for i in range(32)])
_flat.family = "qwen21"
_sd = {k: torch_stub._T(v.copy()) for k, v in _qwen_keys.items()}
_before = {k: v.data.copy() for k, v in _sd.items()}
_rep, _warns = N._scale_lora_weights(_sd, _flat, "/x/qwen.safetensors", "qwen21", 1.0, True)
_touched = [k for k in _sd if not _np.array_equal(_before[k], _sd[k].data)]
check("a flat map modifies no tensor at all", not _touched, _touched[:4])
check("and raises no scaling warning", not _warns, _warns)

# every block disabled => every block's contribution is removed, which is the
# legitimate, large change a map can make. Confirm it is total, not partial.
_off = core.SliderState.from_block_ids([f"block_{i}" for i in range(32)])
_off.family = "qwen21"
for _b in _off.blocks:
    _off.blocks[_b] = core.BlockState(primary_enabled=False, primary_strength=1.0)
_sd2 = {k: torch_stub._T(v.copy()) for k, v in _qwen_keys.items()}
N._scale_lora_weights(_sd2, _off, "/x/qwen.safetensors", "qwen21", 1.0, True)
_zeroed = [k for k in _sd2 if k.endswith(".lora_B.weight") and "transformer_blocks" in k
           and float(_sd2[k].data.flat[0]) == 0.0]
check("an off block zeroes its own module only",
      len(_zeroed) == 32 * 7, len(_zeroed))
check("but never the text-encoder keys",
      float(_sd2["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"].data.flat[0]) == 0.5)

# The magnitude the mutation can apply. This is arithmetic, not opinion.
#
# Fizgig's magnitude is 0.2 + intensity * 2.8, so its intensity 0.0 still moves
# every chosen block by +-0.2 -- there is no setting that leaves the file alone,
# which makes "is the graph wired correctly?" impossible to answer by looking.
# This port drops the floor at exactly 0.0 and keeps it everywhere else, so the
# recipe is unchanged at the values the GUI actually used (its refine profile is
# 0.25, its exploration default 0.964).
_ids = [f"block_{i}" for i in range(32)]
_base = core.SliderState.from_block_ids(_ids)
_base.family = "qwen21"

_vs0 = core.roll_variants(_base, _ids, num_mutations=3, intensity=0.0, structure=0.0,
                          anchor="block_0", seed=3)
check("at intensity 0 nothing at all changes",
      all(not _v.diff_blocks(_base) for _v in _vs0),
      [len(_v.diff_blocks(_base)) for _v in _vs0])
# BlockState is a dataclass, so equality is field-by-field -- stronger than
# comparing the strengths alone, it also catches an enabled flag flipping.
check("at intensity 0 the state is untouched, block for block",
      all(all(_v.blocks[b] == _base.blocks[b] for b in _base.blocks) for _v in _vs0),
      [(b, _base.blocks[b], _vs0[0].blocks[b]) for b in _base.blocks
       if _vs0[0].blocks[b] != _base.blocks[b]][:4])
check("at intensity 0 no block is ever disabled",
      all(all(_v.blocks[b].primary_enabled for b in _base.blocks) for _v in _vs0))

# And one step up from zero the floor is back, so a gentle roll still explores.
# The magnitude is `0.2 + intensity * 2.8`, so 0.1 gives a span of +-0.48 -- the
# 0.2 is the FLOOR of the span, not a cap on the delta. Worth stating, because
# reading it as a cap is exactly the mistake that made "intensity 0 is safe" look
# true (it was not: even at 0.0 the span was +-0.2).
_vs1 = core.roll_variants(_base, _ids, num_mutations=3, intensity=0.1, structure=0.0,
                          anchor="block_0", seed=3)
_deltas1 = [abs(_v.blocks[_b].primary_strength - 1.0)
            for _v in _vs1 for _b in _v.diff_blocks(_base)]
_span1 = 0.2 + 0.1 * 2.8
check("at intensity 0.1 the floor is back (blocks move again)", bool(_deltas1), _deltas1)
check("and the span is the recipe's 0.2 + intensity * 2.8",
      all(d <= _span1 + 1e-9 for d in _deltas1) and max(_deltas1) > 0.2,
      (round(max(_deltas1), 4), round(_span1, 4)))

print("\nG. loader diagnostics (the '4 identical renders' question)")
warn_report = None


class _FakeModel:
    """ComfyUI hands the loader a ModelPatcher; all we touch is .model."""
    def __init__(self):
        self.model = object()
        self.cond_stage_model = None


class _FakeComfy(types.ModuleType):
    """Stands in for comfy.sd / comfy.utils / comfy.lora."""
    def __init__(self, name, patch_count, apply_error=None):
        super().__init__(name)
        self._patch_count = patch_count
        self._apply_error = apply_error

    def load_torch_file(self, path, safe_load=True):
        return dict(SCALED_SD)

    def load_lora_for_models(self, model, clip, lora_sd, sm, sc):
        if self._apply_error:
            raise self._apply_error
        return object(), None

    def model_lora_keys_unet(self, m, km):
        return km

    def model_lora_keys_clip(self, c, km):
        return km

    def load_lora(self, lora_sd, key_map, **kw):
        return {f"module_{i}": (None, None) for i in range(self._patch_count)}


SCALED_SD = {"transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch_stub._T([1.0])}


loader = N.FizgigLoraBlockLoader()
map_json = v2

def _run_loader(patch_count, apply_error=None, block_map=None, name="bonnie_qwen.safetensors",
                family="qwen21", map_family=None):
    import sys as _s
    fake_sd = types.ModuleType("comfy.sd")
    fake_utils = types.ModuleType("comfy.utils")
    fake_lora = types.ModuleType("comfy.lora")
    fake_pkg = types.ModuleType("comfy")
    fk = _FakeComfy("comfy.sd", patch_count, apply_error)
    fake_sd.load_lora_for_models = fk.load_lora_for_models
    fake_utils.load_torch_file = fk.load_torch_file
    fake_lora.model_lora_keys_unet = fk.model_lora_keys_unet
    fake_lora.model_lora_keys_clip = fk.model_lora_keys_clip
    fake_lora.load_lora = fk.load_lora
    fake_pkg.sd, fake_pkg.utils, fake_pkg.lora = fake_sd, fake_utils, fake_lora
    _s.modules["comfy"] = fake_pkg
    _s.modules["comfy.sd"] = fake_sd
    _s.modules["comfy.utils"] = fake_utils
    _s.modules["comfy.lora"] = fake_lora
    try:
        bm = block_map if block_map is not None else map_json
        if map_family is not None and bm.strip():
            st = core.SliderState.from_json(json.loads(bm))
            st.family = map_family
            bm = json.dumps(st.to_json())
        return loader.load(_FakeModel(), None, name, bm, family=family)
    finally:
        for m in ("comfy", "comfy.sd", "comfy.utils", "comfy.lora"):
            _s.modules.pop(m, None)


rep, = (_run_loader(3)[2],)
check("report reaches the loader output", "blocks in map" in rep, rep)
check("no warning when the LoRA patches modules", "WARNING" not in rep, rep)

rep_none, = (_run_loader(3)[2],)
rep_unpatched, = (_run_loader(3, apply_error=None, block_map="")[2],)
check("an empty block_map is called out", "no map to apply" in rep_unpatched, rep_unpatched)
# a Klein map pointed at a Qwen file: the loader must say so, not go quiet
# A foreign map is now REFUSED, not warned about: two families can share a block
# count (Qwen and Krea 2 both have 32), so applying one to the other scales the
# wrong blocks and distorts the picture while every counter looks healthy.
try:
    _run_loader(3, family="auto", map_family="klein9b")
    check("a map from another family is refused", False)
except ValueError as e:
    msg = str(e)
    check("a map from another family is refused",
          "family mismatch" in msg and "klein9b" in msg and "qwen21" in msg, msg)
    check("and the message says what would go wrong",
          "distorts" in msg or "wrong blocks" in msg, msg)
# a widget that fights the map too
try:
    _run_loader(3, family="klein9b", map_family="qwen21")
    check("a widget disagreeing with the map is refused", False)
except ValueError as e:
    check("a widget disagreeing with the map is refused",
          "family mismatch" in str(e), str(e))
# and a matching pair stays quiet
rep_same = _run_loader(3, family="auto", map_family="qwen21")[2]
check("a matching pair raises no family warning", "family mismatch" not in rep_same, rep_same)

rep_mismatch = _run_loader(0)[2]
check("zero patched modules warns explicitly",
      "NOT patched" in rep_mismatch and "identically" in rep_mismatch, rep_mismatch)

try:
    _run_loader(3, apply_error=RuntimeError("shape mismatch"))
    check("an apply failure raises with the report attached", False)
except RuntimeError as e:
    check("an apply failure raises with the report attached",
          "shape mismatch" in str(e) and "blocks in map" in str(e))

print("\n   variant states really do differ (the algorithm side)")
q_active = core.scan_block_ids(p_qwen, family="qwen21")
q_base = core.SliderState.from_block_ids(q_active)
q_vars = core.roll_variants(q_base, q_active, num_mutations=5, intensity=0.8,
                            structure=1.0, anchor="block_0", seed=99)
for i, st in enumerate(q_vars):
    check(f"variant {i + 1} differs from the baseline", bool(st.diff_blocks(q_base)))
check("the four variants differ from each other",
      len({json.dumps(st.to_block_set()) for st in q_vars}) == 4)
check("all four roll at the same sampler seed",
      len({st.seed for st in q_vars}) == 1)

print("\nH. FizgigCheckpointScan")
run = os.path.join(TMP, "run")
os.makedirs(run, exist_ok=True)
for e in (1, 2, 3):
    open(os.path.join(run, f"hero-{e:06d}.safetensors"), "wb").close()
scan = N.FizgigCheckpointScan()
paths, labels, count = scan.scan(run, 0)
check("count", count == 3)
check("epoch labels", labels.split("\n") == ["1", "2", "3"])
check("paths are absolute", all(os.path.isabs(p) for p in paths.split("\n")))

print()
if fails:
    print(f"{len(fails)} FAILURE(S): " + "; ".join(fails))
    sys.exit(1)
print("All node-level checks passed.")
