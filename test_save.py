"""The save node, on real safetensors files.

torch is not installed here, so a shim stands in for it: tensors are numpy
arrays behind the .to()/.dtype/shape API the node touches, and safetensors.torch
is wrapped onto safetensors.numpy. That exercises the node's own logic --
module grouping, multiplier absorption, alpha/rank -- against real file I/O.
"""

import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# ---- torch shim ------------------------------------------------------------
class _T:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32) if not isinstance(data, _T) else data.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return "float32"

    def to(self, *a, **k):
        return self

    # The core measures factor norms through these three, so the shim needs
    # them or strength_from_pair silently returns nothing.
    def float(self):
        return self

    def cpu(self):
        return self

    def norm(self):
        return _T([float(np.linalg.norm(self.data))])

    def item(self):
        return float(self.data.reshape(-1)[0])

    def __mul__(self, o):
        return _T(self.data * (o.data if isinstance(o, _T) else o))

    def __rmul__(self, o):
        return _T(o * self.data)

    # the save node forms the sentinel alpha as `t * 0.0 + 1.0`, which is how a
    # scalar gets written into a tensor of the right dtype without a cast
    def __add__(self, o):
        return _T(self.data + (o.data if isinstance(o, _T) else o))

    def __radd__(self, o):
        return _T(o + self.data)

    def __sub__(self, o):
        return _T(self.data - (o.data if isinstance(o, _T) else o))

    def __truediv__(self, o):
        return _T(self.data / (o.data if isinstance(o, _T) else o))


torch_stub = types.ModuleType("torch")
torch_stub.float32 = "float32"
torch_stub.tensor = lambda v, dtype=None: _T([v])
sys.modules["torch"] = torch_stub

# ---- safetensors.torch shim on the numpy backend ---------------------------
import safetensors.numpy as st_np

st_torch = types.ModuleType("safetensors.torch")
st_torch.load_file = lambda path, **k: {k2: _T(v) for k2, v in st_np.load_file(path).items()}


def _save_file(sd, path, metadata=None):
    st_np.save_file({k: (v.data if isinstance(v, _T) else np.asarray(v, dtype=np.float32))
                     for k, v in sd.items()}, path, metadata=metadata)


st_torch.save_file = _save_file
import safetensors
safetensors.torch = st_torch
sys.modules["safetensors.torch"] = st_torch

# ---- folder_paths stub -----------------------------------------------------
D = tempfile.mkdtemp()
LORA_DIR = os.path.join(D, "loras")
BAKE_DIR = os.path.join(D, "bakes")
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(BAKE_DIR, exist_ok=True)

fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda kind: sorted(f for f in os.listdir(LORA_DIR)
                                           if f.endswith(".safetensors")) if kind == "loras" else []
fp.get_full_path = lambda kind, name: os.path.join(LORA_DIR, name)
fp.get_folder_paths = lambda kind: [BAKE_DIR] if kind == "fizgig_loras" else []
fp.get_output_directory = lambda: D
sys.modules["folder_paths"] = fp

from comfyui_fizgig_explorer import explorer_core as core  # noqa: E402
from comfyui_fizgig_explorer import nodes as N             # noqa: E402

fails = []


def check(label, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(label)


def write_qwen_lora(name, scale=1.0, blocks=4, lokr=False, glora=False):
    sd = {}
    for i in range(blocks):
        for mod in ("attn.to_q", "img_mlp.proj"):
            stem = f"transformer.transformer_blocks.{i}.{mod}"
            if lokr:
                sd[f"{stem}.lokr_w1"] = _T(np.ones((4, 4)) * scale)
                sd[f"{stem}.lokr_w2"] = _T(np.ones((4, 4)) * scale)
                sd[f"{stem}.lokr_alpha"] = _T([1.0])
            elif glora:
                sd[f"{stem}.glora_a"] = _T(np.ones((8, 16)))
                sd[f"{stem}.glora_b"] = _T(np.ones((8, 16)))
            else:
                sd[f"{stem}.lora_A.weight"] = _T(np.ones((8, 16)) * scale)
                sd[f"{stem}.lora_B.weight"] = _T(np.ones((8, 16)) * scale)
                sd[f"{stem}.alpha"] = _T([16.0])
    sd["lora_te1_x.lora_up.weight"] = _T(np.ones((4, 4)))     # not a block
    path = os.path.join(LORA_DIR, name)
    st_np.save_file({k: v.data for k, v in sd.items()}, path,
                    metadata={"ss_sd_model_name": "Qwen Image 2.1"})
    return path


def state_json(blocks):
    return json.dumps({"blocks": blocks, "seed": 7, "family": "qwen21"})


def bs(strength=1.0, enabled=True):
    return {"primary_enabled": enabled, "primary_strength": strength,
            "donor_enabled": True, "donor_strength": 0.0}


print("A. no-op: nothing moved -> the file is written unchanged")
src = write_qwen_lora("qwen_plain.safetensors")
before = {k: v.copy() for k, v in st_np.load_file(src).items()}
save = N.FizgigLoraSave()
out_path, = save.save("qwen_plain.safetensors",
                      state_json({f"block_{i}": bs() for i in range(4)}),
                      "", False)
after = st_np.load_file(out_path)
check("every tensor identical", set(before) == set(after)
      and all(np.array_equal(before[k], after[k]) for k in before),
      [k for k in before if k not in after or not np.array_equal(before[k], after[k])])
check("alpha left at the original rank",
      float(after["transformer.transformer_blocks.0.attn.to_q.alpha"][0]) == 16.0)
check("saved into the fizgig_loras folder", os.path.dirname(out_path) == BAKE_DIR, out_path)
check("metadata records the source", st_np.load_file(out_path) is not None)

print("\nB. a multiplier is absorbed into lora_B, alpha becomes the rank")
out_path, = save.save("qwen_plain.safetensors",
                      state_json({"block_0": bs(2.5), "block_1": bs(1.0),
                                  "block_2": bs(1.0), "block_3": bs(1.0)}),
                      "scaled.safetensors", True)
after = st_np.load_file(out_path)
check("block_0 lora_B scaled by 2.5",
      abs(float(after["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"][0, 0]) - 2.5) < 1e-6)
check("block_0 lora_A NOT scaled",
      float(after["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"][0, 0]) == 1.0)
check("block_0 alpha set to the rank (16)",
      float(after["transformer.transformer_blocks.0.attn.to_q.alpha"][0]) == 16.0)
check("block_1 untouched (multiplier 1.0)",
      float(after["transformer.transformer_blocks.1.attn.to_q.lora_B.weight"][0, 0]) == 1.0)
check("the text-encoder key passes through",
      float(after["lora_te1_x.lora_up.weight"][0, 0]) == 1.0)

print("\nC. a disabled block is dropped, not zeroed")
out_path, = save.save("qwen_plain.safetensors",
                      state_json({"block_0": bs(enabled=False), "block_1": bs(),
                                  "block_2": bs(), "block_3": bs()}),
                      "dropped.safetensors", True)
after = st_np.load_file(out_path)
gone = [k for k in after if "transformer_blocks.0." in k]
check("no tensors remain for the disabled block", gone == [], gone)
check("its neighbours are still there",
      "transformer.transformer_blocks.1.attn.to_q.lora_B.weight" in after)

print("\nD. round trip: the core reads the strengths back")
out_path, = save.save("qwen_plain.safetensors",
                      state_json({"block_0": bs(2.5), "block_1": bs(0.4),
                                  "block_2": bs(1.0), "block_3": bs(1.0)}),
                      "roundtrip.safetensors", True)
baked = {k: _T(v) for k, v in st_np.load_file(out_path).items()}
ref = {k: _T(v) for k, v in st_np.load_file(src).items()}
recovered = core.strength_from_pair(baked, ref, [f"block_{i}" for i in range(4)], family="qwen21")
check("block_0 reads back as 2.5", abs(recovered.get("block_0", 0) - 2.5) < 1e-4, recovered)
check("block_1 reads back as 0.4", abs(recovered.get("block_1", 0) - 0.4) < 1e-4, recovered)
check("untouched blocks read back as 1.0",
      all(abs(recovered.get(f"block_{i}", 0) - 1.0) < 1e-4 for i in (2, 3)), recovered)
state_back = core.state_from_baked(baked, ref, [f"block_{i}" for i in range(4)], family="qwen21")
check("state_from_baked reproduces the multipliers",
      abs(state_back.blocks["block_0"].primary_strength - 2.5) < 1e-4)

print("\nE. LyCORIS: the multiplier goes into w1, alpha becomes the sentinel")
lokr_src = write_qwen_lora("qwen_lokr.safetensors", lokr=True)
out_path, = save.save("qwen_lokr.safetensors",
                      state_json({"block_0": bs(3.0), "block_1": bs(),
                                  "block_2": bs(), "block_3": bs()}),
                      "lokr_baked.safetensors", True)
after = st_np.load_file(out_path)
check("lokr_w1 scaled by 3.0",
      abs(float(after["transformer.transformer_blocks.0.attn.to_q.lokr_w1"][0, 0]) - 3.0) < 1e-6)
check("lokr_w2 NOT scaled",
      float(after["transformer.transformer_blocks.0.attn.to_q.lokr_w2"][0, 0]) == 1.0)
# Default: alpha = 1, not the sentinel. The sentinel is Fizgig's convention and a
# loader that does not read it as "already applied" multiplies by ~1e6 -- a
# destroyed model with every slider still reading 1.0. Plain alpha = 1 is read
# identically by every LyCORIS loader; w1 already carries the multiplier.
check("alpha stays at 1 by default (no sentinel)",
      abs(float(after["transformer.transformer_blocks.0.attn.to_q.lokr_alpha"][0]) - 1.0) < 1e-6,
      float(after["transformer.transformer_blocks.0.attn.to_q.lokr_alpha"][0]))

# ...and the sentinel is available for a loader that wants it.
out_sent, = save.save("qwen_lokr.safetensors",
                      state_json({"block_0": bs(3.0), "block_1": bs(),
                                  "block_2": bs(), "block_3": bs()}),
                      "lokr_sentinel.safetensors", True, sentinel_alpha=True)
sent = st_np.load_file(out_sent)
check("sentinel_alpha=True writes the >=1e6 marker",
      float(sent["transformer.transformer_blocks.0.attn.to_q.lokr_alpha"][0]) >= 1e6,
      float(sent["transformer.transformer_blocks.0.attn.to_q.lokr_alpha"][0]))
check("and w1 carries the same multiplier either way",
      abs(float(sent["transformer.transformer_blocks.0.attn.to_q.lokr_w1"][0, 0])
          - float(after["transformer.transformer_blocks.0.attn.to_q.lokr_w1"][0, 0])) < 1e-6)
check("an untouched LoKR block keeps its own alpha",
      float(after["transformer.transformer_blocks.1.attn.to_q.lokr_alpha"][0]) == 1.0)
check("a LoKR file stays a LoKR file",
      "transformer.transformer_blocks.0.attn.to_q.lokr_w1" in after
      and "transformer.transformer_blocks.0.attn.to_q.lora_up.weight" not in after)

print("\nF. refusals")
glora_src = write_qwen_lora("qwen_glora.safetensors", glora=True)
out_path, = save.save("qwen_glora.safetensors",
                      state_json({"block_0": bs(2.0), "block_1": bs(),
                                  "block_2": bs(), "block_3": bs()}),
                      "glora_baked.safetensors", True)
after = st_np.load_file(out_path)
check("GLoRA keys are written unchanged",
      float(after["transformer.transformer_blocks.0.attn.to_q.glora_a"][0, 0]) == 1.0)
try:
    save.save("qwen_plain.safetensors",
              json.dumps({"blocks": {f"h3blk_{i}": bs() for i in range(4)},
                          "seed": 1, "family": "h3"}),
              "bad.safetensors", True)
    check("a foreign-family map is refused", False)
except ValueError as e:
    check("a foreign-family map is refused", "h3" in str(e) and "qwen21" in str(e), str(e))
try:
    save.save("qwen_plain.safetensors", "", "empty.safetensors", True)
    check("an empty map is refused", False)
except ValueError as e:
    check("an empty map is refused", "empty" in str(e).lower(), str(e))

print("\nG. no-op safety across the whole file")
src2 = write_qwen_lora("qwen_all.safetensors", scale=1.7)
before2 = {k: v.copy() for k, v in st_np.load_file(src2).items()}
out_path, = save.save("qwen_all.safetensors",
                      state_json({f"block_{i}": bs() for i in range(4)}),
                      "noop.safetensors", True)
after2 = st_np.load_file(out_path)
check("all tensors byte-identical when nothing moved",
      all(np.array_equal(before2[k], after2[k]) for k in before2),
      [k for k in before2 if not np.array_equal(before2[k], after2[k])])

print()
if fails:
    print(f"{len(fails)} FAILURE(S): " + "; ".join(fails))
    sys.exit(1)
print("All save-node checks passed.")
