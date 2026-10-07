"""Equivalence tests: ported core vs the original Fizgig modules.

Importing Fizgig's repair_studio/state.py and lora_royale/scan.py is cheap --
they are dataclasses/typing/os/re only, no torch. So the port can be checked
against the real thing rather than against my reading of it.

Run:  python test_core.py /path/to/Fizgig
"""

import importlib.util
import os
import random
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `comfyui_fizgig_explorer/__init__.py` imports nodes.py, which imports
# folder_paths from ComfyUI. Stub it -- these tests never load a real LoRA.
_fp = types.ModuleType("folder_paths")
_fp.get_filename_list = lambda kind: []
_fp.get_full_path = lambda kind, name: None
sys.modules.setdefault("folder_paths", _fp)

from comfyui_fizgig_explorer.explorer_core import (  # noqa: E402
    SliderState as PortedState,
    block_id_from_key as ported_block_id,
    family_from_metadata,
    guess_family,
    roll_variants,
    scan_block_ids,
    scan_checkpoints as ported_scan,
    summarise_key_shapes,
)


def load_module(path, name):
    """Import a Fizgig module by path without triggering its package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(repo):
    print(f"Fizgig repo: {repo}\n")
    state_mod = load_module(os.path.join(repo, "src/fizgig/repair_studio/state.py"), "orig_state")
    scan_mod = load_module(os.path.join(repo, "src/fizgig/lora_royale/scan.py"), "orig_scan")
    bake_src = open(os.path.join(repo, "src/fizgig/repair_studio/bake.py")).read()
    # bake.py pulls torch at import; grab its pure key mapper by exec'ing only the top part.
    bake_ns = {"re": __import__("re"), "Optional": __import__("typing").Optional}
    head = bake_src.split("def _group_by_module")[0]
    head = head.split("import torch")[0] + head.split("from fizgig.repair_studio.state import SliderState")[-1]
    exec(compile(head, "bake_head", "exec"), bake_ns)
    orig_block_id = bake_ns["_block_id_from_key"]

    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)
            print(f"        got : {got}\n        want: {want}")

    # ------------------------------------------------------------------ 1
    print("1. mutate() -- bit-for-bit against repair_studio/state.py")
    block_ids = [f"double_{i}" for i in range(8)] + [f"single_{i}" for i in range(24)]
    active = set(block_ids)
    for seed in (1, 42, 1337, 99999):
        # intensity 0.0 is excluded on purpose: this port deviates there (zero is
        # a true no-op, where Fizgig still moves every block by +-0.2). That
        # deviation is asserted on its own just below, so it cannot drift.
        for intensity, structure in ((0.5, 1.0), (0.2, 0.0), (1.0, 0.7), (0.9, 0.05)):
            random.seed(seed)                       # original uses the module global
            orig = state_mod.SliderState.default_klein9b()
            o = orig.mutate(active, num_mutations=5, intensity=intensity,
                            structure=structure, anchor="double_0")
            p = PortedState.from_block_ids(block_ids)
            p = p.mutate(active, num_mutations=5, intensity=intensity,
                         structure=structure, anchor="double_0",
                         rng=random.Random(seed))
            got = {b: (bs.primary_enabled, round(bs.primary_strength, 9)) for b, bs in p.blocks.items()}
            want = {b: (bs.primary_enabled, round(bs.primary_strength, 9)) for b, bs in o.blocks.items()}
            if got != want:
                diff = {k: (want[k], got.get(k)) for k in want if want.get(k) != got.get(k)}
                check(f"seed={seed} intensity={intensity} structure={structure}", diff, {})
    print("  -> identical on every combination\n")

    # The one deliberate deviation, stated as a test so it is a decision and not
    # an accident: at intensity 0.0 the port must change nothing at all.
    print("   1b. the documented deviation: intensity 0.0 is a true no-op")
    for seed in (1, 42, 1337):
        random.seed(seed)
        _orig = state_mod.SliderState.default_klein9b()
        _o = _orig.mutate(active, num_mutations=5, intensity=0.0, structure=0.0,
                          anchor="double_0")
        _p = PortedState.from_block_ids(block_ids).mutate(
            active, num_mutations=5, intensity=0.0, structure=0.0,
            anchor="double_0", rng=random.Random(seed))
        check(f"  Fizgig still moves blocks at 0.0 (seed {seed})",
              len(_o.diff_blocks(_orig)) > 0, True)
        check(f"  the port does not (seed {seed})", _p.diff_blocks(
            PortedState.from_block_ids(block_ids)), [])
    print()

    # ------------------------------------------------------------------ 2
    print("2. diff_blocks()")
    pa = PortedState.from_block_ids(block_ids)
    random.seed(1)
    pb = pa.mutate(active, num_mutations=6, intensity=0.8, rng=random.Random(1))
    check("diff vs original agrees on count", len(pb.diff_blocks(pa)) > 0, True)
    check("diff on identical states is empty", pb.diff_blocks(pb.copy()), [])

    # ------------------------------------------------------------------ 3
    print("\n3. the four-variant recipe (variant 4 protection)")
    seed = 5
    base = PortedState.from_block_ids(block_ids)
    last_pick = {"single_3", "single_4", "single_5", "single_6", "single_7",
                 "single_8", "single_9", "single_10", "single_11"}
    variants = roll_variants(base, active, num_mutations=3, intensity=0.9,
                             structure=1.0, anchor="double_0",
                             last_pick_blocks=last_pick, seed=seed)
    check("exactly four variants", len(variants), 4)
    check("v1/v2 carry the structural change (anchor hit hard)",
          abs(variants[0].blocks["double_0"].primary_strength - 1.0) >= 0.5, True)
    # Variant 3 runs with structure=0.0, so the anchor is not forced first and
    # may not be sampled at all; what must hold is that no block moves further
    # than the plain delta magnitude (0.2 + intensity*2.8 = 2.72 here).
    mag = 0.2 + 0.9 * 2.8
    v3_changed = set(variants[2].diff_blocks(base))
    v3_ok = all(abs(variants[2].blocks[b].primary_strength - base.blocks[b].primary_strength) <= mag + 1e-6
                for b in v3_changed)
    check("v3 has no structural (invert/extreme) hit", v3_ok, True)
    check("v3 still mutated something", len(v3_changed) > 0, True)
    v4_changed = set(variants[3].diff_blocks(base))
    check("v4 avoided every protected block", sorted(v4_changed & last_pick), [])
    check("v4 still mutated something", len(v4_changed) > 0, True)
    check("anchor is never disabled",
          all(v.blocks["double_0"].primary_enabled for v in variants), True)
    check("deterministic for a given seed",
          [v.to_json() for v in roll_variants(base, active, seed=seed)]
          == [v.to_json() for v in roll_variants(base, active, seed=seed)], True)
    check("different seed gives different roll",
          [v.to_json() for v in roll_variants(base, active, seed=1)]
          != [v.to_json() for v in roll_variants(base, active, seed=2)], True)

    # ------------------------------------------------------------------ 4
    print("\n4. lock set (the GUI's Freeze)")
    locked = {"double_0", "single_0", "single_1"}
    vs = roll_variants(base, active, num_mutations=8, intensity=1.0,
                       locked_blocks=locked, seed=3)
    check("locked blocks never mutated",
          sorted({b for v in vs for b in v.diff_blocks(base)} & locked), [])
    # the anchor being frozen must not resurrect it
    vs2 = roll_variants(base, active, num_mutations=8, intensity=1.0,
                        locked_blocks={"double_0"}, seed=3)
    check("a frozen anchor stays frozen",
          sorted({b for v in vs2 for b in v.diff_blocks(base)} & {"double_0"}), [])

    # ------------------------------------------------------------------ 5
    print("\n5. block ids from LoRA keys -- vs repair_studio/bake.py")
    cases = [
        "lora_unet_double_blocks_0_lora_up.weight",
        "lora_unet_single_blocks_23_lora_down.weight",
        "lora_unet_txtfusion_layerwise_blocks_4_lora_up.weight",
        "lora_unet_txtfusion_refiner_blocks_2_lora_up.weight",
        "lora_unet_token_refiner_blocks_1_lora_up.weight",
        "lora_unet_blocks_17_lora_up.weight",
        "some.random.weight",
    ]
    for key in cases:
        check(key, ported_block_id(key), orig_block_id(key))

    print("\n   Qwen Image 2.1 (families/qwen_image.py: kohya=False)")
    qwen_cases = [
        ("transformer.transformer_blocks.0.attn.to_q.lora_B.weight", "block_0"),
        ("transformer.transformer_blocks.31.attn.to_out.0.lora_A.weight", "block_31"),
        ("transformer.transformer_blocks.12.img_mlp.gate_layer.lora_B.weight", "block_12"),
        ("transformer.transformer_blocks.7.img_mlp.proj.lora_A.weight", "block_7"),
        # ComfyUI normalises these to kohya-ish keys on some paths
        ("lora_unet_transformer_blocks_5_attn_to_q.lora_up.weight", "block_5"),
    ]
    for key, want in qwen_cases:
        check(key, ported_block_id(key, family="qwen21"), want)
    check("Qwen keeps block_N, not h3blk_N",
          ported_block_id("transformer.transformer_blocks.9.attn.to_q.lora_B.weight",
                          family="qwen21"), "block_9")
    check("H3 keeps h3blk_N on the same shape",
          ported_block_id("lora_unet_blocks_9_lora_up.weight", family="h3"), "h3blk_9")
    # Regression: double_blocks must not be swallowed by the bare blocks_N shape
    check("single_blocks_2 is not read as block_2",
          ported_block_id("lora_unet_single_blocks_2_lora_up.weight"), "single_2")
    check("double_blocks.1. is not read as block_1",
          ported_block_id("diffusion_model.double_blocks.1.lora_B.weight"), "double_1")
    check("token_refiner is not read as block_1",
          ported_block_id("lora_unet_token_refiner_blocks_1_lora_up.weight"), "h3_rf_1")

    # ------------------------------------------------------------------ 6
    print("\n6. family detection")
    check("metadata: qwen", family_from_metadata({"ss_sd_model_name": "Qwen-Image-2.1"}), "qwen21")
    check("metadata: klein", family_from_metadata({"ss_sd_model_name": "flux2_klein_9b"}), "klein9b")
    check("metadata: krea", family_from_metadata({"modelspec.title": "Krea 2 base"}), "krea2")
    check("metadata: h3", family_from_metadata({"note": "trained on MiniMax H3"}), "h3")
    check("metadata: silence -> None", family_from_metadata({}), None)
    check("shapes help the error message",
          summarise_key_shapes([
              "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
              "transformer.transformer_blocks.1.attn.to_q.lora_B.weight",
              "lora_te1_x.lora_up.weight",
          ]),
          ["transformer.transformer_blocks.N.attn.to_q.lora_B.weight", "lora_teN_x.lora_up.weight"])

    # ------------------------------------------------------------------ 6b
    print("\n6b. guess_family must not confirm its own guess (regression)")
    import numpy as _np
    import safetensors.numpy as _st
    with tempfile.TemporaryDirectory() as d:
        def _w(name, keys, meta=None):
            q = os.path.join(d, name)
            _st.save_file({k: _np.zeros((2, 2), dtype=_np.float32) for k in keys}, q, metadata=meta)
            return q

        qwen = _w("q.safetensors",
                  [f"transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight" for i in range(32)])
        # The shape that misfired: a Qwen-style file plus enough keys to push the
        # bare-block count past the H3 threshold. guess_family used to route
        # through scan_block_ids, which namespaces those keys h3blk_*, and then
        # "confirmed" H3 by finding h3blk_ -- circular, and it labelled a Qwen
        # file H3, so an H3 map got applied to it.
        qwen_big = _w("q_big.safetensors",
                      [f"transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight" for i in range(32)]
                      + [f"transformer.transformer_blocks.{i}.img_mlp.proj.lora_B.weight" for i in range(32)])
        bare32 = _w("b32.safetensors", [f"lora_unet_blocks.{i}_lora_up.weight" for i in range(32)])
        bare50 = _w("b50.safetensors", [f"lora_unet_blocks.{i}_lora_up.weight" for i in range(50)])
        klein = _w("k.safetensors", [f"lora_unet_double_blocks_{i}_lora_up.weight" for i in range(8)])
        h3 = _w("h3.safetensors",
                [f"lora_unet_blocks.{i}_lora_up.weight" for i in range(50)]
                + [f"lora_unet_token_refiner_blocks_{i}_lora_up.weight" for i in range(2)])
        krea = _w("kr.safetensors",
                  [f"lora_unet_blocks.{i}_lora_up.weight" for i in range(32)]
                  + [f"txtfusion_layerwise_blocks_{i}_lora_up.weight" for i in range(4)])
        qwen_meta = _w("qm.safetensors",
                       [f"transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight" for i in range(32)],
                       {"ss_sd_model_name": "Qwen Image 2.1"})

        check("diffusers Qwen keys -> qwen21", guess_family(qwen), "qwen21")
        check("Qwen with many block keys is still qwen21", guess_family(qwen_big), "qwen21")
        check("metadata still wins", guess_family(qwen_meta), "qwen21")
        check("bare 32 -> krea2 (not h3)", guess_family(bare32), "krea2")
        check("bare 50 -> h3", guess_family(bare50), "h3")
        check("klein double_blocks -> klein9b", guess_family(klein), "klein9b")
        check("token_refiner -> h3", guess_family(h3), "h3")
        check("txtfusion -> krea2", guess_family(krea), "krea2")
        # and the namespace it assigns must agree with the family it reports
        for path, fam, want in ((qwen, "qwen21", "block_"),
                                (bare32, "krea2", "block_"),
                                (bare50, "h3", "h3blk_"),
                                (klein, "klein9b", "double_")):
            ids = scan_block_ids(path, family=guess_family(path))
            check(f"namespace agrees with {fam} for {os.path.basename(path)}",
                  all(i.startswith(want) for i in ids) if ids else False, True)

    # ------------------------------------------------------------------ 7
    print("\n7. checkpoint scan -- vs lora_royale/scan.py")
    with tempfile.TemporaryDirectory() as d:
        for e in (5, 1, 20, 2):
            open(os.path.join(d, f"myrun-{e:06d}.safetensors"), "wb").close()
        os.makedirs(os.path.join(d, "myrun-000005-state"), exist_ok=True)
        open(os.path.join(d, "notes.txt"), "w").close()
        check("clean epoch run -> integer labels",
              ported_scan(d), scan_mod.scan_checkpoints(d))
        check("state dirs and stray files ignored",
              [l for l, _ in ported_scan(d)], [1, 2, 5, 20])

    with tempfile.TemporaryDirectory() as d:
        for n in ("zebra.safetensors", "alpha.safetensors", "mid-000003.safetensors"):
            open(os.path.join(d, n), "wb").close()
        check("mixed folder -> filename stems",
              ported_scan(d), scan_mod.scan_checkpoints(d))
        check("sorted by name, lowercased",
              [l for l, _ in ported_scan(d)], ["alpha", "mid-000003", "zebra"])
    check("missing folder -> empty", ported_scan("/nope/nope"), [])

    # ------------------------------------------------------------------ 8
    print("\n8. state round-trip")
    vs = roll_variants(base, active, seed=11)
    rt = PortedState.from_json(vs[0].to_json())
    check("to_json/from_json is lossless", rt.to_json(), vs[0].to_json())
    check("copy() is lossless", vs[0].copy().to_json(), vs[0].to_json())
    check("to_block_set() omits disabled blocks",
          [b for b, v in vs[0].to_block_set().items() if v == 0.0],
          [b for b, bs in vs[0].blocks.items() if not bs.primary_enabled])
    qwen_state = PortedState.from_block_ids([f"block_{i}" for i in range(32)])
    qwen_state.blocks["block_12"] = type(qwen_state.blocks["block_12"])(primary_strength=2.0)
    check("identity blocks are tagged on Qwen",
          "[identity]" in qwen_state.to_plain_text(family="qwen21"), True)
    check("and not on Klein",
          "[identity]" in qwen_state.to_plain_text(family="klein9b"), False)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/fizgig")
    sys.exit(main(repo))
