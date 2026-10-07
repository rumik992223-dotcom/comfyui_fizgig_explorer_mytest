"""Dump a LoRA's key layout and metadata -- what the Explorer sees.

    python inspect_lora.py path/to/file.safetensors

Prints the metadata, the block keys grouped by stem, the family this package
would read, the block ids it would use, and how many tensors each module carries.
That is everything needed to tell a Qwen file read as H3 from a real H3 one.
"""
import collections
import json
import os
import re
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def main(path):
    h = read_header(path)
    meta = h.pop("__metadata__", {}) or {}
    keys = sorted(h)

    print(f"file: {os.path.basename(path)}")
    print(f"tensors: {len(keys)}")
    print()
    print("metadata:")
    for k, v in meta.items():
        print(f"    {k} = {v}")
    if not meta:
        print("    (none -- family detection falls back to key shape)")
    print()

    # which naming style the file actually uses
    styles = {
        "kohya bare blocks_N (lora_unet_blocks_N_)": r"^lora_unet_blocks_(\d+)_",
        "kohya Klein (double/single_blocks)":        r"lora_unet_(double|single)_blocks_",
        "kohya token_refiner (H3)":                  r"lora_unet_token_refiner_blocks_(\d+)_",
        "kohya txtfusion (Krea 2)":                  r"txtfusion_",
        "diffusers (transformer_blocks.N.)":         r"transformer\.transformer_blocks\.(\d+)\.",
    }
    print("naming styles present:")
    nums = {}
    for label, pat in styles.items():
        hits = [k for k in keys if re.search(pat, k)]
        if hits:
            got = [int(m.group(1)) for k in hits
                   for m in [re.search(pat, k)] if m and m.groups()]
            nums[label] = (len(hits), max(got) if got else None)
            print(f"    {label}: {len(hits)} tensors"
                  + (f", block indices 0..{max(got)}" if got else ""))
    if not nums:
        print("    (no recognised block naming -- this file adapts nothing the")
        print("     Explorer can address by block)")
    print()

    # what each block carries
    per_block = collections.Counter()
    for k in keys:
        m = re.search(r"(?:blocks_|\.)(\d+)(?:[_.]|$)", k)
        if m:
            per_block[int(m.group(1))] += 1
    if per_block:
        counts = collections.Counter(per_block.values())
        print("tensors per block index:")
        for per, howmany in sorted(counts.items(), reverse=True):
            print(f"    {per} tensors x {howmany} block(s)")
        print(f"    block indices: {min(per_block)}..{max(per_block)}"
              f"  ({len(per_block)} blocks)")
    print()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from comfyui_fizgig_explorer import explorer_core as core
    except Exception:
        import explorer_core as core

    fam = core.guess_family(path)
    print(f"family this package reads: {fam}")
    if meta.get("ss_sd_model_name"):
        print(f"    (metadata ss_sd_model_name says {meta['ss_sd_model_name']!r})")
    ids = core.scan_block_ids(path, family=fam)
    print(f"block ids it would use: {len(ids)}  {ids[:4]}{' ...' if len(ids) > 4 else ''}")
    for other in ("qwen21", "krea2", "h3", "klein9b"):
        if other == fam:
            continue
        other_ids = core.scan_block_ids(path, family=other)
        if other_ids:
            print(f"    as {other}: {len(other_ids)} ids, {other_ids[:3]} ...")
    print()
    print("If the family above is wrong for the model you are sampling, set the")
    print("`family` widget on the Explorer and the loader explicitly -- an H3 map")
    print("applied to a Qwen file scales tensors under the wrong names.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1])
