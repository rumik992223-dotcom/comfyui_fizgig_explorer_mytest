"""comfyui_fizgig_explorer -- LoRA the Explorer, ported from Fizgig.

Upstream: https://github.com/shootthesound/Fizgig (Apache-2.0)
This port keeps the mutation recipe, the block map and the checkpoint scanner;
the render and the bake stay in ComfyUI's own nodes. See explorer_core.py for
the file-by-file provenance.
"""

import os

# A folder of its own for bakes, so FizgigLoraSave can write the picked variant
# and the Explorer can read it back by name. Entirely optional -- an absolute
# path works too.
try:
    import folder_paths

    _BAKES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bakes")
    os.makedirs(_BAKES, exist_ok=True)
    folder_paths.add_model_folder_path("fizgig_loras", _BAKES)
except Exception:            # pragma: no cover -- ComfyUI always provides it
    pass

# The nodes are imported here, and a failure here is worth a loud message rather
# than a traceback nobody reads: ComfyUI registers nothing from a pack whose
# __init__ raises, so the UI reports every node in a saved workflow as a
# "Missing node type" and adds them to its "Unknown pack" list. The usual cause
# is a file set that is out of sync -- nodes.py imports helpers such as
# state_from_baked and strength_from_pair from explorer_core.py, and an older
# explorer_core.py does not have them.
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:                                  # noqa: BLE001
    import traceback

    traceback.print_exc()
    raise RuntimeError(
        "comfyui_fizgig_explorer: nodes.py could not be imported, so NONE of "
        "this pack's nodes are registered -- the UI will report every Fizgig "
        "node in your workflow as a missing node type.\n"
        f"  {type(_e).__name__}: {_e}\n"
        "  Most likely a file set that is out of sync. Update every .py in "
        f"{os.path.dirname(os.path.abspath(__file__))!r} together: nodes.py "
        "imports helpers (state_from_baked, strength_from_pair, ...) that only "
        "exist in a matching explorer_core.py."
    ) from _e

__version__ = "0.5.6"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
