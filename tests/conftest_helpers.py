"""Reusable test helpers for the MM Nodepack unit tests (no ComfyUI runtime needed)."""
import os
import sys
import types


def make_folder_paths_stub(input_dir, output_dir, temp_dir=None, input_files=None):
    """Build a fake `folder_paths` module backed by real directories (like ComfyUI's).

    get_annotated_filepath maps a bare name to <input_dir>/<name> and strips the
    ' [input]' suffix ComfyUI appends to upload combo values.
    """
    if temp_dir is None:
        raise ValueError("temp_dir is required (the pack caches preview thumbnails there)")
    mod = types.ModuleType("folder_paths")

    def _annotated(name):
        name = (name or "").strip()
        if name.endswith("[input]"):
            name = name[: -len("[input]")].strip()
        return os.path.join(input_dir, name)

    mod.get_input_directory = lambda: input_dir
    mod.get_output_directory = lambda: output_dir
    mod.get_temp_directory = lambda: temp_dir
    mod.get_annotated_filepath = _annotated
    mod.filter_files_content_types = lambda files, content_types: list(files or [])
    if input_files is not None:
        mod.get_input_files = lambda: list(input_files)
    return mod


class FolderPathsStub:
    """Install a fake folder_paths module in sys.modules; removes it on exit."""

    def __init__(self, module):
        self.module = module

    def __enter__(self):
        sys.modules["folder_paths"] = self.module
        return self.module

    def __exit__(self, *exc):
        sys.modules.pop("folder_paths", None)
        return False
