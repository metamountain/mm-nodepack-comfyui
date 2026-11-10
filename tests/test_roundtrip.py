"""Unit tests for the pure helpers of roundtrip_nodes (no ComfyUI server needed).

Run:
  D:\\ComfyUI_windows_portable\\python_embeded\\python.exe D:\\Coding\\Comfyui\\comfyui-image-roundtrip\\tests\\test_roundtrip.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from roundtrip_nodes import (  # noqa: E402
    IMAGE_EXTS,
    load_image_file,
    make_source,
    next_target,
    save_tensor,
)


def check(cond, msg):
    assert cond, msg
    print(f"ok: {msg}")


def main():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "haus.png")
        Image.new("RGB", (8, 5), (200, 30, 30)).save(src)

        # next_target default: haus_1.png, then haus_2.png
        t1 = next_target(src)
        check(t1 == os.path.join(d, "haus_1.png"), f"next_target(src) -> {os.path.basename(t1)}")
        open(os.path.join(d, "haus_1.png"), "wb").close()
        t2 = next_target(src)
        check(t2 == os.path.join(d, "haus_2.png"), f"next free number -> {os.path.basename(t2)}")

        # suffix without '#': haus_edit.png, then haus_edit_2.png
        t3 = next_target(src, suffix="_edit")
        check(t3 == os.path.join(d, "haus_edit.png"), f"suffix _edit -> {os.path.basename(t3)}")
        open(t3, "wb").close()
        t4 = next_target(src, suffix="_edit")
        check(t4 == os.path.join(d, "haus_edit_2.png"), f"existing _edit -> {os.path.basename(t4)}")

        # save_as: relative name, subfolder with ext, absolute path
        t5 = next_target(src, save_as="neu")
        check(t5 == os.path.join(d, "neu.png"), f"save_as neu -> {t5}")
        t6 = next_target(src, save_as=os.path.join("sub", "x.jpg"))
        check(t6 == os.path.join(d, "sub", "x.jpg"), f"save_as sub\\x.jpg -> {t6}")
        absa = os.path.join(d, "abs.png")
        t7 = next_target(src, save_as=absa)
        check(t7 == absa, "absolute save_as used as-is")
        open(absa, "wb").close()
        t8 = next_target(src, save_as=absa)
        check(t8 == os.path.join(d, "abs_2.png"), f"existing absolute save_as -> {os.path.basename(t8)}")

        # format: jpg, and 'same as source' keeps .webp
        t9 = next_target(src, fmt="jpg")
        check(t9.endswith(".jpg"), f"fmt jpg -> {os.path.basename(t9)}")
        webp = os.path.join(d, "bild.webp")
        Image.new("RGB", (4, 4)).save(webp)
        t10 = next_target(webp)
        check(t10.endswith(".webp"), f"same as source webp -> {os.path.basename(t10)}")

        # save_tensor: readable PNG and JPG with the right size
        arr = torch.rand(5, 8, 3)
        png = os.path.join(d, "out.png")
        save_tensor(arr, png)
        with Image.open(png) as im:
            check(im.size == (8, 5) and im.mode == "RGB", "save_tensor png readable, size (8,5)")
        jpg = os.path.join(d, "out.jpg")
        save_tensor(arr, jpg, quality=90)
        with Image.open(jpg) as im:
            check(im.size == (8, 5), "save_tensor jpg readable, size (8,5)")
        # new folder (save_as with a new folder) is created
        deep = os.path.join(d, "a", "b", "c.png")
        save_tensor(arr, deep)
        check(os.path.isfile(deep), "save_tensor creates missing folders")

        # load_image_file: RGB + mask
        img, mask = load_image_file(src)
        check(tuple(img.shape) == (1, 5, 8, 3) and img.dtype == torch.float32, "load_image_file shape/dtype")
        check(mask.shape[0] == 1, "load_image_file mask")

        # make_source (NACHTRAG D): path/width/height/format/name
        s = make_source(src)
        check(s["path"] == os.path.abspath(src), "make_source path")
        check(s["width"] == 8 and s["height"] == 5, "make_source width/height")
        check(s["format"] == ".png", "make_source format")
        check(s["name"] == "haus", "make_source name")
        check(tuple(s["image"].shape) == (1, 5, 8, 3), "make_source image tensor")
        check("mask" in s and "path" in s and "image" in s, "make_source keys")

        check(IMAGE_EXTS == (".png", ".jpg", ".jpeg", ".webp"), "IMAGE_EXTS")

    print("all ok")


if __name__ == "__main__":
    main()
