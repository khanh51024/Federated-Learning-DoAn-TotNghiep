"""One shared rule for turning a decoded (EXIF-normalised) PIL image into RGB, used by training and by the web app.

Transparent images are composited onto black. This is exactly the rule CanonicalClassificationDataset has applied
since v7 (dataset.py, before Q3A-1); the web app used to call .convert("RGB") directly, which drops alpha
differently. Both now call this function, so training and inference see the same pixels."""
from PIL import Image


def to_rgb_on_black(image):
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (0, 0, 0))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return image.convert("RGB")
