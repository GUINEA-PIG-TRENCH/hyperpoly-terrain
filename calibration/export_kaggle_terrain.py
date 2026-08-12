"""export_kaggle_terrain.py

Decodes one Kaggle heightmap tile into a float32 elevation field and writes it
as a flat binary so the Node fractype pipeline (material_calibration.js +
geology_quantizer.js) can consume real terrain without a PNG decoder.
"""
import os
import json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = os.path.join(HERE, "kaggle", "Heightmap_and_Textures", "train", "hmap_-100_34__0_T.png")
OUT_ELEV = os.path.join(HERE, "kaggle_field", "kaggle_elev.bin")
META = os.path.join(HERE, "kaggle_field", "terrain_meta.json")


def decode_height(path):
    im = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    diff = float(np.abs(r - g).mean())
    h16 = (g * 256.0 + r) / 65535.0
    hgray = (r + g + b) / (3.0 * 255.0)
    if diff < 1.0:
        h = hgray
        method = "grayscale(R=G=B)"
    else:
        h = h16
        method = "16bit(R+G*256)"
    return h, {"tile": os.path.basename(path), "method": method,
               "r_g_mean_abs_diff": round(diff, 4),
               "h_min": float(h.min()), "h_max": float(h.max()),
               "shape": [int(h.shape[0]), int(h.shape[1])]}


def main():
    os.makedirs(os.path.dirname(OUT_ELEV), exist_ok=True)
    h, meta = decode_height(TILE)
    h.astype("<f4").tofile(OUT_ELEV)
    with open(META, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote", OUT_ELEV, meta)


if __name__ == "__main__":
    main()
