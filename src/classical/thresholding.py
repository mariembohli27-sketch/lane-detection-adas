"""Classical pipeline, part 1: binary mask of the lane markings.

Steps: preprocess -> color mask (white + yellow) -> gradient mask (Sobel x)
       -> combine -> region of interest.

Usage (from the project root, virtual environment active):
    python src/classical/thresholding.py
    python src/classical/thresholding.py "data\\sample\\...\\20.jpg"
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow "from src.common.visualize import ..." when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.visualize import find_first_image, preprocess  # noqa: E402

# ---- Parameters to tune (change one at a time and compare the figure) ----
WHITE_L_MIN = 200                 # HLS lightness above which a pixel counts as white
YELLOW_H = (15, 35)               # OpenCV hue range for yellow (0-179)
YELLOW_S_MIN, YELLOW_V_MIN = 80, 120
SOBEL_THRESH = (30, 255)          # gradient magnitude range (0-255 after scaling)
ROI_HORIZON = 0.40                # fraction of image height where the ROI starts


def color_mask(image_bgr):
    """White markings (high lightness) OR yellow markings (hue + saturation)."""
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = hls[:, :, 1] >= WHITE_L_MIN
    yellow = (
        (hsv[:, :, 0] >= YELLOW_H[0]) & (hsv[:, :, 0] <= YELLOW_H[1])
        & (hsv[:, :, 1] >= YELLOW_S_MIN) & (hsv[:, :, 2] >= YELLOW_V_MIN)
    )
    return (white | yellow).astype(np.uint8) * 255


def gradient_mask(image_bgr):
    """Horizontal Sobel on the HLS lightness channel: lane edges are mostly vertical."""
    l_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)[:, :, 1]
    sobel_x = np.abs(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3))
    scaled = np.uint8(255 * sobel_x / max(sobel_x.max(), 1))
    return ((scaled >= SOBEL_THRESH[0]) & (scaled <= SOBEL_THRESH[1])).astype(np.uint8) * 255


def region_of_interest(mask):
    """Keep only a trapezoid in front of the vehicle (removes sky and roadside)."""
    h, w = mask.shape[:2]
    polygon = np.array([[(0, h), (int(0.40 * w), int(ROI_HORIZON * h)),
                         (int(0.60 * w), int(ROI_HORIZON * h)), (w, h)]], dtype=np.int32)
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, polygon, 255)
    return cv2.bitwise_and(mask, roi), polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="path to an image (default: first in data/sample/)")
    parser.add_argument("--out", default="reports/figures")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    path = Path(args.image) if args.image else find_first_image(Path("data/sample"))
    if not path.is_file():
        raise FileNotFoundError(f"Not an image file: {path} (the path must end with the file name, e.g. 20.jpg)")
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"File exists but is not a valid image: {path}")

    image = preprocess(cv2.resize(raw, (1280, 720)))
    color = color_mask(image)
    grad = gradient_mask(image)
    combined = cv2.bitwise_or(color, grad)
    final, polygon = region_of_interest(combined)

    overlay = image.copy()
    cv2.polylines(overlay, polygon, True, (0, 255, 0), 3)

    panels = [
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "Preprocessed + ROI"),
        (color, "Color mask (white + yellow)"),
        (grad, "Gradient mask (Sobel x)"),
        (final, "Combined + ROI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mask_{path.parent.name}_{path.stem}.png"
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
