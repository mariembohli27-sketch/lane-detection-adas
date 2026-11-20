"""Before/after preprocessing visualization (CLAHE + Gaussian blur).

Usage (from the project root, with the virtual environment active):
    python src/common/visualize.py                      # first image found in data/sample/
    python src/common/visualize.py data/sample/img.jpg  # a specific image
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_first_image(folder: Path) -> Path:
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}. Put some in data/sample/.")
    return images[0]


def apply_clahe(image_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE on the lightness channel only, so colors are preserved."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def preprocess(image_bgr, size=(1280, 720), blur_kernel=(5, 5)):
    resized = cv2.resize(image_bgr, size)
    enhanced = apply_clahe(resized)
    return cv2.GaussianBlur(enhanced, blur_kernel, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="path to an image (default: first in data/sample/)")
    parser.add_argument("--out", default="reports/figures", help="output folder for the figure")
    parser.add_argument("--no-show", action="store_true", help="save only, do not open a window")
    args = parser.parse_args()

    path = Path(args.image) if args.image else find_first_image(Path("data/sample"))
    original = cv2.imread(str(path))
    if original is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    original = cv2.resize(original, (1280, 720))
    processed = preprocess(original)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, img, title in zip(axes, (original, processed), ("Original", "CLAHE + Gaussian blur")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV is BGR, Matplotlib wants RGB
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"preprocessing_{path.parent.name}_{path.stem}.png"
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
