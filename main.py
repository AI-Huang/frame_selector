import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MODELS_DIR = Path("data/models")
DEFAULT_MODEL_NAME = "yolo26x.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLO26x prediction for extracted frames."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="data/frames",
        help="Image, video, directory, glob, or stream source. Directories are scanned recursively. Default: data/frames.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Ultralytics model weights or YAML. Default: {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--project",
        default="data/yolo26x/runs",
        help="Directory where prediction results are saved.",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Run name under the project directory.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run prediction without saving annotated images or videos.",
    )
    return parser


def resolve_source(source: str) -> str | Path | list[str]:
    if source.startswith(("http://", "https://")):
        return source

    source_path = Path(source)
    if not source_path.is_dir():
        return source_path

    image_paths = sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found recursively in {source_path}")
    return [str(path) for path in image_paths]


def resolve_model(model: str) -> str:
    model_path = DEFAULT_MODEL_PATH if model == DEFAULT_MODEL_NAME else Path(model)
    if model_path.parent != Path("."):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return str(model_path)


def main() -> int:
    args = build_parser().parse_args()
    source = resolve_source(args.source)

    model = YOLO(resolve_model(args.model))
    model.predict(
        source=source,
        project=args.project,
        name=args.name,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=not args.no_save,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
