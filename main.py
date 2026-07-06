import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from yolo_prediction import DirectorySource, predict_source

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
        help="Local image, video, or directory source. Directories are scanned recursively. Default: data/frames.",
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
        "--batch",
        type=int,
        default=1,
        help="Batch size for non-recursive sources. Recursive directories are processed one image at a time. Default: 1.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run prediction without saving annotated images or videos.",
    )
    return parser


def resolve_model(model: str) -> str:
    model_path = DEFAULT_MODEL_PATH if model == DEFAULT_MODEL_NAME else Path(model)
    if model_path.parent != Path("."):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return str(model_path)


def main() -> int:
    args = build_parser().parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(
            f"error: Source does not exist: {source_path}\n"
            "Create frames first, for example:\n"
            "  uv run python scripts/extract_all_frames.py",
            file=sys.stderr,
        )
        return 2

    source: Path | DirectorySource
    if source_path.is_dir():
        image_paths = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(
                f"error: No images found recursively in {source_path}", file=sys.stderr
            )
            return 2
        source = DirectorySource(source_path, image_paths)
    else:
        source = source_path

    model = YOLO(resolve_model(args.model))
    predict_source(
        model,
        source,
        project=args.project,
        name=args.name,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=not args.no_save,
        batch=args.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
