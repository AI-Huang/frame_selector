import argparse
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DirectorySource:
    root: Path
    image_paths: list[Path]


def has_glob_magic(source: str) -> bool:
    return any(char in source for char in "*?[")


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


def resolve_source(source: str) -> str | Path | DirectorySource:
    if source.startswith(("http://", "https://")):
        return source

    source_path = Path(source)
    if source_path.exists() and not source_path.is_dir():
        return source_path

    if not source_path.is_dir():
        if has_glob_magic(source):
            return source_path
        raise FileNotFoundError(
            f"Source does not exist: {source_path}\n"
            "Create frames first, for example:\n"
            "  uv run python scripts/extract_all_frames.py\n"
            "or pass an existing image, video, directory, glob, or URL."
        )

    image_paths = sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found recursively in {source_path}")
    return DirectorySource(source_path, image_paths)


def resolve_model(model: str) -> str:
    model_path = DEFAULT_MODEL_PATH if model == DEFAULT_MODEL_NAME else Path(model)
    if model_path.parent != Path("."):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return str(model_path)


def consume_prediction(results) -> None:
    for _ in results:
        pass


def main() -> int:
    args = build_parser().parse_args()
    try:
        source = resolve_source(args.source)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    model = YOLO(resolve_model(args.model))
    predict_kwargs = {
        "project": str(Path(args.project).resolve()),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "save": not args.no_save,
        "exist_ok": True,
        "stream": True,
    }

    if isinstance(source, DirectorySource):
        base_name = args.name.strip("/")
        for image_path in source.image_paths:
            relative_parent = image_path.parent.relative_to(source.root)
            run_name = str(Path(base_name) / relative_parent)
            consume_prediction(
                model.predict(source=str(image_path), **predict_kwargs, name=run_name)
            )
    else:
        consume_prediction(
            model.predict(
                source=source,
                batch=max(args.batch, 1),
                **predict_kwargs,
                name=args.name,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
