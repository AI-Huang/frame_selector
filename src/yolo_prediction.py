from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DirectorySource:
    root: Path
    image_paths: list[Path]


def consume_prediction(results) -> None:
    for _ in results:
        pass


def predict_source(
    model: Any,
    source: str | Path | DirectorySource,
    *,
    project: str,
    name: str,
    conf: float,
    imgsz: int,
    device: str | None,
    save: bool,
    batch: int,
) -> None:
    """
    :param imgsz: YOLO inference input size. Larger values preserve more detail,
        but use more GPU memory.
    """
    predict_kwargs = {
        "project": str(Path(project).resolve()),
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "save": save,
        "exist_ok": True,
        "stream": True,
    }

    if isinstance(source, DirectorySource):
        base_name = name.strip("/")
        for image_path in source.image_paths:
            relative_parent = image_path.parent.relative_to(source.root)
            run_name = str(Path(base_name) / relative_parent)
            consume_prediction(
                model.predict(source=str(image_path), **predict_kwargs, name=run_name)
            )
        return

    consume_prediction(
        model.predict(
            source=source,
            batch=max(batch, 1),
            **predict_kwargs,
            name=name,
        )
    )
