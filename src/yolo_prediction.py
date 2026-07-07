from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path("data")
ImagePath = str | Path


class YOLOInferencer:
    """
    Run YOLO inference and save visualization images.

    :param model: Loaded Ultralytics YOLO model.
    :param output_dir: Directory where YOLO visualization images are saved.
    :param conf: Confidence threshold.
    :param imgsz: YOLO inference input size. Larger values preserve more detail,
        but use more GPU memory.
    :param device: Inference device, for example ``cpu``, ``0``, or ``0,1``.
    :param save: Whether to save visualization images or videos.
    """

    def __init__(
        self,
        model: YOLO,
        *,
        output_dir: str | Path,
        data_dir: str | Path = DATA_DIR,
        conf: float,
        imgsz: int,
        device: str | None,
        save: bool,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.save = save

    def predict(self, image_path: ImagePath, *, name: str) -> None:
        self.model.predict(
            source=str(image_path),
            project=str(self.output_dir.resolve()),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            save=self.save,
            exist_ok=True,
            name=self._prediction_name(name, image_path),
        )

    def predict_batch(
        self,
        image_paths: list[ImagePath],
        *,
        name: str,
        batch: int,
    ) -> None:
        images_by_run_name: dict[str, list[str]] = {}
        for image_path in image_paths:
            run_name = self._prediction_name(name, image_path)
            images_by_run_name.setdefault(run_name, []).append(str(image_path))

        for run_name, run_image_paths in images_by_run_name.items():
            self.model.predict(
                source=run_image_paths,
                batch=max(batch, 1),
                project=str(self.output_dir.resolve()),
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                save=self.save,
                exist_ok=True,
                name=run_name,
            )

    def _prediction_name(self, name: str, image_path: ImagePath) -> str:
        base_name = name.strip("/")
        relative_parent = self._data_relative_parent(image_path)
        if relative_parent == Path("."):
            return base_name
        return str(Path(base_name) / relative_parent)

    def _data_relative_parent(self, image_path: ImagePath) -> Path:
        parent = Path(image_path).parent.resolve()
        data_dir = self.data_dir.resolve()
        try:
            return parent.relative_to(data_dir)
        except ValueError:
            return Path(".")


def predict_source(
    model: YOLO,
    source: ImagePath | list[ImagePath],
    *,
    output_dir: str | Path,
    data_dir: str | Path = DATA_DIR,
    name: str,
    conf: float,
    imgsz: int,
    device: str | None,
    save: bool,
    batch: int,
) -> None:
    """
    Run YOLO inference and save visualization images.

    :param model: Loaded Ultralytics YOLO model.
    :param source: Local image path or image path list.
    :param output_dir: Directory where YOLO visualization images are saved.
    :param data_dir: Runtime data directory used to derive stable output names.
    :param name: Run name under the output directory.
    :param conf: Confidence threshold.
    :param imgsz: YOLO inference input size. Larger values preserve more detail,
        but use more GPU memory.
    :param device: Inference device, for example ``cpu``, ``0``, or ``0,1``.
    :param save: Whether to save visualization images or videos.
    :param batch: Batch size for image path lists.
    """
    inferencer = YOLOInferencer(
        model,
        output_dir=output_dir,
        data_dir=data_dir,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=save,
    )
    if isinstance(source, list):
        inferencer.predict_batch(source, name=name, batch=batch)
        return

    inferencer.predict(source, name=name)
