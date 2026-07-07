from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "inference" / "yolo26x" / "results"


class YOLOInferencer:
    """
    Run YOLO inference and save visualization images.

    :param model: Loaded Ultralytics YOLO model.
    :param output_dir: Directory where YOLO visualization images are saved.
    :param results_dir: Directory where YOLO JSON results are saved.
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
        results_dir: str | Path = RESULTS_DIR,
        data_dir: str | Path = DATA_DIR,
        conf: float,
        imgsz: int,
        device: str | None,
        save: bool,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.save = save

    def predict(self, image_path: str | Path, *, name: str) -> list[Results]:
        results = self.model.predict(
            source=str(image_path),
            project=str(self.output_dir.resolve()),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            save=self.save,
            exist_ok=True,
            name=self._prediction_name(name, image_path),
        )
        self.save_results(results)
        return results

    def predict_batch(
        self,
        image_paths: list[str | Path],
        *,
        name: str,
        batch: int,
    ) -> list[Results]:
        images_by_run_name: dict[str, list[str]] = {}
        for image_path in image_paths:
            run_name = self._prediction_name(name, image_path)
            images_by_run_name.setdefault(run_name, []).append(str(image_path))

        results: list[Results] = []
        for run_name, run_image_paths in images_by_run_name.items():
            results.extend(
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
            )
        self.save_results(results)
        return results

    def save_results(self, results: list[Results]) -> None:
        for result in results:
            output_path = self._result_output_path(result)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.to_json(), encoding="utf-8")

    def _result_output_path(self, result: Results) -> Path:
        source_path = Path(result.path)
        relative_path = self._data_relative_path(source_path)
        return self.results_dir / relative_path.with_suffix(".json")

    def _prediction_name(self, name: str, image_path: str | Path) -> str:
        base_name = name.strip("/")
        relative_parent = self._data_relative_parent(image_path)
        if relative_parent == Path("."):
            return base_name
        return str(Path(base_name) / relative_parent)

    def _data_relative_parent(self, image_path: str | Path) -> Path:
        parent = Path(image_path).parent.resolve()
        data_dir = self.data_dir.resolve()
        try:
            return parent.relative_to(data_dir)
        except ValueError:
            return Path(".")

    def _data_relative_path(self, image_path: str | Path) -> Path:
        path = Path(image_path).resolve()
        data_dir = self.data_dir.resolve()
        try:
            return path.relative_to(data_dir)
        except ValueError:
            return Path(path.name)


def predict_source(
    model: YOLO,
    source: str | Path | list[str | Path],
    *,
    output_dir: str | Path,
    results_dir: str | Path = RESULTS_DIR,
    data_dir: str | Path = DATA_DIR,
    name: str,
    conf: float,
    imgsz: int,
    device: str | None,
    save: bool,
    batch: int,
) -> list[Results]:
    """
    Run YOLO inference and save visualization images.

    :param model: Loaded Ultralytics YOLO model.
    :param source: Local image path or image path list.
    :param output_dir: Directory where YOLO visualization images are saved.
    :param results_dir: Directory where YOLO JSON results are saved.
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
        results_dir=results_dir,
        data_dir=data_dir,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=save,
    )
    if isinstance(source, list):
        return inferencer.predict_batch(source, name=name, batch=batch)

    return inferencer.predict(source, name=name)
