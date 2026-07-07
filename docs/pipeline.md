# 推理管线说明

本文档说明 `frame_selector` 当前的目录约定和 YOLO 推理流程。运行数据统一放在仓库根目录的 `data/` 入口下；该入口按代码与数据解耦原则指向外部运行数据目录，不应把帧、模型、推理结果直接写入源码目录。

## 相关目录

| 目录 | 作用 |
| --- | --- |
| `data/` | 运行数据统一入口，承载输入帧、模型权重和推理输出。 |
| `data/frames/` | 默认输入帧目录。`main.py` 会递归扫描其中支持的图片文件。 |
| `data/models/` | 默认模型权重目录。默认权重路径是 `data/models/yolo26x.pt`。 |
| `data/inference/` | 默认推理结果目录。YOLO visualization 图默认写入这里。 |
| `data/inference/predict/` | 默认 run name 为 `predict` 时的输出目录。 |
| `docs/` | 项目文档目录，记录使用方式、管线和目录约定。 |
| `scripts/` | 辅助脚本目录，例如抽帧脚本。 |
| `src/` | 核心源码目录，包含帧选择逻辑和 YOLO 推理封装。 |

## infer 具体步骤

默认命令：

```bash
uv run python main.py
```

等价于读取 `data/frames`，加载 `data/models/yolo26x.pt`，并把结果写入 `data/inference/predict`。

1. 解析命令行参数。

   `main.py` 读取输入源、模型路径、输出目录、run name、置信度、推理尺寸、设备、batch size 和是否保存 visualization 图。

2. 检查输入源是否存在。

   默认输入源是 `data/frames`。如果路径不存在，程序会提示先运行抽帧脚本：

   ```bash
   uv run python scripts/extract_all_frames.py
   ```

3. 判断输入源类型。

   如果输入源是目录，程序会递归扫描支持的图片后缀，例如 `.jpg`、`.png`、`.webp`、`.tiff`。如果输入源是单张图片，程序会直接进入单图推理。其他类型会报错退出。

4. 加载 YOLO 模型。

   默认模型是 `data/models/yolo26x.pt`。如果传入 `--model yolo26x.pt`，程序会解析到默认模型路径；如果传入其他路径，则按用户给定路径加载。

5. 创建 `YOLOInferencer`。

   `YOLOInferencer` 保存推理参数，包括输出目录 `output_dir`、置信度 `conf`、输入尺寸 `imgsz`、设备 `device`、是否保存结果 `save`，以及用于计算稳定输出路径的 `data_dir`。

6. 执行单图或批量推理。

   单张图片调用 `predict(image_path, name=...)`。目录输入会先得到图片路径列表，再调用 `predict_batch(image_paths, name=..., batch=...)`。

7. 生成相对 `data/` 的输出目录。

   推理输出的 run name 会根据图片父目录相对 `data/` 的路径生成。例如：

   ```text
   data/frames/1/frame_000001.jpg
   ```

   默认会写入：

   ```text
   data/inference/predict/frames/1/
   ```

   这样可以避免把本机绝对路径写入推理结果目录，同时保持不同输入子目录的结果可区分。

8. 保存 YOLO visualization 图。

   默认会保存结果图。如果使用 `--no-save`，程序仍会执行推理，但不保存 visualization 图。

## 常用命令

默认推理：

```bash
uv run python main.py
```

指定输入目录：

```bash
uv run python main.py data/frames
```

指定设备和推理尺寸：

```bash
uv run python main.py data/frames --device 0 --imgsz 960
```

只推理不保存 visualization 图：

```bash
uv run python main.py data/frames --no-save
```

指定推理输出目录：

```bash
uv run python main.py data/frames --output-dir data/inference
```
