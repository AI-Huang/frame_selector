# frame_selector 使用文档

`frame_selector` 用于从一组已经预先抽取好的图片帧中，按画质和多样性选择一批代表帧。它不会重新解码视频，只读取传入的图片文件。

## 安装依赖

项目使用 `uv` 管理依赖：

```bash
uv sync
```

安装完成后可以验证模块是否可导入：

```bash
uv run python -c "from frame_selector import select_from_frame_dir; print('ok')"
```

## 基本用法

传入按时间顺序排列的图片路径列表，函数会返回被选中的图片路径，以及对应的元数据和质量分。

```python
from pathlib import Path

from frame_selector import select_from_frame_dir


frame_dir = Path("frames")
frame_paths = sorted(str(path) for path in frame_dir.glob("*.jpg"))

selected_paths, fps, total, timing, frame_quality, reject_info = select_from_frame_dir(
    frame_paths,
    source_fps=1.0,
    min_frames=8,
    max_frames=64,
    frames_per_second=0.5,
)

if reject_info is not None:
    print("没有选出可用帧：", reject_info["reject_reason"])
else:
    print("选中帧数量：", len(selected_paths))
    print("选中帧路径：", selected_paths)
    print("质量分：", frame_quality.tolist())
    print("耗时信息：", timing)
```

## 输入要求

- `frame_paths` 必须按采集时间从早到晚排序。
- 支持 OpenCV 可读取的图片格式，例如 `.jpg`、`.jpeg`、`.png`、`.webp`。
- 如果传入 `times_s`，长度必须与 `frame_paths` 一致，单位为秒。
- 如果不传 `times_s`，函数会按 `i / source_fps` 合成每帧时间戳。
- 不可解码的图片会被跳过，不会中断整个选择流程。

## 返回值

`select_from_frame_dir()` 返回 6 个值：

```python
(selected_paths, fps, total, timing, frame_quality, reject_info)
```

- `selected_paths`：按时间顺序排列的已选图片路径列表。
- `fps`：输出 fps。优先使用参数 `fps`；当 `fps <= 0` 时回退到 `source_fps`。
- `total`：输出总帧数。优先使用参数 `total_frames`；当 `total_frames <= 0` 时回退到 `len(frame_paths)`。
- `timing`：耗时和抽样信息，包含 `scan_ms`、`select_ms`、`sampled`、`target_count`、`duration_s` 等字段。
- `frame_quality`：每个已选帧的质量分，类型为 `numpy.ndarray`，分数范围约为 0 到 1。
- `reject_info`：拒绝信息。成功选出帧时为 `None`；失败时包含 `reject_reason`、`quality_breakdown` 等字段。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `source_fps` | `1.0` | 上游抽帧采样率，仅在未传 `times_s` 时用于合成时间戳。 |
| `fps` | `0.0` | 原始视频 fps，仅用于回传元数据。小于等于 0 时使用 `source_fps`。 |
| `total_frames` | `0` | 原始视频总帧数，仅用于回传元数据。小于等于 0 时使用图片数量。 |
| `min_frames` | `8` | 条件允许时至少选择的帧数。 |
| `max_frames` | `64` | 最多选择的帧数。 |
| `frames_per_second` | `0.5` | 目标选择密度，例如 `0.5` 表示约每 2 秒选 1 帧。 |
| `blur_drop_ratio` | `0.25` | 按清晰度分位数丢弃较模糊帧的比例。 |
| `brightness_min` | `5.0` | 允许的最低亮度，过暗帧会被过滤。 |
| `brightness_max` | `250.0` | 允许的最高亮度，过亮帧会被过滤。 |
| `quality_reject_threshold` | `0.0` | 当前用于拒绝信息中的阈值记录。 |

## 选择逻辑概览

1. 逐张读取图片，并缩放到 `256x256` 用于计算质量指标。
2. 计算清晰度、亮度、对比度、饱和度、熵和彩色度。
3. 根据清晰度分位数和亮度范围过滤候选帧。
4. 在候选帧中结合时间位置、清晰度和颜色缩略图做多样性选择。
5. 将候选数组索引映射回原始图片路径，并按时间顺序返回。

## 失败场景

当无法选出帧时，`selected_paths` 为空，`reject_info` 会说明原因：

- `no_frame_files`：传入的 `frame_paths` 为空。
- `no_decodable_frames`：所有图片都无法被 OpenCV 解码。
- `all_frames_outside_brightness_range`：所有候选帧都被亮度范围过滤掉。

可以根据 `reject_info["quality_breakdown"]` 查看候选帧质量指标的最小值、平均值和最大值。
