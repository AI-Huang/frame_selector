import time

import cv2
import numpy as np

MIN_FRAMES_DEFAULT = 8  # 条件允许时至少抽取这么多帧
FRAMES_PER_SECOND_DEFAULT = 0.5  # 每 2 秒视频抽 1 帧


def frame_quality_metrics(
    image: np.ndarray, gray: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """计算单帧的清晰度、亮度、对比度、饱和度、熵和彩色度。"""
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    entropy = float(-(prob * np.log2(prob)).sum())

    blue, green, red = cv2.split(image.astype(np.float32))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    colorful = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    return sharpness, brightness, contrast, saturation, entropy, colorful


def color_thumb(image: np.ndarray, size: int = 8) -> np.ndarray:
    """生成用于相似度计算的小尺寸颜色缩略图。"""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.min())) / span


def combine_quality(
    blur: float,
    brightness: float,
    contrast: float,
    saturation: float,
    entropy: float,
    colorful: float,
) -> float:
    """把多项画质指标合成为 0 到 1 之间的质量分。"""
    brightness_score = 1.0 - min(abs(float(brightness) - 127.5) / 127.5, 1.0)
    sharpness_score = min(np.log1p(max(float(blur), 0.0)) / np.log1p(1000.0), 1.0)
    contrast_score = min(max(float(contrast), 0.0) / 64.0, 1.0)
    saturation_score = min(max(float(saturation), 0.0) / 128.0, 1.0)
    entropy_score = min(max(float(entropy), 0.0) / 8.0, 1.0)
    colorful_score = min(max(float(colorful), 0.0) / 128.0, 1.0)
    return float(
        0.30 * sharpness_score
        + 0.25 * brightness_score
        + 0.15 * contrast_score
        + 0.10 * saturation_score
        + 0.10 * entropy_score
        + 0.10 * colorful_score
    )


def quality_breakdown(
    blurs: list[float],
    brights: list[float],
    contrasts: list[float],
    sats: list[float],
    entropies: list[float],
    colorfuls: list[float],
) -> dict[str, float]:
    """汇总候选帧的画质指标，便于调用方记录拒绝原因。"""
    metrics = {
        "blur": blurs,
        "brightness": brights,
        "contrast": contrasts,
        "saturation": sats,
        "entropy": entropies,
        "colorful": colorfuls,
    }
    return {
        f"{name}_{stat}": round(float(value), 4)
        for name, values in metrics.items()
        for stat, value in (
            ("min", np.min(values) if values else 0.0),
            ("mean", np.mean(values) if values else 0.0),
            ("max", np.max(values) if values else 0.0),
        )
    }


def select_diverse(
    times_s: np.ndarray,
    blurs: np.ndarray,
    brights: np.ndarray,
    thumbs_flat: np.ndarray,
    *,
    target_count: int,
    duration_s: float,
    blur_drop_ratio: float,
    brightness_min: float,
    brightness_max: float,
    min_select: int,
) -> np.ndarray:
    """先按质量过滤，再用时间和颜色特征做最远点采样。"""
    if times_s.size == 0:
        return np.empty((0,), dtype=np.int64)

    blur_floor = float(np.quantile(blurs, min(max(blur_drop_ratio, 0.0), 1.0)))
    mask = (
        (blurs >= blur_floor)
        & (brights >= float(brightness_min))
        & (brights <= float(brightness_max))
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return np.empty((0,), dtype=np.int64)

    count = min(max(int(min_select), int(target_count)), int(candidates.size))
    features = np.column_stack(
        (
            _normalize(times_s[candidates]) * max(float(duration_s), 1.0),
            _normalize(blurs[candidates]),
            thumbs_flat[candidates].astype(np.float64) / 255.0,
        )
    )

    selected = [int(np.argmin(np.abs(times_s[candidates] - duration_s / 2.0)))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < count:
        next_local = int(np.argmax(min_dist))
        if next_local in selected:
            break
        selected.append(next_local)
        next_dist = np.linalg.norm(features - features[next_local], axis=1)
        min_dist = np.minimum(min_dist, next_dist)

    selected_candidates = np.sort(candidates[np.asarray(selected, dtype=np.int64)])
    return selected_candidates.astype(np.int64)


def select_from_frame_dir(
    frame_paths: list[str],
    *,
    times_s: list[float] | None = None,
    source_fps: float = 1.0,
    fps: float = 0.0,
    total_frames: int = 0,
    max_frames: int = 64,
    min_frames: int = MIN_FRAMES_DEFAULT,
    frames_per_second: float = FRAMES_PER_SECOND_DEFAULT,
    blur_drop_ratio: float = 0.25,
    brightness_min: float = 5.0,
    brightness_max: float = 250.0,
    quality_reject_threshold: float = 0.0,
) -> tuple:
    """从已预抽帧目录中智能选帧（不重新解码视频）。

    行为上与 :func:`extract_meta_smart` 保持一致，但候选集是现成的图片文件列表，
    而不是从视频中解码出的帧。由于上游通常已经稀疏抽帧（例如约 1 fps），这里会
    对**每一帧**评分，不再做第一轮子采样，也不再进行第二轮重新解码。

    参数
    ----------
    frame_paths : list[str]
        图片路径列表，已按采集顺序预先排序（见 :func:`list_frame_files`）。
    times_s : list[float] | None
        每帧对应的真实视频时间戳（秒），与 ``frame_paths`` 一一对齐。当为 ``None``
        时，假设上游采样率均匀，并按 ``i / source_fps`` 合成时间戳。
    source_fps : float
        上游抽帧采样率，仅在 ``times_s`` 为 ``None`` 时使用。
    fps, total_frames : float, int
        原始视频元数据，会原样回传到返回元组中，便于缓存条目保留真实 fps / 帧数。
        ``fps<=0`` 时回退到 ``source_fps``；``total_frames<=0`` 时回退到
        ``len(frame_paths)``。

    返回
    -------
    ``(selected_paths, fps, total, timing, frame_quality, reject_info)`` — 形状与
    :func:`extract_meta_smart` 相同，但第一个元素是按时间顺序排列的**已选图片路径**
    列表，而不是帧序号。
    """
    out_fps = float(fps) if fps and fps > 0 else float(source_fps or 1.0)
    out_total = (
        int(total_frames) if total_frames and total_frames > 0 else len(frame_paths)
    )

    if not frame_paths:
        return (
            [],
            out_fps,
            out_total,
            {
                "scan_ms": 0.0,
                "select_ms": 0.0,
                "sampled": 0,
            },
            np.empty((0,), dtype=np.float32),
            {
                "scan_ms": 0.0,
                "select_ms": 0.0,
                "read_ms": 0.0,
                "extract_ms": 0.0,
                "sampled": 0,
                "kept": 0,
                "mean_quality": 0.0,
                "quality_pass": False,
                "quality_threshold": quality_reject_threshold,
                "reject_reason": "no_frame_files",
                "quality_breakdown": quality_breakdown([], [], [], [], [], []),
            },
        )

    if times_s is not None and len(times_s) == len(frame_paths):
        times_arr = np.asarray(times_s, dtype=np.float64)
    else:
        sfps = float(source_fps) if source_fps and source_fps > 0 else 1.0
        times_arr = np.arange(len(frame_paths), dtype=np.float64) / sfps

    duration_s = float(times_arr[-1]) if times_arr.size else 0.0
    target_count = int(round(duration_s * float(frames_per_second)))
    target_count = max(int(min_frames), min(int(max_frames), target_count))

    # ── 第 1 轮：每帧只读取一次，在 256×256 缓冲图上计算指标。
    t_scan = time.perf_counter()
    valid_pos: list[int] = []
    blurs: list[float] = []
    brights: list[float] = []
    contrasts: list[float] = []
    sats: list[float] = []
    entropies: list[float] = []
    colorfuls: list[float] = []
    thumbs: list[np.ndarray] = []

    for pos, path in enumerate(frame_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, br, co, sa, en, cf = frame_quality_metrics(small, gray)
        valid_pos.append(pos)
        blurs.append(sh)
        brights.append(br)
        contrasts.append(co)
        sats.append(sa)
        entropies.append(en)
        colorfuls.append(cf)
        thumbs.append(color_thumb(small))
    scan_ms = round((time.perf_counter() - t_scan) * 1000, 1)

    if not valid_pos:
        return (
            [],
            out_fps,
            out_total,
            {
                "scan_ms": scan_ms,
                "select_ms": 0.0,
                "sampled": 0,
            },
            np.empty((0,), dtype=np.float32),
            {
                "scan_ms": scan_ms,
                "select_ms": 0.0,
                "read_ms": 0.0,
                "extract_ms": scan_ms,
                "sampled": 0,
                "kept": 0,
                "mean_quality": 0.0,
                "quality_pass": False,
                "quality_threshold": quality_reject_threshold,
                "reject_reason": "no_decodable_frames",
                "quality_breakdown": quality_breakdown([], [], [], [], [], []),
            },
        )

    # ── 选择阶段：复用质量门控和最远点采样核心逻辑。
    t_sel = time.perf_counter()
    valid_pos_a = np.asarray(valid_pos, dtype=np.int64)
    blurs_a = np.asarray(blurs, dtype=np.float64)
    brights_a = np.asarray(brights, dtype=np.float64)
    thumbs_flat = np.asarray(thumbs, dtype=np.int16).reshape(len(thumbs), -1)
    cand_times = times_arr[valid_pos_a]

    selected_local = select_diverse(
        cand_times,
        blurs_a,
        brights_a,
        thumbs_flat,
        target_count=target_count,
        duration_s=duration_s if duration_s > 0 else float(len(valid_pos)),
        blur_drop_ratio=blur_drop_ratio,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_select=min_frames,
    )
    select_ms = round((time.perf_counter() - t_sel) * 1000, 1)

    if selected_local.size == 0:
        sample_q = np.array(
            [
                combine_quality(
                    blurs[i],
                    brights[i],
                    contrasts[i],
                    sats[i],
                    entropies[i],
                    colorfuls[i],
                )
                for i in range(len(blurs))
            ],
            dtype=np.float32,
        )
        return (
            [],
            out_fps,
            out_total,
            {
                "scan_ms": scan_ms,
                "select_ms": select_ms,
                "sampled": len(valid_pos),
            },
            np.empty((0,), dtype=np.float32),
            {
                "scan_ms": scan_ms,
                "select_ms": select_ms,
                "read_ms": 0.0,
                "extract_ms": round(scan_ms + select_ms, 1),
                "sampled": len(valid_pos),
                "kept": 0,
                "mean_quality": (
                    round(float(sample_q.mean()), 4) if sample_q.size else 0.0
                ),
                "quality_pass": False,
                "quality_threshold": quality_reject_threshold,
                "reject_reason": "all_frames_outside_brightness_range",
                "quality_breakdown": quality_breakdown(
                    blurs, brights, contrasts, sats, entropies, colorfuls
                ),
            },
        )

    # ``selected_local`` 是候选数组的索引；这里映射回原始文件路径。
    selected_paths = [frame_paths[int(valid_pos_a[i])] for i in selected_local]
    frame_quality = np.array(
        [
            combine_quality(
                blurs[int(i)],
                brights[int(i)],
                contrasts[int(i)],
                sats[int(i)],
                entropies[int(i)],
                colorfuls[int(i)],
            )
            for i in selected_local
        ],
        dtype=np.float32,
    )

    return (
        selected_paths,
        out_fps,
        out_total,
        {
            "scan_ms": scan_ms,
            "select_ms": select_ms,
            "sampled": len(valid_pos),
            "target_count": int(target_count),
            "duration_s": round(float(duration_s), 2),
        },
        frame_quality,
        None,
    )
