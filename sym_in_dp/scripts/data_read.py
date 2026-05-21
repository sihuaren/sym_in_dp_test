#!/usr/bin/env python3
"""
Inspect a parquet episode and export RGB camera views as a tiled video.

The script is intentionally schema-tolerant. It can decode common image cell
formats used in robotics datasets:
  - encoded image bytes, such as PNG or JPEG
  - numpy/list arrays shaped HWC or CHW
  - dict / struct values with "bytes" or "path" fields
  - image paths relative to the parquet file directory
"""

import argparse
import math
import pathlib
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_PARQUET = (
    "/hard_data/user_dataset/rensihua_dataset/realbot_260518/"
    "cake_box_260514/episode_00000.parquet"
)


BAD_IMAGE_NAME_TOKENS = (
    "depth",
    "mask",
    "seg",
    "label",
    "point",
    "cloud",
    "state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)

GOOD_IMAGE_NAME_TOKENS = (
    "rgb",
    "image",
    "camera",
    "cam",
    "color",
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def first_non_missing(values: Iterable[Any]) -> Optional[Any]:
    for value in values:
        if not is_missing(value):
            return value
    return None


def find_nested_value(value: Any, keys: tuple[str, ...]) -> Optional[Any]:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if key in value and not is_missing(value[key]):
            return value[key]
    for nested in value.values():
        if isinstance(nested, dict):
            found = find_nested_value(nested, keys)
            if found is not None:
                return found
    return None


def to_uint8_rgb(array: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(array)
    if arr.ndim == 0:
        return None

    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        return None

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if finite.any() and arr[finite].max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def decode_encoded_image(data: Any) -> Optional[np.ndarray]:
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        return None
    encoded = np.frombuffer(data, dtype=np.uint8)
    if encoded.size == 0:
        return None
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image_value(value: Any, base_dir: pathlib.Path) -> Optional[np.ndarray]:
    if is_missing(value):
        return None

    if isinstance(value, dict):
        image_bytes = find_nested_value(value, ("bytes", "data", "image"))
        image = decode_image_value(image_bytes, base_dir) if image_bytes is not None else None
        if image is not None:
            return image

        image_path = find_nested_value(value, ("path", "file", "filename"))
        image = decode_image_value(image_path, base_dir) if image_path is not None else None
        if image is not None:
            return image
        return None

    image = decode_encoded_image(value)
    if image is not None:
        return image

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_image_value(value.item(), base_dir)
        if value.ndim == 1 and value.dtype == np.uint8 and value.size > 16:
            image = decode_encoded_image(value.tobytes())
            if image is not None:
                return image
        return to_uint8_rgb(value)

    if isinstance(value, (list, tuple)):
        try:
            return to_uint8_rgb(np.asarray(value))
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        candidate = pathlib.Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.is_file():
            bgr = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return None


def value_description(value: Any, base_dir: pathlib.Path) -> str:
    if value is None:
        return "None"
    image = decode_image_value(value, base_dir)
    if image is not None:
        return f"{type(value).__name__}, decoded_rgb_shape={tuple(image.shape)}, decoded_dtype={image.dtype}"
    if isinstance(value, dict):
        return f"dict, keys={list(value.keys())}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"{type(value).__name__}, bytes={len(value)}"
    if isinstance(value, np.ndarray):
        return f"ndarray, shape={tuple(value.shape)}, dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
            return f"{type(value).__name__}, len={len(value)}, array_shape={tuple(arr.shape)}, array_dtype={arr.dtype}"
        except (TypeError, ValueError):
            return f"{type(value).__name__}, len={len(value)}"
    if isinstance(value, str):
        preview = value if len(value) <= 80 else value[:77] + "..."
        return f"str, len={len(value)}, sample={preview!r}"
    return f"{type(value).__name__}, sample={value!r}"


def count_missing(series: pd.Series) -> int:
    return sum(1 for value in series if is_missing(value))


def print_report(path: pathlib.Path, dataframe: pd.DataFrame) -> None:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow
    arrow_types = {field.name: str(field.type) for field in schema}

    print("\n=== Parquet file ===")
    print(f"path: {path}")
    print(f"rows: {len(dataframe)}")
    print(f"columns: {len(dataframe.columns)}")
    print(f"row_groups: {metadata.num_row_groups}")
    print(f"created_by: {metadata.created_by}")

    print("\n=== Arrow schema ===")
    print(schema)

    print("\n=== Column summary ===")
    base_dir = path.parent
    for column in dataframe.columns:
        series = dataframe[column]
        sample = first_non_missing(series)
        sample_desc = value_description(sample, base_dir) if sample is not None else "all values missing"
        print(
            f"- {column}: "
            f"pandas_dtype={series.dtype}, "
            f"arrow_type={arrow_types.get(column, '<nested or unavailable>')}, "
            f"column_shape=({len(series)},), "
            f"missing={count_missing(series)}, "
            f"sample={sample_desc}"
        )


def column_name_image_score(column: str) -> int:
    name = column.lower()
    score = 0
    if any(token in name for token in GOOD_IMAGE_NAME_TOKENS):
        score += 2
    if "rgb" in name:
        score += 2
    if any(token in name for token in BAD_IMAGE_NAME_TOKENS):
        score -= 4
    return score


def detect_rgb_columns(
    dataframe: pd.DataFrame,
    base_dir: pathlib.Path,
    requested_columns: Optional[list[str]],
    probe_rows: int,
) -> list[str]:
    if requested_columns:
        missing = [column for column in requested_columns if column not in dataframe.columns]
        if missing:
            raise KeyError(f"Requested image columns not found: {missing}")
        return requested_columns

    candidates: list[tuple[int, str]] = []
    rows_to_probe = min(len(dataframe), probe_rows)
    for column in dataframe.columns:
        name_score = column_name_image_score(column)
        success_count = 0
        for row_idx in range(rows_to_probe):
            image = decode_image_value(dataframe[column].iloc[row_idx], base_dir)
            if image is not None and image.ndim == 3 and image.shape[-1] == 3:
                success_count += 1
                break
        if success_count:
            candidates.append((name_score, column))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in candidates]


def short_label(column: str) -> str:
    label = column.replace("observation.", "").replace("images.", "")
    if len(label) > 28:
        label = "..." + label[-25:]
    return label


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def make_placeholder(size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(
        image,
        "missing",
        (max(8, width // 2 - 45), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return label_image(image, label)


def make_tiled_frame(
    images: list[np.ndarray],
    labels: list[str],
    tile_size: tuple[int, int],
    grid_cols: int,
    pad: int,
    draw_labels: bool,
) -> np.ndarray:
    width, height = tile_size
    grid_rows = math.ceil(len(images) / grid_cols)
    canvas_h = grid_rows * height + (grid_rows + 1) * pad
    canvas_w = grid_cols * width + (grid_cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = pad + row * (height + pad)
        x0 = pad + col * (width + pad)
        tile = resize_rgb(image, tile_size)
        if draw_labels:
            tile = label_image(tile, label)
        canvas[y0 : y0 + height, x0 : x0 + width] = tile
    return canvas


def choose_tile_size(
    dataframe: pd.DataFrame,
    base_dir: pathlib.Path,
    image_columns: list[str],
    tile_height: int,
) -> tuple[int, int]:
    for column in image_columns:
        for value in dataframe[column]:
            image = decode_image_value(value, base_dir)
            if image is not None:
                height, width = image.shape[:2]
                if tile_height > 0:
                    width = max(1, round(width * tile_height / height))
                    height = tile_height
                return width, height
    raise RuntimeError("Could not decode any image from selected columns.")


def write_tiled_video(
    dataframe: pd.DataFrame,
    path: pathlib.Path,
    output: pathlib.Path,
    image_columns: list[str],
    fps: float,
    max_frames: Optional[int],
    tile_height: int,
    grid_cols: Optional[int],
    draw_labels: bool,
    fourcc_name: str,
) -> None:
    if not image_columns:
        raise RuntimeError("No RGB image columns were detected. Pass --image-columns manually.")

    base_dir = path.parent
    labels = [short_label(column) for column in image_columns]
    tile_size = choose_tile_size(dataframe, base_dir, image_columns, tile_height)
    if grid_cols is None:
        grid_cols = len(image_columns)
    grid_cols = max(1, min(grid_cols, len(image_columns)))

    n_frames = len(dataframe) if max_frames is None else min(len(dataframe), max_frames)
    first_tiles: list[np.ndarray] = []
    for column, label in zip(image_columns, labels):
        image = decode_image_value(dataframe[column].iloc[0], base_dir)
        if image is None:
            image = make_placeholder(tile_size, label)
        first_tiles.append(image)
    first_frame = make_tiled_frame(first_tiles, labels, tile_size, grid_cols, pad=4, draw_labels=draw_labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(output), fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}")

    try:
        for frame_idx in range(n_frames):
            tiles: list[np.ndarray] = []
            for column, label in zip(image_columns, labels):
                image = decode_image_value(dataframe[column].iloc[frame_idx], base_dir)
                if image is None:
                    image = make_placeholder(tile_size, label)
                tiles.append(image)
            frame = make_tiled_frame(tiles, labels, tile_size, grid_cols, pad=4, draw_labels=draw_labels)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print("\n=== Video ===")
    print(f"rgb_columns: {image_columns}")
    print(f"frames_written: {n_frames}")
    print(f"fps: {fps}")
    print(f"tile_size: {tile_size}")
    print(f"output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a parquet episode and save tiled RGB camera views to video."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_PARQUET,
        help="Input parquet path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path. Defaults to <input_stem>_rgb_views.mp4 beside the parquet file.",
    )
    parser.add_argument(
        "--image-columns",
        nargs="+",
        default=None,
        help="Image columns to use. If omitted, RGB columns are auto-detected.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--tile-height", type=int, default=240, help="Height of each camera tile.")
    parser.add_argument("--grid-cols", type=int, default=None, help="Number of camera tiles per row.")
    parser.add_argument("--probe-rows", type=int, default=30, help="Rows to probe for image auto-detection.")
    parser.add_argument("--fourcc", default="mp4v", help="OpenCV fourcc, for example mp4v or avc1.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw camera names on video tiles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")

    output_path = (
        pathlib.Path(args.output).expanduser()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_rgb_views.mp4")
    )

    dataframe = pd.read_parquet(input_path)
    print_report(input_path, dataframe)

    rgb_columns = detect_rgb_columns(
        dataframe=dataframe,
        base_dir=input_path.parent,
        requested_columns=args.image_columns,
        probe_rows=args.probe_rows,
    )
    write_tiled_video(
        dataframe=dataframe,
        path=input_path,
        output=output_path,
        image_columns=rgb_columns,
        fps=args.fps,
        max_frames=args.max_frames,
        tile_height=args.tile_height,
        grid_cols=args.grid_cols,
        draw_labels=not args.no_labels,
        fourcc_name=args.fourcc,
    )


if __name__ == "__main__":
    main()
