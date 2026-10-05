"""Minimal PNG encoding and display transforms (NumPy and the standard library only)."""
from __future__ import annotations

import struct
import zlib

import numpy as np

# Black -> purple -> red -> orange -> pale yellow, for absolute differences.
_STOPS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float64)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(image: np.ndarray) -> bytes:
    """(H, W) grey or (H, W, 3) RGB uint8 -> PNG bytes."""
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        raise ValueError("PNG input is uint8")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        colour = 0
    elif image.ndim == 3 and image.shape[2] == 3:
        colour = 2
    else:
        raise ValueError("PNG input is (H, W) or (H, W, 3)")
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """A stimulus frame in [0, 1] (float) or uint8 -> uint8 for display."""
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    return np.round(np.clip(np.nan_to_num(frame.astype(np.float64)), 0.0, 1.0) * 255).astype(np.uint8)


def heatmap(values: np.ndarray, vmax: float) -> np.ndarray:
    """(H, W) nonnegative values -> (H, W, 3) uint8 colour map, saturating at `vmax` (0 maps to black)."""
    values = np.asarray(values, dtype=np.float64)
    scaled = np.zeros_like(values) if vmax <= 0 else np.clip(values / vmax, 0.0, 1.0)
    position = scaled * (len(_STOPS) - 1)
    low = np.minimum(np.floor(position).astype(int), len(_STOPS) - 2)
    fraction = (position - low)[..., None]
    colour = _STOPS[low] * (1 - fraction) + _STOPS[low + 1] * fraction
    return np.round(colour).astype(np.uint8)


def enlarge(image: np.ndarray, minimum: int = 128) -> np.ndarray:
    """Nearest-neighbour enlargement so that the shorter side is at least `minimum` pixels."""
    factor = max(1, int(np.ceil(minimum / max(1, min(image.shape[:2])))))
    return image if factor == 1 else np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
