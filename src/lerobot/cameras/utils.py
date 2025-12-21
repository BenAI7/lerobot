#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera {key} with config {cfg}: {e}") from e

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None


def get_cv2_backend() -> int:
    import cv2

    if platform.system() == "Windows":
        return int(cv2.CAP_MSMF)  # Use MSMF for Windows instead of AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux and others
        return int(cv2.CAP_ANY)


def packed_stereo_split_shapes(height: int, width: int, mode: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Given a packed stereo frame shape (H, W, 3), return shapes for (left, right) images.

    Supported modes:
    - "side_by_side" (aliases: "sbs", "lr")
    - "top_bottom" (aliases: "tb", "ud")
    """
    m = mode.strip().lower()
    if m in ("side_by_side", "sbs", "lr"):
        if width % 2 != 0:
            raise ValueError(f"packed stereo width must be even for side_by_side split, got {width=}")
        return (height, width // 2, 3), (height, width // 2, 3)
    if m in ("top_bottom", "tb", "ud"):
        if height % 2 != 0:
            raise ValueError(f"packed stereo height must be even for top_bottom split, got {height=}")
        return (height // 2, width, 3), (height // 2, width, 3)
    raise ValueError(f"Unsupported packed stereo split mode: {mode!r}")


def split_packed_stereo_frame(frame, mode: str):
    """
    Split a packed stereo RGB frame into (left, right) frames.

    `frame` is expected to be a NumPy array shaped (H, W, 3).
    """
    # Lazy import: numpy is a heavy dependency but already required by cameras.
    import numpy as np  # type: ignore

    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError(f"Expected packed stereo frame as np.ndarray with shape (H, W, C); got {type(frame)=}, {getattr(frame, 'shape', None)=}")
    h, w, c = frame.shape
    if c != 3:
        raise ValueError(f"Expected 3-channel packed stereo frame; got {c=}")

    m = mode.strip().lower()
    if m in ("side_by_side", "sbs", "lr"):
        if w % 2 != 0:
            raise ValueError(f"packed stereo width must be even for side_by_side split, got {w=}")
        mid = w // 2
        return frame[:, :mid, :], frame[:, mid:, :]
    if m in ("top_bottom", "tb", "ud"):
        if h % 2 != 0:
            raise ValueError(f"packed stereo height must be even for top_bottom split, got {h=}")
        mid = h // 2
        return frame[:mid, :, :], frame[mid:, :, :]

    raise ValueError(f"Unsupported packed stereo split mode: {mode!r}")