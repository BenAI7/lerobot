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

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations with FOURCC format
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # With 90° rotation and MJPG format
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # With YUYV format
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        backend: Optional OpenCV VideoCapture backend. On Windows, some webcams work better
                 with "DSHOW" than "MSMF". Supported values:
                 - int (e.g. cv2.CAP_DSHOW)
                 - str: "MSMF", "DSHOW", "ANY", "AUTO" (case-insensitive)
        split_mode: Optional packed stereo split mode for "stitched" stereo cameras that output a single
                    side-by-side or top-bottom image (e.g. 640x240 containing two 320x240 views).
                    Supported values:
                    - None (no splitting)
                    - "side_by_side" (aliases: "sbs", "lr")
                    - "top_bottom" (aliases: "tb", "ud")
        split_include_packed: If True, keep the original packed image under the original camera key
                              in addition to emitting `{key}_left` / `{key}_right`.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        fourcc: FOURCC code for video format (e.g., "MJPG", "YUYV", "I420"). Defaults to None (auto-detect).

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - FOURCC codes must be 4-character strings (e.g., "MJPG", "YUYV"). Some common FOUCC codes: https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - Setting FOURCC can help achieve higher frame rates on some cameras.
    """

    index_or_path: int | Path
    backend: str | int | None = None
    split_mode: str | None = None
    split_include_packed: bool = False
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )

        if self.split_mode is not None:
            m = str(self.split_mode).strip().lower()
            if m not in ("side_by_side", "sbs", "lr", "top_bottom", "tb", "ud"):
                raise ValueError(
                    f"`split_mode` must be one of: side_by_side/sbs/lr/top_bottom/tb/ud, but '{self.split_mode}' is provided."
                )
