import struct
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image


@dataclass
class MilimgHeader:
    """storage data in .milimg file (this part is unchanged)"""

    version: int
    width: int
    height: int
    color_payload: bytes
    alpha_payload: Optional[bytes] = None


def parse_milimg_container(file_path: str) -> MilimgHeader:
    """
    parse .milimg container file, extract metadata and compressed data blocks.(this function is unchanged)
    """
    with open(file_path, "rb") as f:
        if f.read(8) != b"Milimg00":
            raise ValueError("file format error: invalid magic number")

        version = struct.unpack(">I", f.read(4))[0]
        if version not in [0, 1]:
            raise ValueError(f"unsupported version: {version}")

        width, height, color_payload_size = struct.unpack(">IIQ", f.read(16))
        color_payload = f.read(color_payload_size)

        alpha_payload = None
        if version == 1:
            alpha_payload_size = struct.unpack(">Q", f.read(8))[0]
            alpha_payload = f.read(alpha_payload_size)

    return MilimgHeader(
        version=version,
        width=width,
        height=height,
        color_payload=color_payload,
        alpha_payload=alpha_payload,
    )


def decode_av1_frame_with_opencv(payload: bytes) -> Optional[np.ndarray]:
    """
    decode single AV1 frame data with OpenCV (cv2.VideoCapture).
    write temporary file, because VideoCapture can't read memory stream directly.
    """
    if not payload:
        return None

    # create a temporary file to store AV1 data
    # delete=False allows us to access the file name after closing the file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ivf") as tmp:
        tmp.write(payload)
        temp_filename = tmp.name

    frame = None
    cap = None
    try:
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            raise IOError(f"OpenCV can't open temporary file: {temp_filename}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError("OpenCV can't read frame")

    except Exception as e:
        print(f"OpenCV decode failed: {e}")
        return None
    finally:
        if cap:
            cap.release()
        os.remove(temp_filename)
    return frame


def decode_milimg(file_path: str) -> Optional[Image.Image]:
    """
    decode .milimg file with OpenCV
    """
    header = parse_milimg_container(file_path)

    print("decoding color data with OpenCV...")
    color_frame_bgr = decode_av1_frame_with_opencv(header.color_payload)
    if color_frame_bgr is None:
        print("decoding color data failed.")
        return None

    color_frame_rgb = cv2.cvtColor(color_frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_image = Image.fromarray(color_frame_rgb)

    if header.version == 1 and header.alpha_payload:
        print("decoding alpha data with OpenCV...")
        alpha_frame_bgr = decode_av1_frame_with_opencv(header.alpha_payload)

        if alpha_frame_bgr is not None:
            alpha_channel_np = alpha_frame_bgr[:, :, 0]
            alpha_image = Image.fromarray(alpha_channel_np, mode="L")

            rgba_image = rgb_image.convert("RGBA")
            rgba_image.putalpha(alpha_image)
            return rgba_image
        else:
            print("Alpha data decode failed, will return only color image.")

    return rgb_image.convert("RGBA")


if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(
        description="decode .milimg file to standard image (like PNG)."
    )
    parser.add_argument("input", help="input .milimg file path (e.g: my_image.milimg)")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="output image file path (e.g: my_image.png). if not specified, will generate same name .png file in current directory",
    )

    args = parser.parse_args()

    if args.output is None:
        input_basename = os.path.basename(args.input)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        args.output = f"{input_name_without_ext}.png"
        print(f"output path not specified, will output to: {args.output}")

    try:
        print(f"start processing file: {args.input}")
        final_image = decode_milimg(args.input)

        if final_image:
            final_image.save(args.output)
            print(f"decode success! image saved as '{args.output}'")

    except FileNotFoundError:
        print(f"error: file '{args.input}' not found.")
    except Exception as e:
        print(f"error: unknown error: {e}")
        print("\n" + "=" * 20 + " traceback " + "=" * 20)
        traceback.print_exc()
        print("=" * 58 + "\n")
