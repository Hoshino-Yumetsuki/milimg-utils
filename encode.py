import struct
import argparse
import io
import av
from PIL import Image


def has_transparency(img: Image.Image) -> bool:
    """Check if Pillow image object contains valid transparency"""
    if img.mode == "P":
        if "transparency" in img.info:
            return True
    elif img.mode == "RGBA":
        # Check if all alpha channel pixels are 255 (fully opaque)
        alpha = img.getchannel("A")
        return any(p < 255 for p in alpha.getdata())
    return False


def encode_to_av1(image: Image.Image, quality: int, is_alpha: bool = False) -> bytes:
    """
    Encode Pillow image to raw AV1 bitstream using PyAV.

    Args:
        image: Pillow image object.
        quality: AV1 encoding CRF quality value (0-63, lower is higher quality).
        is_alpha: If True, encode as grayscale; otherwise encode as color.

    Returns:
        Encoded raw AV1 bitstream (bytes).
    """
    output_buffer = io.BytesIO()

    # Use IVF as temporary container to facilitate raw stream extraction
    with av.open(output_buffer, mode="w", format="ivf") as container:
        # Use efficient libaom-av1 encoder
        stream = container.add_stream("libaom-av1", rate=30)
        stream.width = image.width
        stream.height = image.height

        # --- Apply all precise parameters from reverse engineering ---
        if is_alpha:
            stream.pix_fmt = "gray8"  # Alpha channel uses 8-bit grayscale
        else:
            stream.pix_fmt = "yuv420p"  # Color channel uses YUV420
            stream.options["colorspace"] = "bt709"
            stream.options["color_range"] = "pc"

        # --- Apply user-defined quality parameter ---
        stream.options["crf"] = str(quality)

        # Convert Pillow image to AV Frame
        frame = av.VideoFrame.from_image(image)

        # Encode
        for packet in stream.encode(frame):
            container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

    # Strip pure AV1 data from IVF container
    # IVF header (32 bytes) + frame header (12 bytes) = 44 bytes
    ivf_data = output_buffer.getvalue()
    return ivf_data[44:]


def encode_milimg(input_path: str, output_path: str, quality: int):
    """
    Main function: load image, encode, and assemble into .milimg file.
    """
    print(f"loading input image: {input_path}")
    # Convert to RGBA for easier processing
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    # Check for valid alpha channel to determine version number
    use_alpha = has_transparency(img)
    version = 1 if use_alpha else 0
    print(
        f"image size: {width}x{height}. valid alpha channel detected: {use_alpha}. will generate version {version} file."
    )

    # Encode color channel
    print(f"encoding color channel (YUV420) with quality (CRF)={quality}...")
    rgb_image = img.convert("RGB")
    color_payload = encode_to_av1(rgb_image, quality, is_alpha=False)
    print(f"color data encoding complete, size: {len(color_payload)} bytes.")

    alpha_payload = None
    if version == 1:
        print(f"encoding alpha channel (Grayscale) with quality (CRF)={quality}...")
        alpha_image = img.getchannel("A")
        alpha_payload = encode_to_av1(alpha_image, quality, is_alpha=True)
        print(f"alpha data encoding complete, size: {len(alpha_payload)} bytes.")

    # Assemble .milimg file
    print("assembling .milimg file...")
    with open(output_path, "wb") as f:
        # Write magic number
        f.write(b"Milimg00")
        # Write version number (4 bytes, big-endian)
        f.write(struct.pack(">I", version))
        # Write metadata (width, height, color data size)
        f.write(struct.pack(">I", width))
        f.write(struct.pack(">I", height))
        f.write(struct.pack(">Q", len(color_payload)))
        # Write color data block
        f.write(color_payload)

        if version == 1:
            # Write alpha metadata and data block
            f.write(struct.pack(">Q", len(alpha_payload)))
            f.write(alpha_payload)

    print(f"\nsuccess! '{output_path}' created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="encode standard image (like PNG) to .milimg format."
    )
    parser.add_argument("input", help="input image file path (e.g: my_image.png)")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="output .milimg file path (e.g: my_image.milimg). if not specified, will generate same name .milimg file in current directory",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=0,
        help="AV1 encoding quality (CRF value, 0-63). lower value means higher quality and larger file. recommended range: 18-40. default: 28.",
    )

    args = parser.parse_args()

    # If output path not specified, generate same name .milimg file in current directory
    if args.output is None:
        import os

        input_basename = os.path.basename(args.input)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        args.output = f"{input_name_without_ext}.milimg"
        print(f"output path not specified, will output to: {args.output}")

    if not (0 <= args.quality <= 63):
        print("error: quality value must be between 0 and 63.")
    else:
        try:
            encode_milimg(args.input, args.output, args.quality)
        except FileNotFoundError:
            print(f"error: input file '{args.input}' not found.")
        except Exception as e:
            import traceback

            print(f"unknown error occurred: {e}")
            traceback.print_exc()
