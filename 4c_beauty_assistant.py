import argparse
import asyncio
import json
from pathlib import Path

from PIL import Image


DEFAULT_IMAGE_PATH = Path(__file__).with_name("sample.jpg")


def analyze_image_file(image_path):
    from modules.pipeline import run_hair_pipeline

    image = Image.open(image_path).convert("RGB")
    return run_hair_pipeline(image)


async def analyze_viam_robot(args):
    from viam.robot.client import RobotClient

    from modules.viam_adapter import analyze_camera

    opts = RobotClient.Options.with_api_key(
        api_key=args.api_key,
        api_key_id=args.api_key_id,
    )
    robot = await RobotClient.at_address(args.robot_address, opts)
    try:
        return await analyze_camera(robot, args.camera_name, mime_type=args.mime_type)
    finally:
        await robot.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Run the 4C Hair CV pipeline.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Local image path to analyze.",
    )
    parser.add_argument("--robot-address", help="Viam machine address.")
    parser.add_argument("--api-key", help="Viam API key.")
    parser.add_argument("--api-key-id", help="Viam API key ID.")
    parser.add_argument("--camera-name", help="Viam camera component name.")
    parser.add_argument(
        "--mime-type",
        default="image/jpeg",
        help="Requested MIME type for Viam camera frames.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the human-readable care report instead of JSON.",
    )
    return parser


def should_use_viam(args):
    viam_fields = [
        args.robot_address,
        args.api_key,
        args.api_key_id,
        args.camera_name,
    ]
    if any(viam_fields) and not all(viam_fields):
        raise ValueError(
            "Viam mode requires --robot-address, --api-key, --api-key-id, and --camera-name."
        )
    return all(viam_fields)


def main():
    args = build_parser().parse_args()
    if should_use_viam(args):
        result = asyncio.run(analyze_viam_robot(args))
    else:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        result = analyze_image_file(args.image)

    if args.report:
        print(result["report"])
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
