from pathlib import Path

from modules.segmentation import segment_hair


def main():
    image_path = Path(__file__).with_name("sample.jpg")
    mask = segment_hair(str(image_path))
    print(
        {
            "image": str(image_path),
            "mask_shape": mask.shape,
            "hair_pixels": int(mask.sum()),
        }
    )


if __name__ == "__main__":
    main()
