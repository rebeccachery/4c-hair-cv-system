from pathlib import Path

from PIL import Image

from modules.features import extract_features
from modules.segmentation import segment_hair


def main():
    image_path = Path(__file__).with_name("sample.jpg")
    image = Image.open(image_path).convert("RGB")
    hair_mask = segment_hair(image)
    print(extract_features(image, hair_mask))


if __name__ == "__main__":
    main()
