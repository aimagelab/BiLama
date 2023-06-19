import pdf2image
import argparse
from pathlib import Path


def pdf_to_images(pdf_path, output_path):
    output_path = Path(output_path) / Path(pdf_path).stem
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Converting {pdf_path} to images in {output_path}')

    images = pdf2image.convert_from_path(pdf_path)
    print(f'Converted {len(images)} images. Saving...')
    for i, image in enumerate(images):
        image.save(output_path / f'{i:04d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    pdf_to_images(args.pdf_path, args.output_path)


if __name__ == '__main__':
    main()


