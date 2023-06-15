"""
This script contains the code to run the redacted text detection algorithm on an input pdf
and get the statistics about the redaction.
"""

# Global imports
import os
import argparse
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path




# Local imports
from .src.redactiondetector import *

def convert_pdf_to_png(input_pdf_path: str):

    # Checking if it is a pdf file
    if not input_pdf_path.lower().endswith('.pdf'):
        raise FileNotFoundError

    # Load the image
    images = convert_from_path(input_pdf_path)
    return images


def main(arguments):
    input_images = convert_pdf_to_png(arguments.pdf_path)
    detector = RedactionDetector()
    # run the algorithm on all images, we have to do some shuffling of the color channels as the
    # algorithm expects images as BGR as input and also outputs them in this format, while PIL expects RGB images
    output_images = []
    output_info = {}

    for i, image in enumerate(input_images):
        input_image = np.array(image)[:, :, ::-1]
        output_image, _, redaction_percentage, num_redacted_regions = detector.run_algorithm(input_image_path_or_array=input_image)
        output_images.append(Image.fromarray(output_image[:, :, ::-1]))
        output_info['page-%d' % (i+1)] = {'Number of redacted regions': int(num_redacted_regions),
                                      'Percentage of redacted text': redaction_percentage}

    output_images[0].save(arguments.output_path, "PDF", resolution=100.0, save_all=True, append_images=output_images[1:])

    print(pd.DataFrame(output_info).astype(int).T)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args)

