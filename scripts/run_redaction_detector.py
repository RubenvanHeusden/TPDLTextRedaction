"""
This script contains the code to run the redacted text detection algorithm on an input pdf
and get the statistics about the redaction.
"""

# Global imports
import io
import os
import argparse
import pandas as pd
from PIL import Image
from img2table.document import Image as TableImage
from pdf2image import convert_from_path




# Local imports
from redactiondetector import *

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

        if args.exclude_tables:
            byte_image = io.BytesIO()
            # image.save expects a file-like as a argument
            image.save(byte_image, format=image.format)
            # Turn the BytesIO object back into a bytes object
            byte_image = byte_image.getvalue()
            table_image = TableImage(src=byte_image)
            # Table identification
            try:
                image_tables = table_image.extract_tables()
            except:
                image_tables = []
                
            # This module is recall oriented so we should require the tables to be of a larger size
            page_width, page_height = image.size
            detected_tables = []
            for table in image_tables:
                table_width = table.bbox.x2 - table.bbox.x1
                table_height = table.bbox.y2 - table.bbox.y1
                # is the size of the table in pixels big enough compared to the complete page?
                if (table_width / page_width) > 0.50 and table_height > 50:
                    # now check the number of cells
                    if ((table.df.shape[0] >= 2) and (table.df.shape[1] >=3)):
                        detected_tables.append(True)
            contains_tables = any(detected_tables)
            if contains_tables:
                redaction_percentage = 0
                num_redacted_regions = 0
                output_images.append(image)
                output_info['page-%d' % (i+1)] = {'Number of redacted regions': int(num_redacted_regions),
                                              'Percentage of redacted text': redaction_percentage}
                continue

        input_image = np.array(image)[:, :, ::-1]
        output_image, _, redaction_percentage, num_redacted_regions = detector.run_algorithm(input_image_path_or_array=input_image)
        output_images.append(Image.fromarray(output_image[:, :, ::-1]))
        output_info['page-%d' % (i+1)] = {'Number of redacted regions': int(num_redacted_regions),
                                      'Percentage of redacted text': redaction_percentage}

    output_images[0].save(arguments.output_path, "PDF", resolution=100.0, save_all=True, append_images=output_images[1:])
    dataframe_name = arguments.output_path.replace('.pdf', '.csv')
    pd.DataFrame(output_info).astype(int).T.to_csv(dataframe_name, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--exclude_tables', type=bool, default=False)
    args = parser.parse_args()

    main(args)

