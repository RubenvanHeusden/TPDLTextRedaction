"""
This script contains the full code for the text redaction algorithm as described in the notebook.
It also contains code to convert PDF images to PNG images so that they can be integrated into a pipeline

"""

import cv2
import pytesseract
import numpy as np
from pytesseract import Output


class RedactionDetector:
    """
    Class that implements the redaction detection algorithm and its separate components, as
    described in the notebook.

    """
    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Function that loads an image from a path.
        :param image_path: string specifying the path to the image
        :return: Numpy array with the image in BGR format.
        """
        # Checking if it is an image
        if image_path.lower().endswith('.png'):
            # Load the image in BGR format
            image = cv2.imread(image_path)
        else:
            raise FileNotFoundError
        return image

    @staticmethod
    def text_preprocessing(image: np.ndarray, text_pre_closing_kernel_size: tuple = (2, 2),
                           text_pre_guassian_blur_size: tuple = (3, 3)) -> np.ndarray:
        """
        :param image: Numpy array representing the input image in BGR format.
        :return: Numpy array with a grayscale image after applied operations.
        Method that applies image preprocessing for input to Tesseract, performs the following
        operations:
        1. Conversion of the iamge to grayscale
        2. Closing of the image with a 2 by 2 kernel to remove noise.
        3. Guassian blur with a 3 by 3 kernel.
        """
        # First we convert the input image to grayscale
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # We set up a kernel for the closing operation
        kernel = np.ones(text_pre_closing_kernel_size, np.uint8)

        # we perform closing, i.e. dilation followed by erosion
        closed_image = cv2.morphologyEx(image_grayscale, cv2.MORPH_CLOSE, kernel)

        # Finally we use a Guassian blur over the image with a 3 by 3 kernel size
        image_blurred = cv2.GaussianBlur(closed_image, ksize=text_pre_guassian_blur_size, sigmaX=0)

        return image_blurred

    @staticmethod
    def redaction_box_preprocessing(image: np.ndarray, box_pre_horizontal_closing_size: tuple = (1, 3),
                                    box_pre_vertical_closing_size: tuple = (3, 1),
                                    box_pre_bilat_filter_size: int = 5,
                                    box_pre_filter_sigma_color: int = 75,
                                    box_pre_filter_sigma_space: int = 75) -> np.ndarray:
        """
        :param image: Numpy array representing the input image in BGR format.
        :return: Numpy array with a grayscale image after applied operations.
        Method that applies image preprocessing for input the morphological operations, performs the following
        operations:
        1. Conversion of the iamge to grayscale
        2. Horizontal opening with a 1 by 3 kernel
        3. Vertical opening with a 3 by 1 kernel
        3. Bilateral filter.
        """
        # First we convert the input image to grayscale
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # We perform two sets of opening operations, a horizontal one, followed by a vertical one.
        horizontal_kernel = np.ones(box_pre_horizontal_closing_size, np.uint8)
        horizontally_opened_image = cv2.morphologyEx(image_grayscale, cv2.MORPH_OPEN, horizontal_kernel)

        # Apply kernel vertically over the horizontally opened image
        vertical_kernel = np.ones(box_pre_vertical_closing_size, np.uint8)
        vertically_opened_image = cv2.morphologyEx(horizontally_opened_image, cv2.MORPH_OPEN, vertical_kernel)

        # Perform a bilateral blur
        bilateral_blurred_image = cv2.bilateralFilter(vertically_opened_image, box_pre_bilat_filter_size,
                                                      box_pre_filter_sigma_color, box_pre_filter_sigma_space)

        return bilateral_blurred_image

    @staticmethod
    def remove_text(text_image: np.ndarray, redaction_box_image: np.ndarray,
                    tesseract_confidence: int = 65):
        # Count the total number of pixes of the pages occupied by words
        words_area = 0
        # Make a copy of the image where we will apply our transformation to.
        image_without_text = redaction_box_image.copy()
        # Get the width and height of the images
        image_height, image_width = redaction_box_image.shape[:2]

        # Set up the code to detect the leftmost and rightmost pieces of a page.
        left_boundary = [image_width]
        right_boundary = [0]
        top_boundary = [image_height]
        bottom_boundary = [0]

        # Specify the codes we want to detect
        codes = ['5.1.1.', '5.1.2.']

        # run tesseract on the image preprocessed for text
        tesseract_output = pytesseract.image_to_data(text_image, lang='nld+eng', output_type=Output.DICT)

        # Get the height of the text
        height = np.array(tesseract_output['height'])
        # get the median height of the text, we will use this to calculate
        # how much of the page is occupied by words
        median_text_height = np.median(height[height < 0.3 * image_height])

        # Get the number of detected text pages
        number_of_boxes = len(tesseract_output['level'])
        for box in range(number_of_boxes):
            # Get the coordinates of the text box if it actually contain any text
            if (tesseract_output['text'][box].strip() != "") and (tesseract_output['conf'][box] != -1) and (
                    tesseract_output['height'][box] < 0.3 * image_height):
                (x, y, w, h) = (tesseract_output['left'][box], tesseract_output['top'][box], tesseract_output['width'][box],
                                tesseract_output['height'][box])

                # If the text contains one of the codes we want to keep it and not remove it
                # from the page
                if any([code in tesseract_output['text'][box] for code in codes]):
                    # If the text is longer we want to adjust the width to include more specific subcodes
                    if len(tesseract_output['text'][box]) > 7:
                        sub_index = tesseract_output['text'][box].find('5.1.')
                        char_width = w / len(tesseract_output['text'][box])
                        w = int(char_width * 7)
                        x += int(char_width * sub_index)
                    # make the redaction boxes a white color
                    cv2.rectangle(image_without_text, (x, y), (x + w, y + h), (0, 0, 0), -1)
                # If its not a redaction box and the confidence is high enough
                # remove the text from hte page
                elif tesseract_output['conf'][box] > tesseract_confidence:
                    words_area += (w * h)
                    cv2.rectangle(image_without_text, (x, y), (x + w, y + h), (255, 255, 255), -1)

                    if median_text_height * 1.1 > tesseract_output['height'][box] > median_text_height * 0.9:
                        left_boundary.append(x)
                        right_boundary.append(x + w)
                        top_boundary.append(y)
                        bottom_boundary.append(y + h)

        image_without_text = cv2.threshold(image_without_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        return image_without_text, words_area, {'left_boundary': min(left_boundary),
                                                'right_boundary': max(right_boundary),
                                                'top_boundary': min(top_boundary),
                                                'bottom_boundary': max(bottom_boundary)}

    @staticmethod
    def determine_contours(image_without_text: np.ndarray, contour_opening_kernel_size: tuple = (5, 5)):
        # Find the contours we have so far and fill them so we can perform more operations on them
        contours = cv2.findContours(image_without_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for contour in contours:
            # filll contours with white
            cv2.drawContours(image_without_text, [contour], -1, (255, 255, 255), -1)

        # Here we remove noise by using an opening operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, contour_opening_kernel_size)
        opened_image = cv2.morphologyEx(image_without_text, cv2.MORPH_OPEN, kernel, iterations=4)

        # Draw rectangles
        new_contours = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = new_contours[0] if len(new_contours) == 2 else new_contours[1]

        return opened_image, new_contours

    @staticmethod
    def filter_contours(original_image: np.ndarray, contours: list, text_boundaries: dict):
        final_image = original_image.copy()
        final_contour_image = np.zeros([original_image.shape[0], original_image.shape[1]], dtype=np.uint8)

        # set thresholds for the sizes of the bounding boxes that we are going to keep
        area_treshold_min = 0.000125 * original_image.shape[0] * original_image.shape[1]
        area_treshold_max = 0.4 * original_image.shape[0] * original_image.shape[1]

        left_text_boundary = [text_boundaries['left_boundary']]
        right_text_boundary = [text_boundaries['right_boundary']]
        top_text_boundary = [text_boundaries['top_boundary']]
        bottom_text_boundary = [text_boundaries['bottom_boundary']]

        # boolean indicating if there is any redacted text on the page
        redacted_bool = False
        number_of_redacted_regions = 0
        total_contour_area = 0
        # save the final contours
        final_contours = []

        for contour in contours:
            # Find extreme points of contours
            contour_left = tuple(contour[contour[:, :, 0].argmin()][0])
            contour_right = tuple(contour[contour[:, :, 0].argmax()][0])
            contour_top = tuple(contour[contour[:, :, 1].argmin()][0])
            contour_bottom = tuple(contour[contour[:, :, 1].argmax()][0])

            # Filter out rectangles that are too small, or where the height is bigger than the width
            if area_treshold_max > cv2.contourArea(contour) > area_treshold_min and (
                    (contour_bottom[1] - contour_top[1]) < (contour_right[0] - contour_left[0])):
                final_contours.append(contour)
                # add the contours into the final image
                cv2.drawContours(final_image, [contour], -1, (0, 255, 0), -1)
                cv2.drawContours(final_contour_image, [contour], -1, (255, 255, 255), -1)

                left_text_boundary.append(contour_left[0])
                right_text_boundary.append(contour_right[0])
                top_text_boundary.append(contour_top[1])
                bottom_text_boundary.append(contour_bottom[1])

                total_contour_area += cv2.contourArea(contour)
                number_of_redacted_regions += 1

        text_area = ((max(right_text_boundary) - min(left_text_boundary)) * (
                    max(bottom_text_boundary) - min(top_text_boundary)))

        return final_image, final_contour_image, final_contours, total_contour_area, text_area


    @staticmethod
    def run_algorithm(input_image_path: str,
                      text_pre_closing_kernel_size: tuple = (2, 2),
                      text_pre_guassian_blur_size: tuple = (3, 3),
                      box_pre_horizontal_closing_size: tuple = (1, 3),
                      box_pre_vertical_closing_size: tuple = (3, 1),
                      box_pre_bilat_filter_size: int = 5,
                      box_pre_filter_sigma_color: int = 75,
                      box_pre_filter_sigma_space: int = 75,
                      tesseract_confidence: int = 65,
                      contour_opening_kernel_size: tuple = (5, 5)):
        """
        This functions implements the complete redaction detection algorithm and contains the options
        to set the parameters used as to experiment with different settings.
        :param input_image_path: string specifying the path to the input image
        :param text_pre_closing_kernel_size: size of the closing kernel for the text preprocessing step
        :param text_pre_guassian_blur_size: size of the kernel for the Gaussian blur for the text
        preprocessing step
        :param box_pre_horizontal_closing_size: size of the horizontal closing operation for the redaction
        box preprocessing step
        :param box_pre_vertical_closing_size:size of the vertical closing operation for the redaction
        box preprocessing step
        :param box_pre_bilat_filter_size: Size of the bilateral filter kernel for the redaction box
        preprocssing step.
        :param box_pre_filter_sigma_color: color sigma ofr the bilateral filter of the redaction box
        preprocessing step
        :param box_pre_filter_sigma_space: space sigma ofr the bilateral filter of the redaction box
        preprocessing ste
        :param tesseract_confidence: integer specifying the confidence level for Tesseract to
        consider something to be text
        :param contour_opening_kernel_size: kernel size of the opening operation in the contour detection step.
        :returns list of lists with the detected boxes, precentage of the words redacted, number of redacted regions
        """

        input_image = RedactionDetector.load_image(input_image_path)
        # Do the preprocessing
        image_text_pre = RedactionDetector.text_preprocessing(input_image, text_pre_closing_kernel_size)

        image_box_pre = RedactionDetector.redaction_box_preprocessing(input_image,
                                                    box_pre_horizontal_closing_size,
                                                    box_pre_vertical_closing_size,
                                                    box_pre_bilat_filter_size,
                                                    box_pre_filter_sigma_color,
                                                    box_pre_filter_sigma_space)
        # Remove the text
        image_without_text, total_words_area, text_boundaries = RedactionDetector.remove_text(image_text_pre, image_box_pre,
                                                                            tesseract_confidence)
        # First contour detection step
        image_with_contours, contours = RedactionDetector.determine_contours(image_without_text, contour_opening_kernel_size)
        # final contouring filtering step
        final_image_with_contours, final_contour_image, final_contours, total_contour_area, total_text_area = \
            RedactionDetector.filter_contours(input_image, contours, text_boundaries)

        # Automatically calculate some statistics on the number of redacted boxes, and the total percentage of
        # the page that is redacted.
        # Check how much of the text area is redacted (%)
        percentage_redacted_textarea = (
                    (total_contour_area / total_text_area) * 100) if total_contour_area and total_text_area else 0

        # Check how much of character area is redacted (%)
        total_area = total_contour_area + total_words_area
        percentage_redacted_words = ((total_contour_area / total_area) * 100) if total_contour_area else 0
        num_of_redacted_regions = len(final_contours)

        return final_contours, percentage_redacted_words, num_of_redacted_regions