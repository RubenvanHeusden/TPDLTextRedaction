"""
This script contains the full code for the text redaction algorithm as described in the notebook.
It also contains code to convert PDF images to PNG images so that they can be integrated into a pipeline

"""

import cv2
import json
import time
import pytesseract
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pytesseract import Output
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


class RedactionDetector:
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