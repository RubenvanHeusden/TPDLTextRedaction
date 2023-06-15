"""
This script contains the code to run the redacted text detection algorithm on an input pdf
and get the statistics about the redaction. The script can also be ran on a folder containing pdf
files.
"""

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_or_folder_path', type=str, required=True)




