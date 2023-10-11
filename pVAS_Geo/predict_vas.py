"""
Predicts VAS scores of mammographic images using the models trained on 
the PROCAS data. It first reads a .tfrecords file at a given path, predicts the 
scores per image and writes the scores, client_id and image side into a csv file.
"""

import argparse
import model_lib
import numpy as np
import time
import os


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument(
        "-ip", "--data-path",
        help="The path to the data folder",
        dest="data_path",
        required=False,
        default="../data/"
    )
    
    parser.add_argument(
        "-df", "--data-file",
        help="The path to the data info csv file",
        dest="data_file",
        required=False,
        default="../data/mock_test_data.csv"
    )

    parser.add_argument(
        "-op", "--results-path",
        help="The path to the folder where the results will be stored",
        dest="results_path",
        required=False,
        default="../results/"
    )

    parser.add_argument(
        "-v", "--view",
        help="The view of the mammographic images",
        dest="view",
        required=False,
        choices=['CC', 'MLO'],
        default='CC'
    )

    parser.add_argument(
        "-t", "--image-type",
        help="The view of the mammographic images",
        dest="image_type",
        required=False,
        choices=['raw', 'processed'],
        default='raw'
    )

    parser.add_argument(
        "-mp", "--model-path",
        help="The path to the model",
        dest="model_path",
        required=False,
        default="../models/raw/MLO_model_raw"
    )   
        
    return parser

if __name__ == "__main__":
    # Set the verbosity level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Build the argument parser
    args = build_parser().parse_args()

    # Measure the execution time
    start_time = time.time()

    # Run the VAS prediction method    
    model_lib.test_model(args.model_path,
                args.data_path,
                args.data_file,
                args.view,
                args.results_path,
                args.image_type 
                )
        
    elapsed_time = (time.time() - start_time)/60.0
    print('Executed in ' + str(int(np.ceil(elapsed_time))) + ' minutes')    