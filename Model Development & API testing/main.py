import os
import argparse

from wifi_sensing import MotionSentimentDetection

from util import*
from variables import*
        
if __name__ == "__main__":
    if not os.path.exists('data and results/weights/dnn weights/'):
        os.makedirs('data and results/weights/dnn weights/')
    if not os.path.exists('data and results/weights/conv1d weights/'):
        os.makedirs('data and results/weights/conv1d weights/')

    parser = argparse.ArgumentParser()
    parser.add_argument(
                    "--ModelType", 
                    help="Select the required model (DNN or Conv1D)")
    args = parser.parse_args()
    model_type = args.ModelType

    model = MotionSentimentDetection(model_type)
    model.run()


'''
python main.py --ModelType dnn

'''