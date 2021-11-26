import numpy as np
import pandas as pd 

# constants
from config import config

# Data management functions (Load, save)
from preprocessing.data_management import load_pipeline, load_dataset

# file management
import os

# Set the saved model directory
model_dir = config.SAVED_MODEL_PATH

# list all model names
last_model_name = os.listdir(model_dir)

# Set the model path
pipeline_file_name = os.path.join(model_dir, last_model_name[-1])
#print('last model saved for prediction: ', pipeline_file_name)

# Load the model
_titanic_pipe = load_pipeline(pipeline_file_name)

def make_prediction(input_data = load_dataset(file_name = config.TRAIN_FILE)):
    data = pd.DataFrame(input_data)
    prediction = _titanic_pipe.predict(data[config.KEEP_FEATURES]) # or data[config.KEEP_FEATURES] if needed)
    output = prediction

    results = {
        'prediction': output,
        'model_name': pipeline_file_name,
        'version':'version1'
    }

    print(output)

    return results

if __name__ == "__main__":
    make_prediction()