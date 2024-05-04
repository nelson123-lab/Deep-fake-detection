import cv2
from mesonet import Meso4
import numpy as np

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

""" 
Testing using a saved image in a path
"""
# def preprocess_image(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     # Resize the image to match the input dimensions of the model
#     img = cv2.resize(img, (image_dimensions['width'], image_dimensions['height']))
#     # Convert image to float32 and normalize it
#     img = img.astype('float32') / 255.0
#     # Expand dimensions to match the input shape of the model
#     img = np.expand_dims(img, axis=0)
#     return img

# def predict_on_image(model, image_path):
#     # Preprocess the image
#     img = preprocess_image(image_path)
#     # Make prediction using the model
#     prediction = model.predict(img)
#     return prediction

"""
Testing using an image file.
"""
def preprocess_image(image):
    # Resize the image to match the input dimensions of the model
    img = cv2.resize(image, (image_dimensions['width'], image_dimensions['height']))
    # Convert image to float32 and normalize it
    img = img.astype('float32') / 255.0
    # Expand dimensions to match the input shape of the model
    img = np.expand_dims(img, axis=0)
    return img

def predict_on_image(model, image):
    # Preprocess the image
    img = preprocess_image(image)
    # Make prediction using the model
    prediction = model.predict(img)
    return prediction

