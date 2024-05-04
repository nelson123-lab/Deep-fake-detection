from mesonet import Meso4
# from preprocessMeso import * 
# from screenCapture import *
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

model = Meso4()
model.load_weights('checkpoints/Meso4_DF.h5')

print(model.summary())
# capture = MSSscreenCaptue()
# image = capture.fullScreenshot()
# capture.saveScreenshot('screenshot1.png')  # Save screenshot to file
# capture.showScreenshot()  # Display the screenshot in a window

# predictions = predict_on_image(model, image)
# print(predictions)