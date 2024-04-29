import os
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import random
from MTCNN_InceptionV1Model import MTCNN_inceptionResnetV1
from mesonet import Meso4

model = MTCNN_inceptionResnetV1()
meso_model = Meso4()

test_data = os.listdir('test_data')
y_test = []
y_pred = []


def count_files(folder_path):
    # Initialize a counter for the number of files
    num_files = 0

    # Iterate through each item (file or folder) in the specified folder
    for _, _, files in os.walk(folder_path):
        # Count each file
        num_files += len(files)
    
    # Return the total number of files
    return num_files


# Testing the classification.
"""
Real = 0
fake = 1
True positives: Deepfake images correctly identified as deepfake.
False positives: Real images incorrectly identified as deepfake.
True negatives: Real images correctly identified as real.
False negatives: Deepfake images incorrectly identified as real.
"""
for folder in test_data:
    if folder == 'Real_frames':
        # Finding the number of real frames.
        Real_count = count_files('test_data' + '/' + folder)
        # Adding similar number of 0 to the y_test data.
        y_test = y_test + [0] * Real_count

        # Iterating through the real frames and predicting the results.
        for root, _, files in os.walk('test_data' + '/' + folder):
            for file in files:
                # Check if the file is an image (you can add more image extensions if needed)
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # Construct the full path to the image
                    image_path = os.path.join(root, file)

                    # MTCNN_inceptionResnxtV1 model predictions.
                    # predictions = model.classify_image(image_path)

                    # Mesonet model predictions
                    predictions = meso_model.predict_on_image(meso_model.model, image_path)
                
                    y_pred = y_pred + [predictions]

    elif folder == 'Fake_frames':
        # Finding the number of Fake frames.
        Fake_count = count_files('test_data' + '/' + folder)
        # Adding similar number of 1 to the y_test data.
        y_test = y_test + [1] * Fake_count

        # Iterating through the real frames and predicting the results.
        for root, _, files in os.walk('test_data' + '/' + folder):
            for file in files:
                # Check if the file is an image (you can add more image extensions if needed)
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # Construct the full path to the image
                    image_path = os.path.join(root, file)

                    # MTCNN_inceptionResnxtV1 model predictions.
                    # predictions = model.classify_image(image_path)

                    # Mesonet model predictions
                    predictions = meso_model.predict_on_image(meso_model.model, image_path)
                    y_pred = y_pred + [predictions]
    else:
        pass

print(y_test)
print(y_pred)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
