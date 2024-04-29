# import torch
# import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image

# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# mtcnn = MTCNN(select_largest = False,
#               post_process = False,
#               device = DEVICE).to(DEVICE).eval()

# model = InceptionResnetV1(pretrained = "vggface2",
#                           classify = True,
#                           num_classes = 1,
#                           device = DEVICE)

# checkpoint = torch.load("Model_weights/resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(DEVICE)
# model.eval()

# def predict(model, input_image: Image.Image):
#     """Predict the label of the input_image"""
#     input_image_tensor = mtcnn(input_image)
#     if input_image_tensor is None:
#         raise Exception('No face detected')

#     input_image_tensor = input_image_tensor.unsqueeze(0)  # add the batch dimension
#     input_image_tensor = F.interpolate(input_image_tensor, size=(256, 256), mode='bilinear', align_corners=False)

#     input_image_tensor = input_image_tensor.to(DEVICE)
#     input_image_tensor = input_image_tensor.to(torch.float32)
#     input_image_tensor /= 255.0

#     with torch.no_grad():
#         output = torch.sigmoid(model(input_image_tensor).squeeze(0))
#         prediction = "real" if output.item() < 0.5 else "fake"

#         real_prediction = 1 - output.item()
#         fake_prediction = output.item()

#         confidences = {
#             'real': real_prediction,
#             'fake': fake_prediction
#         }

#     return confidences

# def classify_image(model, image_path, threshold = 0.5):
#   input_image = Image.open(image_path)
#   confidences = predict(model, input_image)
#   # classify as real and fake
#   if confidences['fake'] >= confidences['real']:
#       return 1
#   else: return 0

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import warnings

class MTCNN_inceptionResnetV1:
    def __init__(self, model_path="Model_weights/resnetinceptionv1_epoch_32.pth", device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        warnings.filterwarnings("ignore")
        self.device = device
        self.mtcnn = MTCNN(select_largest=False, post_process=False, device=device).eval()
        self.model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=device)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict(self, input_image: Image.Image):
        """Predict the label of the input_image"""
        input_image_tensor = self.mtcnn(input_image)
        if input_image_tensor is None:
            raise Exception('No face detected')

        input_image_tensor = input_image_tensor.unsqueeze(0)  # add the batch dimension
        input_image_tensor = F.interpolate(input_image_tensor, size=(256, 256), mode='bilinear', align_corners=False)

        input_image_tensor = input_image_tensor.to(self.device)
        input_image_tensor = input_image_tensor.to(torch.float32)
        input_image_tensor /= 255.0

        with torch.no_grad():
            output = torch.sigmoid(self.model(input_image_tensor).squeeze(0))
            prediction = "real" if output.item() < 0.5 else "fake"

            real_prediction = 1 - output.item()
            fake_prediction = output.item()

            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }

        return confidences

    def classify_image(self, image_path, threshold = 0.5):
        input_image = Image.open(image_path)
        confidences = self.predict(input_image)
        # classify as real and fake
        if confidences['fake'] >= confidences['real']:
            return 1
        else:
            return 0

# face_classifier = MTCNN_inceptionResnetV1()
# result = face_classifier.classify_image("test_data/Fake_frames/3fake_0.jpg")
# print(result)