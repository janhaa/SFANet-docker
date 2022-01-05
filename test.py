import cv2
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from datasets.crowd import Crowd
from models import M_SFANet_UCF_QNRF

# Simple preprocessing.
trans = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])

# An example image with the label = 1236.
img = Image.open("./images/img_0071.jpg").convert('RGB')
height, width = img.size[1], img.size[0]
height = round(height / 16) * 16
width = round(width / 16) * 16
img = cv2.resize(np.array(img), (width,height), cv2.INTER_CUBIC)
img = trans(Image.fromarray(img))[None, :]

model = M_SFANet_UCF_QNRF.Model()
# Weights are stored in the Google drive link.
# The model are originally trained on a GPU but, we can also test it on a CPU.
model.load_state_dict(torch.load("./Paper's_weights_UCF_QNRF/best_M-SFANet*_UCF_QNRF.pth", 
                                 map_location = torch.device('cpu')))

# Evaluation mode
model.eval()
density_map = model(img)
# Est. count = 1168.37 (67.63 deviates from the ground truth)
print(torch.sum(density_map).item())
