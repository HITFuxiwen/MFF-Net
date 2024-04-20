from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os


image = Image.open('./preds3/test_new/RGB_VST_rcmt_cross/132-1.png').convert('L')
new_img = trans.Scale((56, 56))(image)
new_img.save('temp.png')