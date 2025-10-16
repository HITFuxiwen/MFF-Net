from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root, sset):
        
        images = []
        labels = []


        img_root = pred_root + sset 
        # img_files = os.listdir(img_root)
        # for img in img_files:
        #     images.append(img_root + img[:-4]+'.png')
        #     labels.append(label_root + '/GT_ISTD/' + img[:-4]+'.png')
        
        img_root = pred_root + '/DSC/' 
        img_files = os.listdir(img_root)
        for img in img_files:
            images.append(img_root + img[:-4]+'.png')
            labels.append(label_root + '/GT_AISTD/' + img[:-4]+'.png')

        # img_root = pred_root + '/AISTD/' 
        # img_files = os.listdir(img_root)
        # for img in img_files:
        #     images.append(img_root + img[:-4]+'.png')
        #     labels.append(label_root + '/GT_AISTD/' + img[:-4]+'.png')
        
        # img_root = pred_root + '/SBU/' 
        # img_files = os.listdir(img_root)
        # for img in img_files:
        #     images.append(img_root + img[:-4]+'.png')
        #     labels.append(label_root + '/GT_SBU/' + img[:-4]+'.png')
        
        # img_root = pred_root + '/USR/' 
        # img_files = os.listdir(img_root)
        # for img in img_files:
        #     images.append(img_root + img[:-4]+'.png')
        #     labels.append(label_root + '/GT_USR/' + img[:-4]+'.png')

        self.image_path = images
        self.label_path = labels

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
