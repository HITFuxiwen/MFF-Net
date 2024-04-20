import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from ImageDepthNet5 import ImageDepthNet5
from ImageDepthNet6 import ImageDepthNet6
from ImageDepthNet7 import ImageDepthNet7
from torch.utils import data
import numpy as np
import os
import eva2

def thresold(output_s , t):
    output_s[output_s > t] = 1.0
    output_s[output_s <= t] = 0.0
    return output_s

def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet5(args)
    # net = nn.DataParallel(net)
    # net = ImageDepthNet3(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    # model_path = args.save_model_dir + '299RGB_VST_resdct.pth'
    for epoch in range(31,32,5):
        
        num = str(epoch)
        model_path = args.save_model_dir + num + 'RGB_VST_rcmt_swam.pth'

        state_dict = torch.load(model_path)
        from collections import OrderedDict
    
        new_state_dict = state_dict

        # load params
        net.load_state_dict(new_state_dict)
        print('Model loaded from {}'.format(model_path))

        # load model
        # net.load_state_dict(torch.load(model_path))
        # model_dict = net.state_dict()
        # print('Model loaded from {}'.format(model_path))

        test_paths = args.test_paths
        test_dir_img = test_paths

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                    Starting testing:
                        dataset: {}
                        Testing size: {}
                    '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, image_w, image_h, image_path = data_batch
            images = Variable(images.cuda())

            starts = time.time()
            # at, ft, outputs_saliency = net(images)
            outputs_saliency = net(images)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1, m16, m8, m4, m1 = outputs_saliency
            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_s = torch.sigmoid(mask_1_1)
            output_s = output_s.data.cpu().squeeze(0)
            # q = torch.mean(output_s)
            # print(q)
            # output_s = thresold(output_s, q)

            image_w , image_h = 640, 480
            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = transform(output_s)

            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/RGB_VST_rcmt_swam/'
            # save_test_path = args.save_test_path_root + dataset + '/RGB_VST3/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            output_s.save(os.path.join(save_test_path, filename + '.png'))

        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list) * 1000))
        eva2.evaluate(args)





