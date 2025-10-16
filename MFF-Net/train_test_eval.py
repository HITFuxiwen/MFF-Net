import os
import torch
import Training
import Testing
import Testing2
import eva2
# from Evaluation import main
import argparse
import eva
from torchsummary import summary
from ImageDepthNet6 import ImageDepthNet6
from ImageDepthNet5 import ImageDepthNet5
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=100000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./80.7_T2T_ViT_t_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--pretrained_model2', default='./cmt_small.pth', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=12, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=60000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=100000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='ISTD/ISTD', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='ckpt/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds3/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='test_new')
    parser.add_argument('--test_set', type=str, default='DSC')
    parser.add_argument('--test_dir', type=str, default='cross')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    num_gpus = 2
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        # Testing.test_net(args, '0')
        Testing2.test_net(args)
    if args.Evaluation:
        eva2.evaluate(args)