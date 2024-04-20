import os.path as osp
from eval import Eval_thread
from dataloader import EvalDataset


def evaluate(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    # pred_dir= osp.join(pred_dir, 'ISTD/RGB_VST_wo/')
    pred_dir= osp.join(pred_dir, 'test_new/RGB_VST_rcmt_swam/')
        
    gt_dir = osp.join(osp.join(gt_dir, 'test_new/test_GT/'))

    loader = EvalDataset(pred_dir, gt_dir)
    thread = Eval_thread(loader, output_dir, cuda=True)
    print(thread.run())
