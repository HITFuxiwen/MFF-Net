import os
import time

import numpy as np
import torch
from torchvision import transforms


class Eval_thread():
    def __init__(self, loader, output_dir, cuda):
        self.loader = loader
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        mae = self.Eval_mae()

        Fm, prec, recall = self.Eval_fmeasure()
        max_f = Fm.max().item()
        mean_f = Fm.mean().item()
        prec = prec.cpu().numpy()
        recall = recall.cpu().numpy()
        avg_p = self.Eval_AP(prec, recall)  # AP
        Fm = Fm.cpu().numpy()
        # print(Fm)
        auc, TPR, FPR = self.Eval_auc()
        TPR = TPR.cpu().numpy()
        FPR = FPR.cpu().numpy()
        # print(TPR)
        # print(FPR)
        return ' {:.4f} AP.'.format(avg_p), ' {:.4f} MAXf.'.format(max_f), ' {:.4f} MAE.'.format(mae), ' {:.4f} AUC.'.format(auc)

    def Eval_fmeasure(self):
            beta2 = 0.3
            avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                trans = transforms.Compose([transforms.ToTensor()])
                for pred, gt in self.loader:
                    if self.cuda:
                        pred = trans(pred).cuda()
                        gt = trans(gt).cuda()
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                        torch.min(pred) + 1e-20)
                    else:
                        pred = trans(pred)
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                        torch.min(pred) + 1e-20)
                        gt = trans(gt)
                    prec, recall = self._eval_pr(pred, gt, 255)
                    f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                    f_score[f_score != f_score] = 0  # for Nan
                    avg_f += f_score
                    avg_p += prec
                    avg_r += recall
                    img_num += 1.0
                Fm = avg_f / img_num
                avg_p = avg_p / img_num
                avg_r = avg_r / img_num
                return Fm, avg_p, avg_r

    def Eval_AP(self, prec, recall):
        ap_r = np.concatenate(([0.], recall, [1.]))
        ap_p = np.concatenate(([0.], prec, [0.]))
        sorted_idxes = np.argsort(ap_r)
        ap_r = ap_r[sorted_idxes]
        ap_p = ap_p[sorted_idxes]
        count = ap_r.shape[0]

        for i in range(count - 1, 0, -1):
            ap_p[i - 1] = max(ap_p[i], ap_p[i - 1])

        i = np.where(ap_r[1:] != ap_r[:-1])[0]
        ap = np.sum((ap_r[i + 1] - ap_r[i]) * ap_p[i + 1])
        return ap

    def Eval_mae(self):
            avg_mae, img_num = 0.0, 0.0
            with torch.no_grad():
                trans = transforms.Compose([transforms.ToTensor()])
                for pred, gt in self.loader:
                    if self.cuda:
                        pred = trans(pred).cuda()
                        gt = trans(gt).cuda()
                    else:
                        pred = trans(pred)
                        gt = trans(gt)
                    mea = torch.abs(pred - gt).mean()
                    if mea == mea:  # for Nan
                        avg_mae += mea
                        img_num += 1.0
                avg_mae /= img_num
                return avg_mae.item()
    def Eval_auc(self):

        avg_tpr, avg_fpr, avg_auc, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                TPR, FPR = self._eval_roc(pred, gt, 500)
                avg_tpr += TPR
                avg_fpr += FPR
                img_num += 1.0
            avg_tpr = avg_tpr / img_num
            avg_fpr = avg_fpr / img_num

            sorted_idxes = torch.argsort(avg_fpr) 
            avg_tpr = avg_tpr[sorted_idxes]
            avg_fpr = avg_fpr[sorted_idxes]
            avg_auc = torch.trapz(avg_tpr, avg_fpr)

            return avg_auc.item(), avg_tpr, avg_fpr


    def _eval_roc(self, y_pred, y, num):
        if self.cuda:
            TPR, FPR = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            fp = (y_temp * (1 - y)).sum()
            tn = ((1 - y_temp) * (1 - y)).sum()
            fn = ((1 - y_temp) * y).sum()

            TPR[i] = tp / (tp + fn + 1e-20)
            FPR[i] = fp / (fp + tn + 1e-20)

        return TPR, FPR

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                    1e-20)
        return prec, recall