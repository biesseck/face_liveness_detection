import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


# Bernardo
class CallBackEpochLogging(object):
    def __init__(self, frequent, total_step, batch_size, num_batches, start_step=0, writer=None):
        self.frequent: int = frequent
        self.rank: int = 0
        self.world_size: int = 1
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.num_batches: int = num_batches
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 total_loss: AverageMeter,
                 train_evaluator,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        # if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
        try:
            # speed: float = self.frequent * self.batch_size / (time.time() - self.tic)   # original
            speed: float = self.frequent * self.num_batches / (time.time() - self.tic)    # Bernardo
            speed_total = speed * self.world_size
        except ZeroDivisionError:
            speed_total = float('inf')

        #time_now = (time.time() - self.time_start) / 3600
        #time_total = time_now / ((global_step + 1) / self.total_step)
        #time_for_end = time_total - time_now
        time_now = time.time()
        time_sec = int(time_now - self.time_start)
        time_sec_avg = time_sec / (global_step - self.start_step + 1)
        eta_sec = time_sec_avg * (self.total_step - global_step - 1)
        time_for_end = eta_sec/3600

        metrics = train_evaluator.evaluate()

        if self.writer is not None:
            # self.writer.add_scalar('time_for_end', time_for_end, global_step)
            self.writer.add_scalar('learning_rate', learning_rate, epoch)
            # self.writer.add_scalar('loss/train_reconst_loss', reconst_loss.avg, epoch)
            # self.writer.add_scalar('loss/train_class_loss', class_loss.avg, epoch)
            self.writer.add_scalar('loss/train_total_loss', total_loss.avg, epoch)
            self.writer.add_scalar('acc/train_acc', metrics['acc'], epoch)
            self.writer.add_scalar('apcer/train_apcer', metrics['apcer'], epoch)
            self.writer.add_scalar('bpcer/train_bpcer', metrics['bpcer'], epoch)
            self.writer.add_scalar('acer/train_acer', metrics['acer'], epoch)
        if fp16:
            msg = " Epoch: %d   TotalLoss %.4f    acc: %.4f%%    apcer: %.4f%%    bpcer: %.4f%%    acer: %.4f%%   LR %.6f   Global Step: %d   " \
                    "Fp16 Grad Scale: %2.f   Speed %.2f samples/sec   Required: %1.f hours" % (
                        epoch, total_loss.avg, metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer'], learning_rate, global_step,
                        grad_scaler.get_scale(), speed_total, time_for_end
                    )
        else:
            msg = " Epoch: %d   TotalLoss %.4f    acc: %.4f%%    apcer: %.4f%%    bpcer: %.4f%%    acer: %.4f%%   LR %.6f   Global Step: %d   Speed %.2f samples/sec   " \
                    "Required: %1.f hours" % (
                        epoch, total_loss.avg, metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer'], learning_rate, global_step, speed_total, time_for_end
                    )
        logging.info(msg)
        total_loss.reset()

        if self.init:
            self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


# Bernardo
class EvaluatorLogging(object):
    def __init__(self, num_samples, batch_size, num_batches):
        self.num_samples: int = num_samples
        self.batch_size: int = batch_size
        self.num_batches: int = num_batches
        
        self.curr_idx = 0
        self.curr_batch = 0
        self.all_pred_labels = torch.ones((num_samples))
        self.all_true_labels = torch.zeros((num_samples))


    def update(self, pred_labels, true_labels):
        assert pred_labels.size(0) == true_labels.size(0), 'Error: pred_labels.size(0) is different from true_labels.size(0). Sizes must be equal.'
        self.all_pred_labels[self.curr_idx:self.curr_idx+pred_labels.size(0)] = pred_labels
        self.all_true_labels[self.curr_idx:self.curr_idx+true_labels.size(0)] = true_labels
        self.curr_idx += pred_labels.size(0)
        self.curr_batch += 1


    def evaluate(self):
        pred_labels = self.all_pred_labels[:self.curr_idx]
        true_labels = self.all_true_labels[:self.curr_idx]

        tp = float(torch.sum(torch.logical_and(pred_labels, true_labels)))
        fp = float(torch.sum(torch.logical_and(pred_labels, torch.logical_not(true_labels))))
        tn = float(torch.sum(torch.logical_and(torch.logical_not(pred_labels), torch.logical_not(true_labels))))
        fn = float(torch.sum(torch.logical_and(torch.logical_not(pred_labels), true_labels)))

        tpr = 0 if (tp + fn == 0) else (tp / (tp + fn)) * 100.0
        fpr = 0 if (fp + tn == 0) else (fp / (fp + tn)) * 100.0

        tar = tpr
        far = fpr
        frr = 0 if (fn + tp == 0) else (fn / (fn + tp)) * 100.0

        acc = 0 if pred_labels.size(0) == 0 else (tp + tn) / float(pred_labels.size(0)) * 100.0

        # source: https://chalearnlap.cvc.uab.cat/challenge/33/track/33/metrics/
        # source: https://docs.openvino.ai/2023.0/omz_tools_accuracy_checker_metrics.html
        apcer = 0 if (fp + tn == 0) else (fp / (fp + tn)) * 100.0
        npcer = 0 if (fn + tp == 0) else (fn / (fn + tp)) * 100.0
        bpcer = npcer
        acer = (apcer + bpcer) / 2.0
        hter = (far + frr) / 2.0

        metrics = {'tp': tp,
                   'fp': fp,
                   'tn': tn,
                   'fn': fn,
                   'tpr': tpr,
                   'fpr': fpr,
                   'tar': tar,
                   'far': far,
                   'frr': frr,
                   'acc': acc,
                   'apcer': apcer,
                   'npcer': npcer,
                   'bpcer': bpcer,
                   'acer': acer,
                   'hter': hter
                   }

        return metrics


    def reset(self):
        self.curr_idx = 0
        self.curr_batch = 0
        self.all_pred_labels[:] = 1
        self.all_true_labels[:] = 0