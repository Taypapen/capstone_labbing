
import time
import shutil
import copy

from tqdm import tqdm
from mixup import mixup_data, mixup_criterion
from utilities import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler


def train_loop(model, dataloader, criterion, epoch, optimizer, mixup=False):

    data_time = AverageTracker()
    batch_time = AverageTracker()

    mixup_alpha = 0.4

    print("Current Epoch: " + str(epoch))

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, (input, lengths, labels) in enumerate(dataloader):
        if mixup:
            input, labels_a, labels_b, lam = mixup_data(input, labels, mixup_alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()
        else:
            labels = labels.cuda()

        optimizer.zero_grad()

        logits = model(input.unsqueeze(1).cuda(), lengths=lengths)

        if mixup:
            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, logits)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item()*input.size(0)

        if mixup:
            running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        else:
            running_corrects += torch.sum(predicted == labels.data)
        running_all += input.size(0)
    print("Running Loss: {}, Running Corrects: {}, Running All: {}".format(running_loss,running_corrects,running_all))

    return model


def evaluate(model, dset_loader, criterion):

    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

    print('{} in total\tCR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


class FullTrainer(object):
    def __init__(self, model, dataloader, epochs, criterion=nn.CrossEntropyLoss(), save_dir='./', state_path=None, model_weights_only=False, lr=0.003, optim='sgd'):
        #Model/Training Params
        self.epoch = 0
        self.model = model
        self.dataset = dataloader
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion
        self.state_path = state_path
        self.optimizer = get_optimizer(optim, self.model.parameters(), lr=lr)
        self.scheduler = CosineScheduler(lr, epochs)
        self.model_only = model_weights_only
        self.allow_size_mismatch = True

        #Checkpoint Params
        self.checkpoint_filename = 'ckpt.pth.tar'
        self.best_fn = 'ckpt.best.pth.tar'
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.best_perf = 0

        self._initialize_weights_randomly()

        if state_path is not None:
            self.load_checkpoint(self.model, self.optimizer)

        self.initialize_training()

        self.eval_best_performance()

    def load_checkpoint(self, model, optimizer):
        checkpoint = torch.load(self.state_path)
        loaded_state_dict = checkpoint['model_state_dict']
        if self.allow_size_mismatch:
            loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
            model_state_dict = model.state_dict()
            model_sizes = {k: v.shape for k, v in model_state_dict.items()}
            mismatched_params = []
            for k in loaded_sizes:
                if loaded_sizes[k] != model_sizes[k]:
                    mismatched_params.append(k)
            for k in mismatched_params:
                del loaded_state_dict[k]
        self.model = model.load_state_dict(loaded_state_dict, strict=False)
        self.optimizer = get_optimizer(optim, self.model.parameters(), lr=self.lr)
        if not self.model_only:
            self.epoch = checkpoint['epoch_idx']
            if self.epoch != self.epochs:
                self.optimizer = optimizer.load_state_dict(checkpoint['optimimzer_state_dict'])
                self.scheduler.adjust_lr(self.optimizer, self.epoch - 1)
                self.best_perf = checkpoint['best_perf']

    def _initialize_weights_randomly(self):
        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

    def initialize_training(self):
        while self.epoch < self.epochs:
            start_epoch = time.time()
            self.model = train_loop(self.model, self.dataset['train'], self.criterion, self.epoch, self.optimizer)
            acc_avg_val, loss_avg_val = evaluate(self.model, self.dataset['val'], self.criterion)
            print('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', self.epoch, loss_avg_val,
                                                                                    acc_avg_val, showLR(self.optimizer)))
            save_dict = {
                'epoch_idx': self.epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }
            self.save_checkpoint(save_dict, acc_avg_val)
            self.scheduler.adjust_lr(self.optimizer, self.epoch)
            self.epoch += 1
            epoch_len = time.time() - start_epoch
            print("Epoch len: {} Estimated Remaining: {} Min".format(str(epoch_len),
                                                                     str(((self.epochs - self.epoch) * epoch_len) / 60)))

    def eval_best_performance(self):
        best_filepath = os.path.join(self.save_dir, self.best_fn)
        assert os.path.isfile(best_filepath)
        checkpoint = torch.load(best_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        acc_avg_test, loss_avg_test = evaluate(self.model, self.dataset['test'], self.criterion)
        print('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))

    def save_checkpoint(self, save_dict, current_score):
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_filename)

        self.is_best = current_score > self.best_perf
        if self.is_best:
            self.best_perf = current_score
            best_filepath = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_perf'] = self.best_perf

        torch.save(save_dict, checkpoint_fp)
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_filepath)


