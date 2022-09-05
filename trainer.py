import logging
import time

import torch
from torch import nn

from fedml.core import ClientTrainer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ImageSkippingTrainer(ClientTrainer):

    def get_model_params(self):
        return self.model.to('cpu').state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def adjust_learning_rate(self, optimizer, epoch, lr):
        """the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
        if epoch < 80:
            lr = lr
        elif epoch < 120:
            lr = lr * 0.1
        else:
            lr = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        for epoch in range(args.epochs):
            self.adjust_learning_rate(optimizer, epoch, args.lr)
            batch_loss = []
            # train for one epoch
            train_loss, train_acc, train_time, avg_skips =  self.trainHelper(train_data, model, criterion, optimizer, epoch, args.conf_thres, device)

            test_loss, prec = self.validate(testloader, model, criterion)
            # train_losses.append(train_loss)
            # train_accs.append(train_acc)
            # train_times.append(train_time)
            # skip_list.append(avg_skips)
            
            # evaluate on test set
            #return avg val accuracy
            # test_losses.append(test_loss)
            # test_accs.append(prec)

            # stats = pd.DataFrame([train_losses, test_losses, train_accs, test_accs, train_times, skip_list], index=['train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_time', 'skip_list'])
            # stats.to_csv(fdir + '/stats.csv')



    def trainHelper(self, train_data, model, criterion, optimizer, epoch, conf_threshold, device):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        skips = AverageMeter()

        end = time.time()
        for i, (input, target) in enumerate(train_data):
            # measure data loading time
            data_time.update(time.time() - end)

            input, target = input.to(device), target.to(device)

            # compute output
            output = model(input)
            probs = torch.nn.functional.softmax(output, dim=1)
            skip_input_sel = (probs < conf_threshold).float()
            x = torch.count_nonzero(skip_input_sel, axis = 1)
            x = torch.where(x == target.size(), torch.ones_like(x), torch.zeros_like(x))
            skip_input_sel = torch.where(probs > conf_threshold, 0, 1)
            # y shape based on class size
            # y = torch.ones(target.shape[0], 1)
            # y = y.cuda()
            # skip_input_sel = (torch.logical_not(torch.matmul(x,y))).float()
            #take skip inputs 
            output = torch.mul(probs,skip_input_sel)

            # indices = (skip_input_sel == 0).nonzero()[:, 0]
            # output[indices, torch.zeros(indices.size(dim=0)).long()] += 20

            target = torch.mul(target, torch.transpose(skip_input_sel,0, 1))[0].to(dtype=torch.int64)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = self.accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            skips.update(skip_input_sel.size()[0] - torch.count_nonzero(skip_input_sel))
            

            if i % 100 == 0:
                logging.info('Trainer_id {}\t'
                    'Local Training Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Skips {skips.val:.3f} ({skips.avg:.3f})\t'.format(self.id,
                    epoch, i, len(train_data), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, skips=skips))

        return losses.avg, top1.avg, batch_time.avg, skips.avg

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        # switch to evaluate mode
        model.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        criterion = nn.CrossEntropyLoss().to(device)

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(test_data):
                input, target = input.to(device), target.to(device)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # # measure accuracy and record loss
                prec = self.accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % 100 == 0:
                #     print('Test: [{0}/{1}]\t'
                #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #     'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                #     i, len(val_loader), batch_time=batch_time, loss=losses,
                #     top1=top1))

        logging.info('Trainer_id {}\t''* Prec {top1.avg:.3f}% '.format(self.id, top1=top1))

        return losses.avg, top1.avg

        