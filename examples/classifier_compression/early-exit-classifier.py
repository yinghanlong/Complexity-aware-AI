#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import time
import os
import sys
import random
import traceback
import logging
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
from torch.optim.lr_scheduler import ReduceLROnPlateau
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model
import parser
import util_bin #Binary Operation
import torch.distributions as distributions
import torch.nn.functional as F

# Logger handle
msglogger = None

def adjust_learning_rate(optimizer, epoch):
    update_list = [80,120,145]#[50,100,140]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def main():
    global msglogger

    # Parse arguments
    prsr = parser.getParser()
    distiller.knowledge_distillation.add_distillation_args(prsr, ALL_MODEL_NAMES, True)
    args = prsr.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(sys.argv, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    best_epochs = [distiller.MutableNamedTuple({'epoch': 0, 'top1': 0, 'sparsity': 0})
                   for i in range(args.num_best_scores)]

    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        if args.workers > 1:
            msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
            exit(1)
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        cudnn.deterministic = True
    else:
        # This issue: https://github.com/pytorch/pytorch/issues/3659
        # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
        # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
        cudnn.benchmark = True

    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
                exit(1)
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                    .format(dev_id, available_gpus))
                    exit(1)
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    args.num_classes = 10 if args.dataset == 'cifar10' else 1000

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(model, chkpt_file=args.resume)
        model.to(args.device)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#nesterov= True

    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    # define the binarization operator
    bin_op = util_bin.BinOp(model)
    print("define binop")
    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1], args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args,bin_op, compression_scheduler)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    args.kd_policy = None
    for epoch in range(start_epoch, start_epoch + args.epochs):   
        #Adjust learning rate
        adjust_learning_rate(optimizer, epoch)
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args, bin_op=bin_op)
            #distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, bin_op,epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        is_best = top1 > best_epochs[-1].top1
        if top1 > best_epochs[0].top1:
            best_epochs[0].epoch = epoch
            best_epochs[0].top1 = top1
            # Keep best_epochs sorted such that best_epochs[0] is the lowest top1 in the best_epochs list
            best_epochs = sorted(best_epochs, key=lambda score: score.top1)
        for score in reversed(best_epochs):
            if score.top1 > 0:
                msglogger.info('==> Best Top1: %.3f on Epoch: %d', score.top1, score.epoch)
        apputils.save_checkpoint(epoch, args.arch, model, optimizer, compression_scheduler,
                                 best_epochs[-1].top1, is_best, args.name, msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args, bin_op=bin_op)


OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args, bin_op):
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit
    # So exiterrors is analogous to classerr for the non-Early Exit case
    if args.earlyexit_lossweights:
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # process the weights including binarization
        bin_op.binarization()
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if not hasattr(args, 'kd_policy') or args.kd_policy is None:
            output = model(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if not args.earlyexit_lossweights:
            loss = criterion(output, target)
            # Measure accuracy and record loss
            classerr.add(output.data, target)
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target, criterion, args)
            #TODO
            #loss_all = earlyexit_loss(output, target, criterion, args)
            #loss = loss_all[args.num_exits-1]

        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        #TODO:backpropagate the gradient of the early exit
        #-----
        #for i in range(args.num_exits-1):#losses of early exits
        #    loss_all[i].backward(retain_graph=True)  
        #loss_all[args.num_exits-1].backward()  
        #-----
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                errs['Top1'] = classerr.value(1)
                errs['Top5'] = classerr.value(5)
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                    errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()


def validate(val_loader, model, criterion, loggers, args, bin_op,epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, bin_op=bin_op,epoch=epoch)


def test(test_loader, model, criterion, loggers, activations_collectors, args, bin_op):
    """Model Test"""
    msglogger.info('--- test ---------------------')

    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args, bin_op=bin_op)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses


def _validate(data_loader, model, criterion, loggers, args, bin_op,epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if args.earlyexit_thresholds:
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    bin_op.binarization()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            output = model(inputs)

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                if not args.earlyexit_thresholds:
                    stats = ('',
                            OrderedDict([('Loss', losses['objective_loss'].mean),
                                         ('Top1', classerr.value(1)),
                                         ('Top5', classerr.value(5))]))
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                        # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                        # accordingly and we will not print for an exit error when that exit is never taken.
                        if args.exit_taken[exitnum]:
                            t1 = 'Top1_exit' + str(exitnum)
                            t5 = 'Top5_exit' + str(exitnum)
                            stats_dict[t1] = args.exiterrors[exitnum].value(1)
                            stats_dict[t5] = args.exiterrors[exitnum].value(5)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    #restore weights from binary
    bin_op.restore()
    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


def earlyexit_loss(output, target, criterion, args):
    loss = 0
    sum_lossweights = 0
    for exitnum in range(args.num_exits-1):
        #TODO:separate the training of early exits and cnn
        #loss.append(criterion(output[exitnum], target))
        loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, target)
    # handle final exit
    #loss.append(criterion(output[args.num_exits-1], target))
    loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].data, target)
    return loss


def earlyexit_validate_loss(output, target, criterion, args):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    this_batch_size = target.size()[0]
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)

    for exitnum in range(args.num_exits):
        # TODO:calculate losses at each sample separately in the minibatch.
        #args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        args.loss_exits[exitnum] = distributions.Categorical(probs=F.softmax(output[exitnum])).entropy()
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits - 1):
            if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break                    # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                         torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1


def earlyexit_validate_stats(args):
    # Print some interesting summary stats for number of data points that could exit early
    top1k_stats = [0] * args.num_exits
    top5k_stats = [0] * args.num_exits
    losses_exits_stats = [0] * args.num_exits
    sum_exit_stats = 0
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            sum_exit_stats += args.exit_taken[exitnum]
            msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
            top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
            top5k_stats[exitnum] += args.exiterrors[exitnum].value(5)
            losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    total_top1 = 0
    total_top5 = 0
    for exitnum in range(args.num_exits):
        total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum, top1k_stats[exitnum], top5k_stats[exitnum])
    msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f", total_top1, total_top5)
    return(total_top1, total_top5, losses_exits_stats)


def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args,bin_op, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        quantizer = quantization.PostTrainLinearQuantizer(model, args.qe_bits_acts, args.qe_bits_wts,
                                                          args.qe_bits_accum, args.qe_mode, args.qe_clip_acts,
                                                          args.qe_no_clip_layers, args.qe_per_channel)
        quantizer.prepare_model()
        model.to(args.device)

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args, bin_op=bin_op)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, best_top1=top1, scheduler=scheduler,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir)


def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        apputils.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        apputils.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')



def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    distiller.utils.assign_layer_fq_names(model)

    genCollectors = lambda: missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.to_xlsx(workbook)


def check_pytorch_version():
    if torch.__version__ < '0.4.0':
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 0.4.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 0.4.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


if __name__ == '__main__':
    try:
        check_pytorch_version()
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
