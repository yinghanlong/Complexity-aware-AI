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
import pickle
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
#import matplotlib.pyplot as plt
from models.imagenet.resnet_pretrain import BasicBlock

from ptflops import get_model_complexity_info
# Logger handle
msglogger = None

confusion_matrix = torch.zeros(100, 100)
def adjust_learning_rate(optimizer, epoch):
    #update_list = [180,220]
    update_list = [30,60]#[130]#[30,60,80]#[60,120,160]#[230,290,330]#[60,120,160]#[200,250,300]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return
'''
def plot_confusion_matrix(cm, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
'''

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
    #enable cloud network or not
    args.cloud = False#True
    #use concatenation for extension block or sum
    args.concat = True
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
    #TODO: change dataset
    args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    args.num_classes = 10 if args.dataset == 'cifar10' else 1000
    args.num_hard = 500

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    #load hard class dictionary
    
    with open('imagenet_test/hard_classes.pickle','rb') as f:
        args.hard_dict = {}
        #for i in range(500):#TODO:random class
        #    args.hard_dict[i]= i
        
        args.hard_dict =pickle.load(f)
        new_label = 0
      
        for h in args.hard_dict:
            args.hard_dict[h] = new_label
            new_label+=1
        
        print(args.hard_dict)
        args.num_hard = len(args.hard_dict)
    f.close()
    args.hard_dict_reverse = {}
    for c in args.hard_dict:
        args.hard_dict_reverse[args.hard_dict[c]] = c
    
    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    #load pretrained model
    #model.module.load_state_dict(torch.load('resnet18-5c106cde.pth'))
    if args.resume or args.evaluate:
        # Define extension block
        downsample0 = nn.Sequential(
	            nn.Conv2d(1024, 512,
	                      kernel_size=1, bias=False),
	            nn.BatchNorm2d(512),
        )
        if args.concat==True:
            model.module.conv1_exit1 = nn.Sequential(
                BasicBlock( 1024,512,downsample=downsample0),#concat two inputs, use a wider conv layer
                BasicBlock( 512,512),
                BasicBlock( 512,512),
                BasicBlock( 512,512)
            )
        else:
            model.module.conv1_exit1 = nn.Sequential(
                BasicBlock( 512,512),
                BasicBlock( 512,512),
                BasicBlock( 512,512),
                BasicBlock( 512,512)
            )
        model.module.fc_exit1 = nn.Linear(512, args.num_hard)#reduce the number of classes to only include hard classes

        #hard class corrector
        downsample1 = nn.Sequential(
                    nn.Conv2d(64, 128,
                              kernel_size=1, stride=2,bias=False),
                    nn.BatchNorm2d(128),
        )
        downsample2 = nn.Sequential(
                    nn.Conv2d(128,256,
                              kernel_size=1, stride=2,bias=False),
                    nn.BatchNorm2d(256),
        )
        downsample3 = nn.Sequential(
                    nn.Conv2d(256,512,
                              kernel_size=1, stride=2,bias=False),
                    nn.BatchNorm2d(512),
        )
        model.module.corrector = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock( 64, 64),
            BasicBlock( 64, 128, stride=2, downsample=downsample1),
            BasicBlock(128, 256, stride=2, downsample=downsample2),
            BasicBlock( 256, 512, stride=2, downsample=downsample3)
        )
    model.to(args.device)
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
        #model.module.fc = torch.nn.Linear(256,args.num_hard).to(args.device) #reduce the number of classes to only include hard classes
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(model, chkpt_file=args.resume)
        model.to(args.device)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#, nesterov= True)#nesterov ON/OFF


    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    # define the binarization operator
    bin_op = util_bin.BinOp(model)
    print("define binop")
    
    #add a two-class classifier
    '''
    params_exit = list(model.module.feature_embed1.parameters()) + list(model.module.feature_embed2.parameters()) + list(model.module.binary_classifier.parameters())
    optimizer_exit =  torch.optim.SGD(params_exit, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov= True)#nesterov ON/OFF
    
    '''
    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.

    #TODO:train with all the data
    args.validation_size=0
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, '/local/a/imagenet/imagenet2012', args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    val_loader = test_loader
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    #Based on validation accuracy
    #threshold=0.75
    args.class_accuracy= torch.zeros(args.num_classes)
    args.class_correct= torch.zeros(args.num_classes)
    args.class_cnt= torch.zeros(args.num_classes)
    

    
    if args.evaluate:
        args.blockidx = 2
        #Count MACs and number of parameters
        #macs,params = get_model_complexity_info(model,(3,224,224),as_strings=True)
        #print(macs,params)
        #return 1
        return evaluate_model(model, criterion,test_loader, pylogger, activations_collectors, args,bin_op, compression_scheduler)
    
    args.kd_policy = None
    args.blockidx = 0

    #-----------------------------#
    #***Train the second block****#
    #start_epoch = 0
    args.blockidx = 1
    
    #run results on the validation set
    #test(val_loader, model, criterion, [pylogger], activations_collectors, args=args, bin_op=bin_op)
    print(args.num_hard,args.hard_dict)
    
    #-----------------------------#
    #***Train the third block****#
    start_epoch=0
    '''
    # Define extension block
    downsample0 = nn.Sequential(
	        nn.Conv2d(1024, 512,
	                  kernel_size=1, bias=False),
	        nn.BatchNorm2d(512),
    )
    model.module.conv1_exit1 = nn.Sequential(
        BasicBlock( 1024,512,downsample=downsample0),#concat two inputs, use a wider conv layer
        BasicBlock( 512,512),
        BasicBlock( 512,512),
        BasicBlock( 512,512)
    )
    model.module.fc_exit1 = nn.Linear(512, args.num_hard)#reduce the number of classes to only include hard classes

    #hard class corrector
    downsample1 = nn.Sequential(
                nn.Conv2d(64, 128,
                          kernel_size=1, stride=2,bias=False),
                nn.BatchNorm2d(128),
    )
    downsample2 = nn.Sequential(
                nn.Conv2d(128,256,
                          kernel_size=1, stride=2,bias=False),
                nn.BatchNorm2d(256),
    )
    downsample3 = nn.Sequential(
                nn.Conv2d(256,512,
                          kernel_size=1, stride=2,bias=False),
                nn.BatchNorm2d(512),
    )
    model.module.corrector = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        BasicBlock( 64, 64),
        BasicBlock( 64, 128, stride=2, downsample=downsample1),
        BasicBlock(128, 256, stride=2, downsample=downsample2),
        BasicBlock( 256, 512, stride=2, downsample=downsample3)
    )
    model.to(args.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
   '''
    #torch.optim.Adam(model.parameters(), lr=args.lr)

    args.blockidx = 2

    #map labels
    '''
    new_label = 0
    for c in range(args.num_classes):
        if c in args.hard_dict:
            args.hard_dict[c]= new_label  # key= old label, value = new label
            new_label+=1
    '''
    
    for epoch in range(start_epoch, start_epoch + args.epochs):   
        #Adjust learning rate
        adjust_learning_rate(optimizer, epoch)
        # This is the main training loop.
        msglogger.info('\n')

        #***Calculate class-wise accuracy****
        args.class_accuracy= torch.zeros(args.num_classes)
        args.class_correct= torch.zeros(args.num_classes)
        args.class_cnt= torch.ones(args.num_classes)

        # Train for one epoch
        # Switch to train mode
        model.train()
        #TODO:no need to update gradients of CNN, only train the linear layers
        #-----
        
        print(args.hard_dict)
        ct=0
        for child in model.modules():
            ct+=1
            if ct>2 and ct<=77:#ct<=55:#ct<=83:# Layer 83: hard class detector# TODO:the num of layers to freeze depends on the num of binary layers
                #print("Fix Layer ",ct,child)
                child.eval()    #***set batch norm layers to eval mode, avoid statistics updation
                for param in child.parameters():
                    param.requires_grad = False #fix the parameters of cnn
        
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
                                 best_epochs[-1].top1, is_best, 'block3', msglogger.logdir)

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

    end = time.time()    
    
    # Switch to train mode
    #model.train()
    correct_sp =0
    trained_num = 0
    for train_step, (inputs, target) in enumerate(train_loader):
        
        # Measure data loading time
        data_time.add(time.time() - end)

        inputs, target = inputs.to(args.device), target.to(args.device)    
        #predict using edge
        #output, exit_enable = model(inputs)#output_edge, exit_enable = model(inputs)

        #Train the third block using only examples of hard classes
        targets = []
        if args.blockidx <2:
            for i in range(2):                
                targets.append(target)
        else:
            for i in range(1):                
                targets.append(target)
            #block 1 labels
            target_hard = target.clone()
            select_samples = []
            for b in range(inputs.size(0)):
                if target[b].item() in args.hard_dict:
                    select_samples.append(b)
                    target_hard[b] = torch.tensor(args.hard_dict[target[b].item()])
            targets.append(target_hard.to(args.device))
            for i in range(2):
                targets[i] = targets[i][select_samples]
            inputs = inputs[select_samples,:,:,:]

        output = model(inputs)
            #output[args.num_exits-1] = output_hard[args.num_exits-1]
            #print('hard samples:',len(select_samples))
                      
        
        
        # Train the HCD at the second block
        if args.blockidx>=2:
            target_batch = []
            for b in range(inputs.size(0)):
                #_, edge_pred = torch.topk(output[args.blockidx][b],1)
                #target_batch.append(torch.equal(edge_pred,target[b]))
                if target[b].item() in args.hard_dict:
                    target_batch.append(True)#args.class_complex[target[b]]) #easy class = 1, hard class = 0
                else:
                    target_batch.append(False)
            target_batch=torch.LongTensor(target_batch).cuda()
            #_, pred_labels = torch.topk(exit_enable,1)
        
            for b in range(inputs.size(0)):
                out_max = F.softmax(output[0][b])
                edge_confid, edge_pred = torch.topk(out_max,1)
                pred_labels_b= ~((edge_pred.item() in args.hard_dict)==False and edge_confid>0.3)
                if bool(pred_labels_b) == bool(target_batch[b]):
                    correct_sp+=1
            trained_num +=inputs.size(0)
            exit_accuracy = correct_sp/ trained_num #precision= true positives/positives
        
            #binary classifier loss
            #loss_exit =  criterion(exit_enable, target_batch)
        else:
            pred_labels =None  
        

        if not args.earlyexit_lossweights:
            loss = criterion(output, target)
            # Measure accuracy and record loss
            classerr.add(output.data, target)
        else:
            # Measure accuracy and record loss
            #loss = earlyexit_loss(output, target, criterion, args)
            #TODO
            loss_all = earlyexit_loss(output, targets, criterion, args)
            loss = loss_all[args.num_exits-1]

        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        #TODO:backpropagate the gradient of the early exit
        #-----
        #for i in range(args.num_exits-1):#losses of early exits
        #    loss_all[i].backward(retain_graph=True)  
        #loss_all[args.num_exits-1].backward()  
        #-----
        
        if args.blockidx==1:
            loss_joint =  loss #+  loss_exit
            loss_joint.backward()
        else:        
            loss.backward()
        #loss_exit.backward()
        
        
        optimizer.step()

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
            if args.blockidx>=2:
                errs['binary top1'] = exit_accuracy

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
    '''
    args.class_accuracy = torch.div(args.class_correct, args.class_cnt)
    #print('CLASS-WISE ACCURACY:',args.class_accuracy)
    #calculate class complexity by comparing with cloud accuracy
    args.class_complex= torch.gt(args.class_accuracy, 0.70)
    print('EASY CLASSES:',args.class_complex)
    print('NUM OF EASY CLASSES:', torch.sum(args.class_complex))
    '''


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
    exiterr = tnt.ClassErrorMeter(accuracy=True, topk=(1,))

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
    
    if args.cloud==True:
        model_cloud = create_model(args.pretrained, args.dataset, 'resnet101',
                         parallel=not args.load_serialized, device_ids=args.gpus)
        model_cloud.module.load_state_dict(torch.load('resnet101.pth'))
        model_cloud.to(args.device)
        model_cloud.eval()
    


    end = time.time()

    correct_sp = 0
    correct_hard = 0
    pred_hard = 1
    valid_num = 1
    correct_easy = 0
    pred_easy = 1
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():

            end = time.time()
            inputs, target = inputs.to(args.device), target.to(args.device)

            #Train the third block using only examples of hard classes
            
            #change labels
            targets = []
            if args.blockidx<2:
                for i in range(2):
                    targets.append(target)
            else:
                for i in range(1):
                    targets.append(target)
                #block 1 labels
                target_hard = target.clone()
                select_samples = []
                for b in range(inputs.size(0)):
                    if target[b].item() in args.hard_dict:
                        select_samples.append(b)
                        target_hard[b] = torch.tensor(args.hard_dict[target[b].item()])
                targets.append(target_hard.to(args.device))
                targets.append(target)
                #for i in range(2):#run only hard classes
                #    targets[i] = targets[i][select_samples]
                #inputs = inputs[select_samples,:,:,:]

            if inputs.size(0)==0:
                continue
            # compute output from model            
            #predict using edge and cloud\
            output= model(inputs)            
            # measure elapsed time
            batch_time.add(time.time() - end)
            #add cloud network
            if args.cloud==True:
                output_cloud = model_cloud(inputs)
                output.append(output_cloud)
                targets.append(target)
            
            if args.blockidx>=2:#Test the Hard class detector at second block
                #binary classifier loss
                target_batch = []
                for b in range(inputs.size(0)):
                    #target_batch.append(torch.equal(edge_pred,target[b]))
                    if target[b].item() in args.hard_dict:
                        target_batch.append(True)#args.class_complex[target[b]]) #easy class = 1, hard class = 0
                    else:
                        target_batch.append(False)
                target_batch=torch.LongTensor(target_batch).cuda()
                #loss_exit =  criterion(exit_enable, target_batch)
                pred_labels = None#target_batch #force it to be correct
                #exit_max = F.softmax(exit_enable,1)
                #pred_labels = torch.gt(exit_max[:,1],0.5)
                #_, pred_labels = torch.topk(exit_enable,1)
            
                for b in range(inputs.size(0)):
                    out_max = F.softmax(output[0][b])
                    edge_confid, edge_pred = torch.topk(out_max,1)
                    pred_labels_b= (edge_pred.item() in args.hard_dict)#~((edge_pred.item() in args.hard_dict)==False an)~((edge_pred.item() in args.hard_dict)==False and edge_confid>0.2)#TODO
                    if bool(pred_labels_b) == bool(target_batch[b]):
                        correct_sp+=1
                    if bool(pred_labels_b)== True: #hard
                        pred_hard+=1
                        if bool(target_batch[b]) == True:
                            correct_hard+=1
                    if bool(pred_labels_b)== False: #easy
                        pred_easy+=1
                        if bool(target_batch[b]) == False:
                            correct_easy+=1
                valid_num +=inputs.size(0)
                exit_accuracy = correct_sp/valid_num
            else:
                pred_labels =None

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, targets, pred_labels, criterion, args)


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
                    #record class-wise precision
                    for i in range(args.num_classes):
                        if args.class_cnt[i] == 0:
                            args.class_cnt[i]+=1
                    args.class_accuracy = torch.div(args.class_correct, args.class_cnt)

                    stats_dict['Time'] = batch_time.mean
                    if args.blockidx>=2:
                        stats_dict['binary top1'] = exit_accuracy
                        stats_dict['hard class exiting precision'] = correct_hard / (pred_hard)
                        stats_dict['easy class exiting precision'] = correct_easy / (pred_easy)
                        #easy_class_accuracy = torch.mul(args.class_accuracy,args.class_complex.int().float())
                        select_hard =[]
                        select_easy =[]
                        for c in range(args.num_classes):
                            if c in args.hard_dict:
                                select_hard.append(c)
                            else:
                                select_easy.append(c)
                        #print(args.class_accuracy[select_hard])
                        stats_dict['hard class precision'] =  100* torch.sum(args.class_accuracy[select_hard]) /  args.num_hard
                        stats_dict['easy class precision'] =  100* torch.sum(args.class_accuracy[select_easy]) /  (args.num_classes-args.num_hard)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if args.blockidx==1:#decide easy/hard classes
        '''
        #construct confusion matrix
        color_map ={}
        for i in range(args.num_classes):
            confusion_matrix[i,i] = 0 #avoid choosing itself
        confused_val, confused_to = torch.max(confusion_matrix,1)
        #print(confusion_matrix)
        #print(confused_val)
        _, sorted_idx = torch.sort(confused_val,descending=True)
        for c in range(args.num_classes):
            i = sorted_idx[c].item()
            if (i not in color_map) and (confused_to[i].item() not in color_map):
                color_map[i]= True
                color_map[confused_to[i]] = False
            elif (i not in color_map):
                color_map[i]= ~color_map[confused_to[i].item()]
            elif (confused_to[i].item() not in color_map):
                color_map[confused_to[i].item()]= ~color_map[i]
        '''
        _, args.hard_class_idx =torch.topk(-1.0 *args.class_accuracy,int(args.num_classes/2)) #take negative to reverse ordering
        args.hard_dict={}
        new_label = 0
        #for hard_idx in color_map:
        #    if color_map[hard_idx]==True:
        #        args.hard_dict[hard_idx] = new_label
        #        new_label+=1
        for hard_idx in range(len(args.hard_class_idx)):
            args.hard_dict[args.hard_class_idx[hard_idx].item()] = new_label
            new_label+=1
        args.num_hard = new_label
        
        args.hard_dict_reverse = {}
        for c in args.hard_dict:
            args.hard_dict_reverse[args.hard_dict[c]] = c
        with open(os.path.join(msglogger.logdir,'hard_classes.pickle'),'wb') as f:
            pickle.dump(args.hard_dict,f)
        f.close()
        #plot_confusion_matrix(confusion_matrix, title='Confusion matrix, CIFAR100')
        #plt.show()

    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


def earlyexit_loss(output, targets, criterion, args):
    loss = []
    sum_lossweights = 0
    #print('Num of exits:',args.num_exits)
    for exitnum in range(len(output)-1):
        #TODO:separate the training of early exits and cnn
        loss.append(criterion(output[exitnum], targets[exitnum]))
        #loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        #sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, targets[exitnum])
    # handle final exit
    loss.append(criterion(output[len(output)-1], targets[len(output)-1]))
    
    #loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[len(output)-1].add(output[len(output)-1].data, targets[len(output)-1])
    return loss


def earlyexit_validate_loss(output, targets, pred_labels, criterion, args):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    this_batch_size = targets[0].size()[0]
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)
    
    '''
    #calculate class complexity by comparing with cloud accuracy
    args.class_complex= torch.gt(args.class_accuracy, 0.60)
    print('EASY CLASSES:',args.class_complex)
    print('NUM OF EASY CLASSES:', torch.sum(args.class_complex))
    '''
    
    

    for exitnum in range(len(output)):
        #TODO: use entropy instead of cross entropy loss
        # calculate losses at each sample separately in the minibatch.
        #args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        args.loss_exits[exitnum] = distributions.Categorical(probs=F.softmax(output[exitnum])).entropy()
        
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit depending on classes or CrossEntropyLoss as confidence measure (lower is more confident)
        #Only allow early exiting if all blocks have been trained
        #***Confidence at final exit
        out_max = F.softmax(output[len(output)-1][batch_index])      
        confidence, _ = torch.topk(out_max,1)
        if args.blockidx>=2: #TODO:disabled
          for exitnum in range(0,len(output) - 1):
            #!!!exit if binary classifier allows
            #TODO: Choose the exiting criterion
            out_max = F.softmax(output[exitnum][batch_index])
            edge_confid, edge_pred = torch.topk(out_max,1)
            
            #(bool(pred_labels[batch_index])==False)
            if exitnum ==0:
                is_easy = (edge_pred.item() in args.hard_dict)==False# and edge_confid>0.2#(hard_prob< args.num_hard/args.num_classes) #(edge_pred.item() in args.hard_dict)==False and edge_confid>0.3
            else:
                is_easy = False#(args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum])
            #if the prediction at the main block is not reliable, send to the cloud directly
            if args.cloud==True:
                if (args.loss_exits[0][batch_index] > args.earlyexit_thresholds[0]):
                    break
            #Exit 0: take if predicted as easy or predicted as hard but confidence at the main block is higher
            #Exit 1: take if it has not exited at exit 0 or send to the cloud
            #if (exitnum==0 and is_easy) or exitnum==1:
            if (exitnum==0 and (is_easy or edge_confid>confidence)) or exitnum==1:

                #if torch.equal(edge_pred, targets[exitnum][batch_index]): #correct
                #    args.class_correct[edge_pred]+=1
                #args.class_cnt[edge_pred]+=1 #precision
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], targets[exitnum][batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break     
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = len(output) -1
            _, edge_pred = torch.topk(output[exitnum][batch_index],1)
            #if args.blockidx==2:
            #    edge_pred_class = args.hard_dict_reverse[edge_pred.item()]
            #else:
            edge_pred_class = edge_pred
            #if torch.equal(edge_pred, targets[exitnum][batch_index]): #correct
            #    args.class_correct[edge_pred_class]+=1
            #args.class_cnt[edge_pred_class]+=1 #precision
            args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                         torch.full([1], targets[exitnum][batch_index], dtype=torch.long))
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

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args, bin_op=bin_op)




def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        apputils.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        apputils.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)

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

