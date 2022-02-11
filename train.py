import os.path

from CropModels import *
from CropDataset import Apple_leaf_dataset, normalize_torch, normalize_05, normalize_dataset, preprocess, \
    preprocess_hflip, \
    preprocess_with_augmentation
import pandas as pd
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast
from torch.autograd import Variable
from torch.cuda.amp import GradScaler

import datetime
from utils import RunningMean, split_dataset_pytorch
import utils
from tqdm import tqdm

random_seed = 42
torch.manual_seed(random_seed)
scaler = GradScaler()

torch.backends.cudnn.benchmark = True

model_dict = {
    'resnet18': resnet18_finetune,
    'resnet34': resnet34_finetune,
    'resnet50': resnet50_finetune,
    'resnet101': resnet101_finetune,
    'resnet152': resnet152_finetune,
    'densenet121': densenet121_finetune,
    'densenet161': densenet161_finetune,
    'densenet201': densenet201_finetune,
    'xception': xception_finetune,
    'inceotionv4': inceptionv4_finetune,
    'inceptionv2': inceptionresnetv2_finetune,
    'nasnet': nasnet_finetune,
    'nasnetmobile': nasnetmobile,
    'mobilenet': mobilenetv3_finetune,
    'googlenet': googlenetv2_finetune,
    'shufflenet': shufflenetv2_finetune
}


def train(epochNum, model_name='resnet18'):
    # 初始化tensorboad
    date = str(datetime.date.today())
    # 创建 /log/日期/InceptionResnet的组织形式  不同模型需要修改不同名称
    writer = SummaryWriter('./log/' + date + '/' + model_name + '/')
    #
    NB_CLASS = 5
    BATCH_SIZE = 128
    IMAGE_SIZE = 224  # 不同模型修改不同的Size
    IMAGE_ROOT = './apple_leaf_disease'
    validation_split = .2

    dataset = Apple_leaf_dataset(img_root=IMAGE_ROOT, transform=preprocess(normalize=normalize_torch,
                                                                           image_size=IMAGE_SIZE))

    train_dataloader, val_dataloader = split_dataset_pytorch(dataset, batch_size=BATCH_SIZE, validation_split=validation_split, random_seed=random_seed)

    model = model_dict[model_name](num_classes=NB_CLASS).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    lx, px = utils.predict(model, val_dataloader)
    min_loss = criterion(px, lx).item()
    print('min_loss is :%f' % (min_loss))
    min_acc = 0.80
    patience = 0
    # momentum = 0.0
    lr = 1e-4
    optimizer_warming = torch.optim.Adam(model.fresh_params(), lr=lr, amsgrad=True, weight_decay=1e-4)
    optimizer_training = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_warming = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_training, milestones=[20, 30, 40], gamma=0.1)
    for epoch in range(epochNum):
        print('Epoch {}/{}'.format(epoch, epochNum - 1))
        print('-' * 10)

        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('save_model/' + model_name + "/loss_best/" + date + '_loss_best.pth')['state_dict'])
            lr = lr / 10
            print('loss has increased lr divide 10 lr now is :%f' % (lr))
        if epoch < 10:  # 第一轮首先训练全连接层
            optimizer = optimizer_warming
            scheduler = None
        else:
            optimizer = optimizer_training
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[17, 27, 27], gamma=0.1)

        running_loss = RunningMean()
        running_corrects = RunningMean()

        train_bar = tqdm(train_dataloader)
        train_bar.desc = "train_step"
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            model.train(True)
            optimizer.zero_grad()
            with autocast():
                n_batchsize = inputs.size(0)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o, labels)) for o in outputs)
                else:
                    loss = criterion(outputs, labels)
            running_loss.update(loss.item(), 1)
            running_corrects.update(torch.sum(preds == labels.data).data, n_batchsize)
            # 梯度放大
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        #
        # val
        #
        lx, px = utils.predict(model, val_dataloader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.item()
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        writer.add_scalar('Val/Acc', accuracy, int((epoch + 1) * len(train_dataloader)))
        writer.add_scalar('Val/Loss', log_loss, int((epoch + 1) * len(train_dataloader)))
        print('[epoch:%d]: val_loss:%f,val_acc:%f,' % (epoch, log_loss, accuracy))

        #
        # save model
        #
        if log_loss < min_loss:
            if not os.path.isdir('./save_model/' + model_name + '/loss_best'):
                os.makedirs('./save_model/' + model_name + '/loss_best')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': log_loss,
                'val_correct': accuracy}, 'save_model/' + model_name + '/loss_best/' + date + '_loss_best.pth')
            patience = 0
            min_loss = log_loss
            print('save new model loss,now loss is ', min_loss)
        else:
            patience += 1
        if accuracy > min_acc:
            if not os.path.isdir('./save_model/' + model_name + '/acc_best'):
                os.makedirs('./save_model/' + model_name + '/acc_best')
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_loss': log_loss,
                        'val_correct': accuracy}, 'save_model/' + model_name + '/acc_best/' + date + '_acc_best.pth')
            min_acc = accuracy
            print('save new model acc,now acc is ', min_acc)


if __name__ == '__main__':
    # model_list = ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet201',
    #               'xception', 'inceotionv4', 'inceptionv2', 'nasnet', 'nasnetmobile', 'mobilenet', 'googlenet',
    #               'shufflenet']
    # for model_name in model_list:
    #     train(100, model_name)
    train(100, "googlenet")
