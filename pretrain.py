# python main.py

# import imp
# from operator import imod
import train
import prediction
import torch
import hyperparameters as HP
# from dataprocessor.datacomposer import get_CIFAR10_re_dataset
# from dataprocessor.datacomposer import get_101_OC_data
# from dataprocessor.datacomposer import get_101_data_split_by_macro_label
from dataprocessor.datacomposer import getData
# from dataprocessor.datacomposer import get_CIFAR100_data_loader
from model.TC2 import TransformerContrastive
from model.TC2 import get_backbone
from model.resnet50 import ResNet50
from model.CNN import CNN_NET
import torch
from tqdm import tqdm
import os
from model.target import get_target
from model.TC2 import get_emb_len

from cProfile import label
from cgi import test
import imp
from locale import normalize
from random import shuffle
import random
from re import X
from dataprocessor.datareader import get_CIFAR10_dataloader
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Resize
import hyperparameters as HP
import os
from torchvision import transforms
from PIL import Image
from utils import single_channel_to_3_channel
from utils import mini_imagenet_Dataset
from utils import ut_zap50k_Dataset
from utils import CIFAR100Pair
from utils import train_transform


# python train.py

import imp
from operator import mod
from os import remove
from re import X
#from tkinter import Y
from model.CNN import CNN_NET
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import numpy as np
from dataprocessor.datacomposer import get_sample_data
import hyperparameters as HP
from model.resnet50 import ResNet50
from tensorboardX import SummaryWriter
from model.TC2 import TransformerContrastive
from model.TC2 import get_emb_len
import random

from utils import eval_with_wilds, get_iter_dict
from utils import calc_accuracy
from utils import writer
from utils import freeze_by_names
from utils import draw
from utils import tag_name
from utils import eval

from model.contrastive import ContrastiveLoss
from model.TargetLoss import TargetLoss
from model.target import get_target
from model.target import Target


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_cifar100_4_pretrain():
    #train_data = CIFAR100Pair('./data/re_cifar100/7_categories/train.pt',transform=train_transform)
    #train_loader = DataLoader(train_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test_data = CIFAR100Pair('./data/re_cifar100/7_categories/test.pt',transform=test_transform)
    #test_loader = DataLoader(test_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Resize(size=(64,64))

    training_set = torch.load('./origine_data/train.pt')
    print('length of pre_training set: ', len(training_set))
    Xs = []
    Ys = []
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./aug_data/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = False, num_workers = 2, drop_last=False)

    #print(training_data_tensors.size(), training_label_tensors.size())
    #print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def pretrain(net, trainloader, testloader, is_con):
    # to train
    '''
    net (nn.Module) the model for training
    trainloader (Dataloader) the dataloader for training data
    testloader (Dataloader) the dataloader for test data
    is_con: whether to add contrastive loss
    '''
    ifdraw = False
    
    EPOCH = 8
    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    best_model = net
    best_acc = 0.0

    cls_loss_func = torch.nn.CrossEntropyLoss()
    contra_loss_func = ContrastiveLoss()
    contra_loss_func = contra_loss_func.cuda()
    target_loss_func = TargetLoss()
    target_loss_func = target_loss_func.cuda()

    batch_num = HP.train_set_size / HP.batch_size
    if ifdraw:
        data_tensor,label_tensor = get_sample_data(HP.data_set)

    t = Target()
    #t.generate_target(num=HP.cls_num,dim=get_emb_len(HP.backbone))
    t.generate_target(num=HP.cls_num,dim=128)

    for epoch in range(EPOCH):

        if ifdraw:
            with torch.no_grad():
                representation,_ = net(data_tensor.cuda())
                draw(X=representation.cpu(),Y=label_tensor,msg='Training,'+'epoch='+str(epoch))
        
        epoch_loss = 0.0
        running_con_loss = 0.0
        running_cls_loss = 0.0
        running_tar_loss = 0.0
        total_num = 0
        train_bar = tqdm(trainloader)
        #for step, (b_x,b_y)in enumerate(trainloader):
        for b_x,b_y in train_bar:
            #target = get_target(num=HP.cls_num,dim=get_emb_len(HP.backbone))
            #target = target.cuda()
            target = t.get_target()
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            #print(b_x.size(),b_y.size())
            representation, cls_rtn = net(b_x)

            cls_loss = cls_loss_func(cls_rtn, b_y)
            contra_loss = contra_loss_func(representation, b_y)
            target_loss = target_loss_func(representation, b_y, target)
            #target_loss = 0
            if is_con:
                loss = HP.alpha * contra_loss + (1-HP.alpha) * cls_loss
            else:
                loss = cls_loss

            if HP.TARGET:
                loss += HP.lmd * target_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            epoch_loss += loss.item()
            running_con_loss += contra_loss
            running_cls_loss += cls_loss
            running_tar_loss += target_loss
            total_num += HP.batch_size

            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch+1,EPOCH,epoch_loss/total_num))

        epoch_loss /= batch_num
        running_con_loss /= batch_num
        running_cls_loss /= batch_num
        running_tar_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)
        accuracy = calc_accuracy(net, testloader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = net
            if ifdraw:
                with torch.no_grad():
                    representation,_ = best_model(data_tensor.cuda())
                    draw(X=representation.cpu(),Y=label_tensor,msg='Best-Training')
                
        writer.add_scalar('accuracy', accuracy, global_step = epoch)
        if is_con:
            print('[epoch %d] total_loss = %.3f   contra_loss = %.3f   cls_loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,running_con_loss,running_cls_loss,accuracy))
        else:
            print('[epoch %d] loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,accuracy))
        
        if HP.TARGET:
            print(f'target loss = {running_tar_loss}')
        epoch_loss = 0
        recall_mean, precision_mean, F1_mean, rtn_accuracy = eval(net,testloader)
        writer.add_scalar('recall', recall_mean, global_step = epoch)
        writer.add_scalar('precision', precision_mean, global_step = epoch)
        writer.add_scalar('F1', F1_mean, global_step = epoch)
        writer.add_scalar('acc2', rtn_accuracy, global_step = epoch)
        #print(rtn_recall_mean, rtn_precision, rtn_F1_mean, rtn_accuracy)
        torch.save({'model': net.state_dict()}, './backup/'+HP.outname+'-current.pth')
    
        
    torch.save({'model': best_model.state_dict()}, './backup/'+tag_name+'.pth')

    print('Finished Training')

pretrainloader, pretestloader = get_cifar100_4_pretrain()

if HP.attention:
    net = TransformerContrastive()
else:
    #net = ResNet50(category_num=HP.cls_num)
    net = get_backbone(HP.backbone)
    
net = net.cuda()
net = torch.nn.parallel.DataParallel(net, device_ids=[0,1])

pretrain(net,pretrainloader,pretestloader,HP.contrastive)

# 保存预训练后的模型参数
pretrain_path = './pretrained/pretrained_resnet101.pt'
torch.save(net.state_dict(), pretrain_path)