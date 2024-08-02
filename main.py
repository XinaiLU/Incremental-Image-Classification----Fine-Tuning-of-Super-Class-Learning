# python main.py

# import imp
# from operator import imod
import train
import prediction
import torch
import hyperparameters as HP
from dataprocessor.datareader import get_ori_dataloader
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
from PIL import ImageEnhance


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#label_to_idx, train_data_loader_dict, test_loader = get_101_data_split_by_macro_label(is_enumerate=False)
#idx_to_label = dict(zip(label_to_idx.values(), label_to_idx.keys()))




#train.train_by_gathering_same_label_data_in_one_batch(net, label_to_idx, train_data_loader_dict, test_loader)
#train.train_by_allocate_different_label_in_one_batch(net, label_to_idx, train_data_loader_dict, test_loader)
'''
if HP.contrastive:
    train.train_con(net,trainloader,testloader)
else:
    train.train_raw(net,trainloader,testloader)
'''

print('Dataset: ', HP.data_set)
backbone = HP.backbone
print(f'Backbone: {backbone}')

trainloader, testloader = getData(HP.data_set)
ori_trainloader,ori_testloader = get_ori_dataloader()


if HP.attention:
    net = TransformerContrastive()
else:
    #net = ResNet50(category_num=HP.cls_num)
    net = get_backbone(HP.backbone)
    
net = net.cuda()
net = torch.nn.parallel.DataParallel(net, device_ids=[0,1])

# pretrain_path = './pretrained/pretrained_resnet101.pt'
# state_dict = torch.load(pretrain_path)
# net.load_state_dict(state_dict)

# train.train(net,trainloader,testloader,HP.contrastive,ori_trainloader)
# train.train(net,trainloader,testloader,HP.contrastive)
# train.train_incremental(net, ori_trainloader, trainloader)
train.train_incremental_distill(net, ori_trainloader, trainloader)

#1.只保存训练好的模型各层参数
torch.save(net.state_dict(), './phase1/net_orig1.pt')    #--net即定义的网络模型
 
#2.将模型结构与训练好的模型各层参数一起保存
torch.save(net, './phase1/net_orig2.pt')                 #--net即定义的网络模型
 
 
# 调用生成预测结果的函数
output_file = 'submission.csv'
prediction.generate_predictions(net, testloader, output_file)