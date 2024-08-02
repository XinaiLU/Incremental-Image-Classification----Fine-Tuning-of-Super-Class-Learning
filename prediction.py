import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 生成预测结果的函数
def generate_predictions(net, testloader, output_file):
    net.eval()  # 设置模型为评估模式
    predictions = []
    with torch.no_grad():
        for b_x, _ in tqdm(testloader, desc='Generating Predictions'):
            # print(b_x)
            b_x = b_x.cuda()
            representation, cls_rtn = net(b_x)
            _, predicted = torch.max(cls_rtn, 1)
            predictions.extend(predicted.cpu().numpy())
    print(len(predictions))
    
    # 将预测结果写入文件
    with open(output_file, 'w') as f:
        f.write("id,label\n")  # 列名
        for i,pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")