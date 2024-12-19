import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from torch.nn.functional import cosine_similarity

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


class CustomDataset(Dataset):
    def __init__(self, directory,filename, transform=None):
        self.directory = directory
        self.transform = transform
        self.images1, self.images2, self.labels = self.load_images(filename)

    def load_images(self,filename):
        images1 = []
        images2 = []
        labels = []
        with open (filename,'r') as datafile:
            # lines = datafile.readline()
            for line in datafile:
                element = line.split()
                images1.append(os.path.join(self.directory,element[0]))
                images2.append(os.path.join(self.directory,element[1]))
                labels.append(int(element[2]))


        # 遍历目录中的所有文件

        # i =0
        # for i in range(20000):
        #     filename1 = tmplist[random.randint(0, len(tmplist) - 1)]
        #     filename2 = tmplist[random.randint(0, len(tmplist) - 1)]
        #     flab1 = filename1.split('_')[0]
        #     flab2 = filename2.split('_')[0]
        #     if i%2 and flab1!=flab2:
        #         filename2=filename1
        #         flab2=flab1
        #     images1.append(os.path.join(self.directory, filename1))
        #     images2.append(os.path.join(self.directory, filename2))

            # labels.append(int(flab1 == flab2))

        return images1, images2, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path1 = self.images1[idx]
        image_path2 = self.images2[idx]
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label = self.labels[idx]
        return image1, image2, label


# 定义 Siamese 网络结构
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 使用预训练的 MobileNet 作为特征提取器
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # 移除原有的分类层

        # # 添加额外的分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # 特征提取
        output1 = self.mobilenet(input1)
        output2 = self.mobilenet(input2)

        # 特征组合
        # combined = torch.cat((output1, output2), dim=1)
        # output = self.classifier(combined)
        return output1,output2


# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == '__main__':
    # 实例化模型和损失函数
    model = SiameseNetwork().to(device)
    criterion =ContrastiveLoss().to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 数据加载器（需要根据你的数据集进行定制）
    transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=Image.NEAREST),  # smaller side resized
        transforms.ToTensor(),
    ])
    root = r'D:\arcfacetrain\arcface\ROI\ROI'
    traindataset = CustomDataset(
        directory=root,
        filename='trainfile.txt',
        transform=transform)
    validdataset = CustomDataset(
        directory=root,
        filename='validfile.txt',
        transform=transform)
    testdataset = CustomDataset(
        directory=root,
        filename='testfile.txt',
        transform=transform)
    train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)  # 训练数据加载器
    test_loader = DataLoader(validdataset, batch_size=32, shuffle=True)
    num_epochs = 20
    bestacc = 0
    # 训练循环
    print("start")
    bstacc =0
    for epoch in range(num_epochs):

        model.train()
        # 使用tqdm创建进度条
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Loss: 0')

        for i, (img1, img2, label) in enumerate(train_loader_tqdm):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            # 前向传播
            output1,output2 = model(img1, img2)
            loss = criterion(output1,output2, label.float())


            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条的描述
            train_loader_tqdm.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

        model_save_path = f"./mobilenet/curr_CL_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)

        model.eval()
        valid_loss = 0
        valid_loader_tqdm = tqdm(test_loader, desc='Validation', leave=False)
        acc = 0

        with torch.no_grad():  # 不计算梯度
            for img1, img2, label in valid_loader_tqdm:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1,output2 = model(img1, img2)
                loss = criterion(output1,output2, label.float())
                valid_loss += loss.item()
                # 更新进度条的描述
                valid_loader_tqdm.set_description(f'Validation Loss: {loss.item():.4f}')
                for i in range(32):
                    o1 = np.linalg.norm(output1[i].cpu())
                    o2 = np.linalg.norm(output2[i].cpu())
                    rslt = np.dot(output1[i].cpu(),output2[i].cpu())/ (np.linalg.norm(output1[i].detach().cpu().numpy()) * np.linalg.norm(output2[i].detach().cpu().numpy()))
                    if rslt > 0.5:
                        rslt = 1
                    if rslt == int(label[i]):
                        acc += 1


        valid_loss /= len(validdataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss:.4f},Accuracy :{acc/1600}')
        if acc>bstacc:
            bstacc=acc
            print(f"current best pth update,now is {model_save_path}")



        # model.eval()
        # total = 0
        # correct = 0
        # with torch.no_grad():  # 不需要计算梯度
        #     # 使用tqdm创建进度条
        #     test_loader_tqdm = tqdm(test_loader, desc='Testing')
        #
        #     for img1, img2, labels in test_loader_tqdm:
        #         img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        #         # 前向传播
        #         num= model(img1, img2)
        #
        #         # 计算距离
        #         euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        #
        #         # 根据距离和阈值判断是否为同一类
        #         # 注意：你需要根据你的任务和数据调整阈值
        #         threshold = 1.0
        #         predictions = (euclidean_distance < threshold).type(torch.int)
        #
        #         total += labels.size(0)
        #         correct += (predictions == labels).sum().item()
        #
        #         # 可选：更新进度条描述
        #         test_loader_tqdm.set_description(f'Testing, Acc: {100 * correct / total:.2f}%')
        #
        # # 计算准确率
        # accuracy = 100 * correct / total
