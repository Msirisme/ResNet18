import copy
import time

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18, Residual
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

class Mydataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.transform=transform
        self.dataset=image_folder_dataset
    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.dataset)

def train_val_data_process():
    ROOT_TRAIN = r'data\train'
    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155, 0.06216329, 0.05930814])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image_folder_dataset = ImageFolder(ROOT_TRAIN)
    train_data = Mydataset(image_folder_dataset,transform=train_transform)
    train_data, val_data = random_split(train_data,
                                        [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader =  Data.DataLoader(train_data,
                                        batch_size=32,
                                        num_workers=0,
                                        shuffle=True)
    val_dataloader = Data.DataLoader(val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=0)
    return train_dataloader,val_dataloader





def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        train_num = 0
        test_num = 0
        # tqdm progress bar for training
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for step, (b_x, b_y) in enumerate(tqdm(train_dataloader, desc='Training')):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # tqdm progress bar for validation
        for step, (b_x, b_y) in enumerate(tqdm(val_dataloader, desc='Validation')):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            test_loss += loss.item() * b_x.size(0)
            test_acc += torch.sum(pre_lab == b_y.data)
            test_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc.double().item() / train_num)
        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_acc.double().item() / test_num)

        print(f"Epoch {epoch + 1}: Train loss: {train_loss_all[-1]:.4f}, Train acc: {train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch + 1}: Val loss: {test_loss_all[-1]:.4f}, Val acc: {test_acc_all[-1]:.4f}")

        if test_acc_all[-1] > best_acc:
            best_acc = test_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "C:/Users/86131/Desktop/DL/源码/ResNet18-1/best_model.pth")

    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": test_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": test_acc_all,
    })

    return train_process
def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 加载需要的模型
    ResNet18 = ResNet18(Residual)
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(ResNet18, train_data, val_data, num_epochs=50)
    matplot_acc_loss(train_process)
