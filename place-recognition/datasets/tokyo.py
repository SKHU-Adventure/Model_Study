import importlib
import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np
from setup import config
from netvlad import NetVLAD
from netvlad import EmbedNet
from netvlad import TripletNet
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#Data 부분
#prof께서 작성해주신 Code에서 살짝 수정
train_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.train_data_path)#dataset.py에 있는 CustomDataset 가져오면서 Setup에 있는 config import해서 사용
test_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.test_data_path)

train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size)
#anc, pos, neg = next(iter(train_loader)) 이런식으로 꺼내 쓰면 됨

#model 선언 부분
model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model.to(device) # GPU mode 
net_vlad = NetVLAD(num_clusters=21, dim=256, alpha=1.0)
embednet = EmbedNet(model, net_vlad).to(device)
triplet = TripletNet(embednet).to(device)
criterion = nn.TripletMarginLoss(margin=0.1, p=2)#가까운건 더 가깝게 먼건 더 멀게 Triplet Loss
n_epoch=0
lr = 1e-4

def vlad_train(epochs, learning_rate):
    optimizer = optim.Adam(triplet.parameters(), lr=learning_rate)  # 최적화 알고리즘 설정
    
    for epoch in range(1, epochs + 1):  # epoch는 1부터 시작
        train_loss = train_epochs(optimizer)
        print(f'Epoch {epoch}, Training loss: {train_loss:.4f}')
        torch.save(triplet.state_dict(), f'/home/student1/캡스톤/place_tokyo/checkpoint_{epoch}.pth')
    
    test_loss = test_epochs()
    print(f'Epoch {epoch}, Testing loss: {test_loss:.4f}')

def train_epochs(optimizer):
    running_loss = 0.0
    triplet.train()  # 모델을 학습 모드로 설정
    
    for i, (images, pos, neg) in enumerate(train_loader):
        images, pos, neg = images.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        features, pos_fit, neg_fit = triplet(images, pos, neg)
        loss = criterion(features, pos_fit, neg_fit)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)  # 평균 훈련 손실 반환

def test_epochs():
    running_loss = 0.0
    triplet.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        for i, (images, pos, neg) in enumerate(test_loader):
            images, pos, neg = images.to(device), pos.to(device), neg.to(device)
            features, pos_fit, neg_fit = triplet(images, pos, neg)
            loss = criterion(features, pos_fit, neg_fit)
            running_loss += loss.item()
    
    return running_loss / len(test_loader)  # 평균 테스트 손실 반환

vlad_train(10, lr)
