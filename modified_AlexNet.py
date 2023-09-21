import torch
import torch.nn as nn
from torchvision.models import alexnet
from torch.utils.data import DataLoader

# 알렉스넷 모델 불러오기 (pre-trained 가중치 포함)
num_classes = 2
model = alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)  # 마지막 fully connected layer 수정

#==================================================python
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# 정답 이미지와 오답 이미지 경로 설정
correct_images_path = "path/to/correct/images"                  ##경로설정
incorrect_images_path = "path/to/incorrect/images"              ##경로설정

# 정답 이미지 데이터셋 생성 (레이블: 1)
correct_dataset = ImageFolder(correct_images_path, transform=ToTensor())
correct_labels = torch.ones(len(correct_dataset))

# 오답 이미지 데이터셋 생성 (레이블: 0)
incorrect_dataset = ImageFolder(incorrect_images_path, transform=ToTensor())
incorrect_labels = torch.zeros(len(incorrect_dataset))

# 정답과 오답 데이터 및 레이블 병합
images = torch.cat([correct_dataset[i][0] for i in range(len(correct_dataset))] +
                   [incorrect_dataset[i][0] for i in range(len(incorrect_dataset))])
labels = torch.cat([correct_labels, incorrect_labels])

# 병합된 데이터로 Dataset 클래스 생성
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        return image, label

# 데이터셋과 데이터로더 설정
dataset = YourDataset(images, labels)
#==================================================

dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 반복 실행 (epoch 수: 20)
for epoch in range(20):
    total_correct = 0
    total_samples = 0
    
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 정확도 계산
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        # 매 batch에서 마지막 iteration의 loss와 정확도 출력
        if (i + 1) == len(dataloader):
            print(f"Epoch [{epoch+1}/20], Last Iteration: {i+1}, Loss: {loss.item():.4f}, Accuracy: {100*(total_correct/total_samples):.2f}%")

"""
#해당 코드는 그냥 출력한다. 정확도를 print하지 않고
#위의 코드가 오류가 난다면 바꿔서 진행해보면 될 것이다.

# 학습 반복 실행 (epoch 수: 20)
for epoch in range(20):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
"""
# 학습된 가중치 저장
torch.save(model.state_dict(), 'alexnet_weights.pth')
