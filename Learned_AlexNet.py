import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import alexnet

num_classes = 2

# 알렉스넷 모델 생성 및 가중치 로드하기
model = alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes)  # num_classes는 분류할 클래스의 개수로 지정해주어야 함.
model.load_state_dict(torch.load('alexnet_weights.pth'))

# 이미지 전처리 및 분류 함수 정의 (이미지 텐서 크기: [1, 1, 1024, 1024])
def classify_image(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = F.softmax(output[0], dim=0).numpy() * 100
    
    return probabilities

#카메라를 통해서 이미지를 받아오고, 전처리를 통해 1024 x 1024짜리로 바꿔야한다.
"""
# 예시 이미지를 위한 텐서 생성 및 분류 수행하기
example_image_path = 'example_image.jpg'
example_image_tensor = preprocess_and_transform(example_image_path)   # 전처리 및 변환 작업 필요함.
probabilities_result = classify_image(example_image_tensor)

print(probabilities_result)
"""