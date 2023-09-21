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
#일단 하단의 코드는 정확하지 않다. 경로설정 및 사이즈 맞춰주는 과정이 필요하다.
import cv2      #pip install opencv-python //this OpenCV install is needed
import numpy as np

# 이미지 로드 및 padding 적용
image_path = "path/to/image.jpg"  # 실제 이미지 파일의 경로로 변경해주세요.
image = cv2.imread(image_path)

desired_height = 3072
desired_width = 2048

padded_image = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)
padded_image[:image.shape[0], :image.shape[1]] = image

# Maxpooling을 통한 크기 조정
target_size = (1024, 1024)
resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)

# 결과 확인을 위한 이미지 출력 (선택 사항)
cv2.imshow("Preprocessed Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
# 예시 이미지를 위한 텐서 생성 및 분류 수행하기
example_image_path = 'example_image.jpg'
example_image_tensor = preprocess_and_transform(example_image_path)   # 전처리 및 변환 작업 필요함.
probabilities_result = classify_image(example_image_tensor)

print(probabilities_result)
"""