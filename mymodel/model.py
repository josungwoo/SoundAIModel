"""두 개의 합성곱 층으로 소리의 무늬를 배우는 작은 CNN입니다.

MobileNetV2나 ImageNet 가중치를 사용하지 않고 모든 가중치를 처음부터 학습합니다.
"""

from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()  # PyTorch가 안쪽 층과 가중치를 관리하게 하는 준비입니다.
        self.layers = nn.Sequential(
            # 소리 그림 1장에서 서로 다른 무늬를 계산하는 16개 특징 지도를 만듭니다.
            # kernel_size=3: 주변 3×3칸을 봅니다. padding=1: 테두리를 채워 가로·세로를 유지합니다.

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),          # 음수를 0으로 바꿔, 여러 층이 복잡한 무늬를 표현하게 합니다.
            nn.MaxPool2d(2),    # 2×2칸의 최댓값을 남겨 가로·세로를 대략 절반으로 줄입니다.

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            # 32개 지도 각각을 평균 숫자 하나로 줄입니다. 출력층 크기를 맞추기 쉬워집니다.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),       # [배치,32,1,1] → [배치,32]로 펼칩니다.
            nn.Linear(32, 3),   # 마지막으로 세 종류의 원시 점수(logit)를 계산합니다.
        )

    def forward(self, features):
        # model(features)를 호출하면 실행됩니다. 출력은 [배치 수,3]입니다.
        # 여기서 softmax를 하지 않습니다. 학습의 CrossEntropyLoss가 필요한 계산을 합니다.
        return self.layers(features)

