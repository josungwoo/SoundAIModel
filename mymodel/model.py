"""두 개의 합성곱 층으로 소리의 무늬를 배우는 작은 CNN입니다.

MobileNetV2나 ImageNet 가중치를 사용하지 않고 모든 가중치를 처음부터 학습합니다.
"""

from torch import nn

# 입력 합성곱 합성곱 분류
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




class BasicBlock(nn.Module):
    """합성곱 두 개와 잔차 연결을 가진 블록입니다."""

    def __init__(self, input, output, stride=1):
        super().__init__()

        # 특징을 계산하는 합성곱 경로
        self.layers = nn.Sequential(

            nn.Conv2d(
                input, output,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(
                output, output,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(output),
        )

        # 입력과 출력의 모양이 같으면 그대로 전달합니다.
        self.shortcut = nn.Identity()

        # 모양이 다르면 지름길에서도 채널과 크기를 맞춥니다.
        if input != output or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input, output,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(output),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        result = self.layers(x)      # 합성곱 계산
        original = self.shortcut(x)  # 입력 전달 또는 모양 변환

        # 잔차 덧셈을 블록 안에서 완료합니다.
        return self.relu(result + original)


class Resnet34(nn.Module):
    """1채널 Log-Mel을 받아 세 종류의 소리를 분류합니다."""

    def __init__(self):
        super().__init__()

        # CONV1: 입력 1채널 → 64채널
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # CONV2: 기본 블록 3개
        self.conv2 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        # CONV3: 기본 블록 4개
        self.conv3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        # CONV4: 기본 블록 6개
        self.conv4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
        )

        # CONV5: 기본 블록 3개
        self.conv5 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.result = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)

        # 각 BasicBlock 안에서 잔차 덧셈까지 수행합니다.
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.result(x)

        return x