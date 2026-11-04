# SoundAIModel

Mac에서 실행하는 학생용 3클래스 음향 분류 프로그램입니다.

- `mymodel/dataset.py`: 자료 준비, 원본 단위 분할, 파형 전처리, Log-Mel 변환.
- `mymodel/model.py`: SmallCNN 구조.
- `mymodel/main.py`: 준비·학습·평가·한 음원 예측.
- `mymodel/requirements.txt`: 기존 Mac 환경에서 기록한 의존성 버전.
- `datasets/`: 원음과 원본 라벨 대신 데이터 파일 목록·공식 다운로드 안내.

## 실행

Python 환경에 requirements.txt의 라이브러리가 필요합니다. 기록된 버전의 설치 가능 여부는 새 환경에서 별도 확인해야 합니다. 가상환경·가중치·실험 결과는 이 저장소에 포함하지 않습니다.

```bash
cd mymodel
python main.py prepare
python main.py train --epochs 1
python main.py evaluate --split val
```

먼저 `datasets/README.md`에 따라 공식 데이터를 준비하세요. 처음 학습하면 results 폴더와 새 모델이 생성됩니다. 저장된 모델이 없으면 평가·예측부터 실행할 수 없습니다.

현재 로컬 상태는 학습 1,389개, 내부 검증 347개, 최종 테스트 215개입니다. 단일 음원이 세 대상 중 하나라는 조건의 분류이며, 다른 소리·무음·복합음·사고 위험 여부를 판정하는 시스템은 아닙니다.

코드 4개는 로컬 원본을 수정하지 않고 그대로 복사했습니다. `datasets/DATA_SUMMARY.json`에 SHA-256을 기록했습니다.
