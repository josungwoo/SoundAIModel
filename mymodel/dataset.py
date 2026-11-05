"""음원과 정답을 준비하고, 파형을 CNN 입력으로 바꾸는 파일입니다.

실행 순서: 데이터 목록 준비 → 음원 구간 읽기 → 파형 정리 → Log-Mel 만들기.
기존 버전과 계산 방법은 같고, 짧게 줄여 쓴 문법을 기본 반복문으로 풀었습니다.
공식 common_utils.py/config.yaml의 Log-Mel 설정을 참고한 구성입니다.
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "datasets" / "extracted"
RESULTS = ROOT / "results"
SPLIT_FILE = RESULTS / "splits.json"

# 번호 0, 1, 2와 이름의 순서를 학습·저장·평가에서 똑같이 사용합니다.
CLASS_IDS = ["M_1_01", "O_3_03", "H_3_03"]
CLASS_NAMES = ["공작기계음", "금속 충돌음", "비명"]
PLOT_NAMES = ["Machine", "Metal", "Scream"]
SETTINGS = {
    "sample_rate": 22050,   # Hz: 소리 1초를 숫자 22,050개로 나타냅니다.
    "num_samples": 90316,   # 한 음원의 길이: 90316 / 22050 ≈ 4.096초입니다.
    "n_fft": 2048,          # 주파수 성분을 계산할 때 한 번에 살펴볼 샘플 수입니다.
    "hop_length": 256,      # 다음 주파수 계산으로 넘어갈 때 이동할 샘플 수입니다.
    "n_mels": 128,          # 소리 그림의 세로 칸 수입니다. 주파수를 128개 영역으로 묶습니다.
    "f_min": 10.0,          # Mel 영역의 가장 낮은 주파수이며 단위는 Hz입니다.
}


def save_json(path, value):
    """path: 저장 경로 / value: 저장할 데이터. 반환값은 없습니다."""
    path = Path(path)
    # parents=True: 중간 폴더도 생성. exist_ok=True: 이미 있는 폴더는 그대로 사용.
    path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def load_waveform(path, settings, segment=None):
    """음원 한 개를 읽고, 길이가 같은 1차원 파형 텐서로 반환합니다.

    path: 음원 경로 / settings: 신호처리 설정.
    segment: [시작 초, 끝 초]. 없으면 전체 음원을 대상으로 합니다.
    잡음 제거와 데시벨 측정은 하지 않습니다.
    """
    target_rate = settings["sample_rate"]
    target_length = settings["num_samples"]
    duration = target_length / target_rate  # 90316 / 22050 ≈ 4.096초

    # with 안에서 파일을 사용하고, 블록을 나가면 파일을 자동으로 닫습니다.
    with sf.SoundFile(str(path)) as audio:
        original_rate = audio.samplerate
        total_seconds = len(audio) / original_rate

        # 1. 전달받은 정답 구간을 사용합니다. 구간이 없으면 전체 음원이 대상입니다.
        if segment is None:
            left = 0.0
            right = total_seconds
        else:
            left = segment[0]
            right = segment[1]
            # JSON 시간의 작은 반올림 오차는 기존과 같이 0.05초까지 허용합니다.
            if not 0 <= left < right <= total_seconds + 0.05:
                raise ValueError(f"정답 구간이 음원 범위를 벗어납니다: {path}")
            right = min(right, total_seconds)

        # 2. 긴 구간은 중앙 창, 짧은 구간은 구간 시작부터 읽습니다.
        segment_length = right - left
        extra_length = max(0.0, segment_length - duration)
        offset = left + extra_length / 2
        # seek는 읽을 위치를 옮깁니다. 초 × 원본 샘플링 주파수 = 샘플 위치.
        start_sample = int(offset * original_rate)
        wanted_samples = round(duration * original_rate)
        remaining_samples = int((right - offset) * original_rate)
        count = min(wanted_samples, remaining_samples)
        audio.seek(start_sample)
        # always_2d=True라서 모노도 [시간, 채널] 모양으로 읽힙니다.
        samples = audio.read(count, dtype="float32", always_2d=True)

    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError(f"비어 있거나 잘못된 숫자가 있는 음원입니다: {path}")

    # 3. 각 시각의 채널 평균: 스테레오 [시간, 2]도 모노 [시간]으로 바꿉니다.
    mono_samples = samples.mean(axis=1)
    waveform = torch.from_numpy(mono_samples.copy())

    # 4. 재샘플링은 같은 시간 길이를 유지하면서 1초당 샘플 수를 맞춥니다.
    if original_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, target_rate)

    # 5. 정확히 90,316개로 맞춥니다. 부족한 칸은 뒤에 0을 채웁니다.
    waveform = waveform[:target_length]
    missing = target_length - len(waveform)
    waveform = torch.nn.functional.pad(waveform, (0, missing))

    # 6. 진폭 정규화: 가장 큰 절댓값으로 나눕니다. 잡음도 함께 커질 수 있습니다.
    # 1e-7을 더하면 무음(최댓값 0)도 0으로 나누지 않습니다.
    peak = waveform.abs().max()
    waveform = waveform / (peak + 1e-7)
    return waveform


def summarize(parts):
    """parts의 train/val/test 목록을 세어, 개수 정보를 반환합니다."""
    summary = {}
    for name, records in parts.items():
        raw_ids = set()  # set은 같은 원본 ID가 여러 번 나와도 하나만 남깁니다.
        class_counts = {}
        for label in CLASS_NAMES:
            class_counts[label] = 0

        for record in records:
            raw_ids.add(record["raw_id"])
            class_name = CLASS_NAMES[record["target"]]
            class_counts[class_name] += 1

        summary[name] = {
            "files": len(records),
            "raw_recordings": len(raw_ids),
            "classes": class_counts,
        }
    return summary


def prepare_data():
    """인자 없이 데이터 목록을 준비하고, 분할 정보 딕셔너리를 반환합니다."""
    # 기존 목록이 있으면 다시 나누지 않습니다. 이 경우 새 자료를 재검사하지 않습니다.
    if SPLIT_FILE.exists():
        print("기존 분할을 재사용합니다:", SPLIT_FILE)
        split_text = SPLIT_FILE.read_text(encoding="utf-8")
        return json.loads(split_text)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {DATA_DIR}")

    # 1. MP3 목록을 만듭니다. 같은 분할 안에서 이름이 같은 JSON과 짝을 찾습니다.
    # 파일명은 짝 찾기에만 사용하며, 소리의 정답은 JSON에서 읽습니다.
    audio_files = {}
    for path in sorted(DATA_DIR.rglob("*.mp3")): # DATA_DIR 폴더 아래에 mp3 파일을 찾아서 리스트로 만든다
        partition = path.relative_to(DATA_DIR).parts[0] # DATA_DIR 아래 첫번째 폴더 이름을 가져온다
        key = (partition, path.stem) # 파일 탐색에 사용할 키값 (폴더이름, 파일이름[확장자제외])
        # 오디오의 경로를 저장하는 딕셔너리를 만든다
        if key not in audio_files: 
            audio_files[key] = []
        audio_files[key].append(path)

    official = {"1.Training": [], "2.Validation": []} # 
    # [공작기계음 개수, 금속 충돌음 개수, 비명 개수]
    label_counts = {"1.Training": [0, 0, 0], "2.Validation": [0, 0, 0]}
    excluded = [] # 예외를 저장할 배열 

    # 2. 세 종류의 JSON만 선택하고, 실제 음원이 정상적으로 읽히는지 확인합니다.

    for label_path in sorted(DATA_DIR.rglob("*.json")): # DATA_DIR 폴더 아래에 json 파일을 찾아서 리스트로 만든다
        partition = label_path.relative_to(DATA_DIR).parts[0]# DATA_DIR 아래 첫번째 폴더 이름을 가져온다
        # 
        if partition not in official:
            continue

        """여기서 부터 다시 봐야 한다,여기서 부터 다시 봐야 한다,여기서 부터 다시 봐야 한다,여기서 부터 다시 봐야 한다,여기서 부터 다시 봐야 한다 """
        # 잘못된 자료는 이유를 기록하고 제외합니다. 무음으로 바꿔 학습하지 않습니다.
        try:
            metadata = json.loads(label_path.read_bytes())
            label = metadata["LabelDataInfo"]
            division1 = str(label["Division1"])
            division2 = str(label["Division2"])
            sound_class = str(label["Class"])
            class_id = division1 + "_" + division2 + "_" + sound_class
            if class_id not in CLASS_IDS:
                continue

            # 순서에 따라 공작기계음=0, 금속 충돌음=1, 비명=2로 바꿉니다.
            target = CLASS_IDS.index(class_id)
            label_counts[partition][target] += 1
            key = (partition, label_path.stem)
            matches = audio_files.get(key, [])
            if len(matches) != 1:
                raise ValueError(f"MP3 짝이 정확히 하나여야 합니다. 발견: {len(matches)}개")

            # 가장 긴 정답 구간을 찾습니다. 길이가 같으면 먼저 나온 구간을 유지합니다.
            segments = label["Segmentations"]
            if len(segments) == 0:
                raise ValueError("정답 시간 구간이 없습니다.")
            segment = segments[0]
            for candidate in segments:
                current_length = segment[1] - segment[0]
                candidate_length = candidate[1] - candidate[0]
                if candidate_length > current_length:
                    segment = candidate

            load_waveform(matches[0], SETTINGS, segment=segment)
            record = {
                "audio": str(matches[0]),
                "json": str(label_path),
                "raw_id": str(metadata["RawDataInfo"]["RawDataId"]),
                "target": target,
                "segment": segment,
            }
            official[partition].append(record)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            excluded.append({"json": str(label_path), "reason": str(error)})

    # 3. 같은 원본 녹음에 서로 다른 정답이 붙어 있지 않은지 확인합니다.
    parts = {"train": [], "val": [], "test": official["2.Validation"]}
    rng = random.Random(42)  # 같은 목록이면 같은 순서로 섞이도록 시작값을 고정합니다.
    owners = {}
    for records in official.values():
        for record in records:
            raw_id = record["raw_id"]
            if raw_id not in owners:
                owners[raw_id] = record["target"]
            if owners[raw_id] != record["target"]:
                raise ValueError("같은 원본 ID에 다른 종류가 붙어 있습니다. 분할 방법을 확인하세요.")

    # 4. Training을 원본 ID 단위로 약 80:20 분할합니다. Validation은 test로 둡니다.
    for target in range(3):
        rows = []
        raw_id_set = set()
        for record in official["1.Training"]:
            if record["target"] == target:
                rows.append(record)
                raw_id_set.add(record["raw_id"])

        raw_ids = sorted(raw_id_set)
        if len(raw_ids) < 2:
            raise ValueError("종류마다 학습·검증으로 나눌 수 있는 원본 수가 필요합니다.")
        rng.shuffle(raw_ids)
        val_count = max(1, round(len(raw_ids) * 0.2))
        val_ids = set(raw_ids[:val_count])
        for row in rows:
            if row["raw_id"] in val_ids:
                parts["val"].append(row)
            else:
                parts["train"].append(row)

    # 5. 같은 원본이 학습·검증·테스트에 겹치면 성적이 부풀 수 있어 중단합니다.
    split_raw_ids = {}
    for name, records in parts.items():
        split_raw_ids[name] = set()
        for record in records:
            split_raw_ids[name].add(record["raw_id"])

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_raw_ids[left].intersection(split_raw_ids[right])
        if overlap:
            raise ValueError(f"{left}/{right} 사이에 원본 ID가 중복됩니다: {sorted(overlap)}")

    for name, records in parts.items():
        targets = set()
        for record in records:
            targets.add(record["target"])
        if targets != {0, 1, 2}:
            raise ValueError(f"{name}에 세 종류의 자료가 모두 필요합니다.")

    # 목록만 저장합니다. 파형과 Log-Mel은 자료를 가져올 때 계산합니다.
    split = {
        "seed": 42,
        "class_ids": CLASS_IDS,
        "class_names": CLASS_NAMES,
        "settings": SETTINGS,
        "parts": parts,
        "summary": summarize(parts),
        "source_label_counts": label_counts,
        "excluded": excluded,
        "raw_id_overlap": False,
        "window_policy": "longest annotation, center, zero pad",
    }
    save_json(SPLIT_FILE, split)
    return split


class SoundDataset(Dataset):
    """DataLoader가 요청한 번호의 소리 특징과 정답을 한 개씩 전달합니다."""

    def __init__(self, records, settings):
        # self는 현재 데이터셋 자신입니다. 목록과 설정을 나중에도 쓰도록 보관합니다.
        self.records = records
        self.settings = settings
        # 여기서는 변환기만 준비합니다. 실제 변환은 make_feature에서 합니다.
        # 파형·Log-Mel은 CPU에서 계산하며, CNN 쪽의 장치는 main.py가 정합니다.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=settings["sample_rate"],
            n_fft=settings["n_fft"],
            hop_length=settings["hop_length"],
            n_mels=settings["n_mels"],
            f_min=settings["f_min"],
        )

    def make_feature(self, waveform):
        """waveform: 1차원 파형. 현재 설정에서는 [1,128,353] 특징을 반환합니다."""
        mel = self.mel(waveform)
        mel = mel.clamp(min=1e-6)  # log(0)을 막기 위해 아주 작은 값에 하한을 둡니다.
        log_mel = mel.log()  # 자연로그입니다. 데시벨 변환은 아닙니다.
        feature = log_mel.unsqueeze(0)  # [128,353] → [1,128,353], 채널 축 추가.
        return feature

    def __len__(self):
        # len(dataset)을 호출하면 실행됩니다. 배치 수가 아닌 자료 수입니다.
        return len(self.records)

    def __getitem__(self, index):
        # dataset[index] 또는 DataLoader가 index번째 자료를 요청하면 실행됩니다.
        row = self.records[index]
        waveform = load_waveform(row["audio"], self.settings, segment=row["segment"])
        feature = self.make_feature(waveform)
        target = torch.tensor(row["target"], dtype=torch.long)
        # 특징은 실수 배열, 정답은 CrossEntropyLoss에서 사용하는 정수 0/1/2입니다.
        return feature, target
