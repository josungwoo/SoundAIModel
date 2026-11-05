"""준비·학습·평가를 실행합니다. 사용법: python main.py --help

소리 입력은 dataset.py, 신경망 구조는 model.py에 있고 실행 절차는 이 파일에 있습니다.
학습과 평가의 차이가 드러나도록 기본적인 반복문으로 작성했습니다.
"""

import argparse # 터미널 명령어 파싱을 위한 라이브러리 
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG 그림으로 저장합니다. Mac 터미널에서도 동작합니다.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (CLASS_IDS, CLASS_NAMES, PLOT_NAMES, RESULTS, SoundDataset,
                     prepare_data, save_json)
from model import SmallCNN
from model import Resnet34


current_model = Resnet34

def choose_device(name):
    """auto는 Mac GPU가 있으면 MPS, 없으면 CPU를 고릅니다."""
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("이 Mac에서 MPS를 사용할 수 없습니다. --device cpu를 사용하세요.")
    return torch.device(name)

#############################################################################
def calculate_metrics(confusion):
    """행=실제 정답, 열=예측인 3×3 혼동행렬에서 성적을 계산합니다."""
    tp = np.diag(confusion)                  # 대각선: 정답과 예측이 같은 개수입니다.
    fp = confusion.sum(axis=0) - tp           # 그 종류라고 했지만 실제로는 다른 소리입니다.
    fn = confusion.sum(axis=1) - tp           # 실제 그 소리인데 다른 소리로 잘못 분류했습니다.
    precision = tp / np.maximum(tp + fp, 1)   # 있다고 예측한 것 중 맞힌 비율입니다.
    recall = tp / np.maximum(tp + fn, 1)      # 실제 그 소리 중 찾아낸 비율입니다.
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {
        "samples": int(confusion.sum()),
        "accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "macro_f1": float(f1.mean()),         # 세 종류 F1을 같은 비중으로 평균 냅니다.
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(fp[i]),
                             "fn": int(fn[i]), "support": int(confusion[i].sum())}
                      for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion.tolist(),
        "matrix_axes": "rows=true, columns=predicted",
    }



def evaluate_model(model, loader, device):
    """가중치를 고치지 않고, 실제 음원에서 얻은 특징으로만 예측합니다."""
    model.eval()  # 평가 모드입니다. 가중치를 수정하거나 초기화하는 명령이 아닙니다.
    confusion = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():  # 평가에는 역전파가 없어 기울기 계산 기록이 필요 없습니다.
        for features, targets in loader: 
            logits = model(features.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("평가 출력에 잘못된 숫자가 있습니다.")
            predicted = logits.argmax(dim=1).cpu()  # 각 자료에서 가장 큰 점수의 번호를 고릅니다.
            for actual, guess in zip(targets.tolist(), predicted.tolist()):
                confusion[actual, guess] += 1
    return calculate_metrics(confusion)
#############################################################################

# 에포크 별 손실률과 정확도를 그래프로 그려서 저장한다
def plot_history(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].plot(epochs, [r["val"]["accuracy"] for r in history], "o-", label="Val accuracy")
    axes[1].plot(epochs, [r["val"]["macro_f1"] for r in history], "s-", label="Val macro F1")
    axes[1].set(title="Internal validation", xlabel="Epoch", ylabel="Score", ylim=(0, 1.05))
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)  # 다 그린 그림의 메모리를 정리합니다.

# 3by3 행렬로 예측과 정답 매트릭스를 만들어서 저장한다
def plot_confusion(result, path, split_name):
    matrix = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=PLOT_NAMES, yticklabels=PLOT_NAMES,
           xlabel="Predicted class", ylabel="True class", title=f"{split_name} confusion matrix")
    for row in range(3):
        for col in range(3):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="Number of windows")
    fig.savefig(path, dpi=150)
    plt.close(fig)

def model_define(device, state=None):
    # 저장 정보를 전달하지 않으면 새 학습용 모델을 만듭니다.
    if state is None:
        model = current_model().to(device)

    # 저장 정보를 전달하면, 저장된 종류에 맞는 모델을 복원합니다.
    else:
        model_name = state.get("model_name", "SmallCNN")

        if model_name == "SmallCNN":
            model = SmallCNN().to(device)
        elif model_name == "Resnet34":
            model = Resnet34().to(device)
        else:
            raise ValueError("지원하지 않는 모델입니다.")

        model.load_state_dict(state["model_state"])

    return model

# 학습 하는 함수 !! 중요 -------------------------------------------------------------
def train(args, device):
    split = prepare_data()
    # 한국 시간 기준 연월일_시분으로 저장합니다. 초와 소수초는 넣지 않습니다.
    run = RESULTS / datetime.now(ZoneInfo("Asia/Seoul")).strftime("train_%Y%m%d_%H%M")
    # 같은 분에 다시 실행해도 기존 가중치와 기록을 덮어쓰지 않습니다.
    try:
        run.mkdir(parents=True)
    except FileExistsError as error:
        raise ValueError("같은 한국 시간 분의 학습 폴더가 이미 있습니다. 다음 분에 다시 실행하세요.") from error
    save_json(run / "splits.json", split)
    train_loader = DataLoader(SoundDataset(split["parts"]["train"], split["settings"]),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SoundDataset(split["parts"]["val"], split["settings"]),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model_define(device)  # 기존 모델 가중치를 불러오지 않고 새로 시작합니다.
    loss_function = nn.CrossEntropyLoss()  # 정답은 정수 0/1/2, 입력은 softmax 전 점수입니다.

    # 옵티마이저를 사용하여 모델의 가중치를 조절합니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_f1 = -1.0
    steps = 0
    started = time.monotonic()
    print(f"학습 장치: {device}, 학습 {len(train_loader.dataset)}개, 검증 {len(val_loader.dataset)}개", flush=True)
    # 실행 중 실패를 한 곳에서 기록합니다. 모든 계산 줄마다 예외 처리를 넣지는 않습니다.
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            count = 0
            for batch_number, (features, targets) in enumerate(train_loader, start=1):
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # 1. 이전 배치의 기울기를 지웁니다. 배운 가중치는 지우지 않습니다.
                logits = model(features)  # 2. 실제 소리 그림으로 세 종류 점수를 계산합니다.
                loss = loss_function(logits, targets)  # 3. 정답과 비교해 틀린 정도를 계산합니다.
                if not torch.isfinite(loss):
                    raise ValueError("손실이 NaN 또는 무한대입니다.")
                loss.backward()        # 4. 어느 가중치를 어떻게 고칠지 계산합니다.
                optimizer.step()       # 5. 가중치를 실제로 고칩니다. 이 단계가 학습입니다.
                steps += 1
                loss_sum += loss.item() * len(targets)
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                count += len(targets)
                if batch_number % 25 == 0 or batch_number == len(train_loader):
                    print(f"epoch {epoch}, batch {batch_number}/{len(train_loader)}, loss {loss.item():.4f}", flush=True)
            val = evaluate_model(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": loss_sum / count,
                   "train_accuracy": correct / count, "val": val}
            history.append(row)
            save_json(run / "history.json", history)
            if val["macro_f1"] > best_f1:
                best_f1 = val["macro_f1"]
                # CPU로 복사해 저장하면 CPU 시연에서도 다시 읽기 쉽습니다.
                torch.save({"model_name": current_model.__name__,"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "class_names": CLASS_NAMES, "class_ids": CLASS_IDS,
                            "settings": split["settings"], "epoch": epoch, "seed": 42,
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            },
                           run / "best.pt")
            print(f"epoch {epoch}: loss={row['train_loss']:.4f}, train ACC={row['train_accuracy']:.4f}, "
                  f"val ACC={val['accuracy']:.4f}, val Macro F1={val['macro_f1']:.4f}", flush=True)
        plot_history(history, run / "learning_curve.png")
        save_json(run / "training_result.json", {"status": "completed", "epochs": len(history),
                  "optimizer_steps": steps, "device": str(device), "batch_size": args.batch_size,
                  "learning_rate": args.learning_rate, "seconds": time.monotonic() - started,
                  "best_val_macro_f1": best_f1, "final_test_executed_during_training": False,
                  "summary": split["summary"]})
        (RESULTS / "latest_run.txt").write_text(run.name, encoding="utf-8")
        print("학습 완료:", run, flush=True)
    except BaseException as error:
        save_json(run / "failure.json", {"error": str(error), "type": type(error).__name__,
                  "optimizer_steps": steps, "completed_epochs": len(history)})
        raise


def load_trained_model(run_name, device):
    """새 구조에 저장 가중치를 입혀, 같은 학습 상태를 복원합니다."""
    if run_name == "latest":
        pointer = RESULTS / "latest_run.txt"
        if not pointer.exists():
            raise FileNotFoundError("저장 모델이 없습니다. 먼저 python main.py train --epochs 1을 실행하세요.")
        run_name = pointer.read_text(encoding="utf-8").strip()
    run = Path(run_name).expanduser()
    if not run.is_absolute():
        run = RESULTS / run
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if state["class_ids"] != CLASS_IDS or state["class_names"] != CLASS_NAMES:
        raise ValueError("저장 모델의 종류 순서가 현재 코드와 다릅니다.")
    model = model_define(device, state)
    model.eval()
    return model, state, run

# 평가
def evaluate(args, device):
    model, state, run = load_trained_model(args.run, device)
    split = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    if state["settings"] != split["settings"]:
        raise ValueError("저장 모델과 분할 파일의 전처리 설정이 다릅니다.")
    dataset = SoundDataset(split["parts"][args.split], state["settings"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    result = evaluate_model(model, loader, device)
    result.update(split=args.split, epoch=state["epoch"], device=str(device),
                  checkpoint=str(run / "best.pt"), annotation_window_evaluation=True, field_test=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = run / f"evaluation_{args.split}_{stamp}.json"
    save_json(output, result)
    plot_confusion(result, output.with_suffix(".png"), args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("저장:", output)

#############################################################################
# 파서를 통해서 CLI 수준에서 AI 모델의 동작을 제어하는 단계 
def main():
    # 파서를 사용하여 명령어 파싱을 위한 객체를 생성합니다.
    parser = argparse.ArgumentParser(description=" 기본CNN 음향 분류") 
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="데이터를 확인하고 원본 단위 분할을 저장합니다.")
    training = commands.add_parser("train", help="새 모델을 훈련하고 내부 검증합니다.")
    training.add_argument("--epochs", type=int, default=1, help="학습 자료를 도는 횟수")
    training.add_argument("--batch-size", type=int, default=8, help="한 번에 처리할 음원 수")
    training.add_argument("--learning-rate", type=float, default=0.001, help="가중치 수정 크기를 조절")
    evaluation = commands.add_parser("evaluate", help="저장 모델을 재로딩하여 평가합니다.")
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--batch-size", type=int, default=8)
    for command in [training, evaluation]:
        command.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    evaluation.add_argument("--run", default="latest", help="결과 폴더 이름 또는 절대 경로")
    args = parser.parse_args() 
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < float("inf"):
            parser.error("에포크·배치는 1 이상, 학습률은 유한한 양수여야 합니다.")
    if args.command == "evaluate" and args.batch_size < 1:
        parser.error("배치는 1 이상이어야 합니다.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)  # 난수 시작점을 맞춥니다. MPS의 완전한 수치 재현을 보장하지는 않습니다.
    torch.set_num_threads(2)  # CPU에서 동시에 계산할 작업 수를 작게 고정합니다.


    # 필수는 아님 // 데이터  확인용도 
    if args.command == "prepare":
        split = prepare_data()
        print(json.dumps({"summary": split["summary"], "excluded": split["excluded"],
                          "raw_id_overlap": split["raw_id_overlap"]}, ensure_ascii=False, indent=2))
        return


    device = choose_device(args.device) # 동적으로 변환된 디바이스값을 반환
    if args.command == "train":
        train(args, device)
    elif args.command == "evaluate":
        evaluate(args, device)


if __name__ == "__main__":
    # 직접 실행할 때만 시작합니다. 다른 파일에서 import해도 학습이 시작되지 않습니다.
    try:
        main()
    # 에러처리 
    except (ValueError, FileNotFoundError) as error:
        print("실행 중단:", error, file=sys.stderr)
        sys.exit(1)
