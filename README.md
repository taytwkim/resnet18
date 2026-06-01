# ResNet-18

Toy training workload to experiment with neural networks. ResNet-18 trained on CIFAR-10.

## 🚀 Getting Started

1. Activate a `venv`.

```shell
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

2. Install dependencies.

* If on CPU or MPS (Apple Silicon):

```shell
pip install -r requirements.txt
```

* If on NVIDIA, prefer CUDA wheels:

```shell
pip install torch torchvision numpy pillow --index-url https://download.pytorch.org/whl/cu121
```

3. Train model.

* For CPU or MPS:

```shell
python3 train.py --epochs 5 --batch-size 128
```

* On NVIDIA GPUs, enable mixed precision (`amp`) for speed:

```shell
python3 train.py --epochs 20 --batch-size 256 --amp --workers 4
```

## 🚂 Training Arguments

```shell
python3 train.py [--epochs <INT>] [--batch-size <INT>] [--lr <FLOAT>] 
                 [--data <PATH>] [--out-dir <DIR>]
                 [--workers <INT>] [--amp] [--label-smoothing <FLOAT>]
                 [--warmup <INT>] [--seed <INT>] [--resume <PATH>]
                 [--save-every <INT>]
```

  * `--epochs`: Number of full passes over the training set.
  * `--batch-size`: Mini-batch size for training and evaluation.
  * `--lr`: Base learning rate for SGD.
  * `--data`: CIFAR-10 data directory; downloaded here if missing.
  * `--out-dir`: Output directory for checkpoints and final weights.
  * `--workers`: DataLoader worker count. CPU: 0–2; GPU: 4–8 typical.
  * `--amp`: Enable mixed precision on CUDA (ignored on CPU/MPS).
  * `--label-smoothing`: Label smoothing for cross-entropy (e.g., 0.1).
  * `--warmup`: Optimizer steps (not epochs) of linear warmup before cosine decay.
  * `--seed`: RNG seed for Python and PyTorch.
  * `--resume`: Resume from checkpoint produced by this script.
  * `--save-every`: Also save a snapshot every N epochs (set as 0 to disable).

### Notes

* A `DataLoader` is a PyTorch utility that prepares and feeds data to the training loop. Here, it reads CIFAR-10 samples, applies transformations like random crop and horizontal flip, and assembles batches. This work is done lazily during training rather than all up front, so batches are prepared as they are needed. The number of workers indicates how many CPU worker processes are used to do that work in the background.
* *Label smoothing* is a small adjustment to the target labels used by the loss function during training. Normally, the correct class is treated as `1` and every incorrect class as `0`. With label smoothing, the correct class is set to a little less than `1`, and the incorrect classes to a little more than `0`. This helps keep the model from becoming too confident, which can improve generalization and sometimes make training more stable.
* *Warmup* means starting training with a smaller learning rate and gradually increasing it over the first several training steps. We use it because the beginning of training can be unstable: the model starts from random weights, and early updates can be noisy or too aggressive if the learning rate is high right away. Warmup gives the optimizer a gentler start before training continues with its normal learning rate updates.

## 🗂️ Directory
```
resnet18/
├─ artifacts/
├─ data/
├─ requirements.txt
└─ train.py
```

* `data/` [Empty directory tracked by git]
    * `cifar-10-python.tar.gz`: the original compressed dataset that `torchvision` downloads.
    * `cifar-10-batches-py/`: extracted from the tarball. This is what `torchvision.datasets.CIFAR10` actually reads.
        * `data_batch_1` … `data_batch_5`: 5 training batches - 10,000 images each.
        * `test_batch`: 10,000 test samples.
* `artifacts/` [Empty directory tracked by git]
    * `*_final_weights.pt` (weights only)
        * Use for inference or fine-tuning.
        * Contains just `model.state_dict()`.
    * `*_best.pt` (full checkpoint)
        * Use to resume training with identical optimizer dynamics.
        * Contains model weights, optimizer state (e.g., momentum), scheduler state (e.g., where you are on the LR curve), plus metadata like epoch and best accuracy.
