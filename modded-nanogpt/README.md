# Modded NanoGPT

This code builds on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and follows the setup used in the [Scion modded-nanogpt examples](https://github.com/LIONS-EPFL/scion/tree/main/examples/modded-nanogpt).

## Setup

Install the dependencies used by modded-nanogpt and prepare FineWeb binary shards.

```
pip install -r requirements.txt
pip install -r data/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade
python data/cached_fineweb100B.py 300 # downloads only the first 30B training tokens to save time
```

The data will be stored in `data/fineweb100B/` as default. The paths can be overridden with `--input_bin` and `--input_val_bin`.

By default, the scripts log validation loss to Weights & Biases under the project `soda-modded-nanogpt`. Set `--wandb_project ""` to disable wandb logging, or pass `--wandb_run_name` to name a run.

## Run

The concrete algorithm implementations of MODA and SODA can be run as single-file training scripts:

```bash
torchrun --standalone --nproc_per_node=4 train_gpt_moda.py
torchrun --standalone --nproc_per_node=4 train_gpt_soda.py
```

Base optimizers (Adam, Muon, uScion) can be wrapped by SODAWrapper:

```bash
torchrun --standalone --nproc_per_node=4 train_gpt_adam_wrapped.py
torchrun --standalone --nproc_per_node=4 train_gpt_muon_wrapped.py
torchrun --standalone --nproc_per_node=4 train_gpt_uscion_wrapped.py
```

Notes:

* `train_gpt_muon_wrapped.py` and `train_gpt_uscion_wrapped.py` use the default `SODAWrapper`.
* `train_gpt_adam_wrapped.py` applies SODAWrapper after the warmup phase, so `SODAWrapper_SkipWarmup` is defined inside this script.
* When changing `n_embd`, remember to change `n_head` accordingly to `n_embd // 128` to maintain head dimension of 128.
