# CauSyncVLA

**Causal Latent World Modeling for Asynchronous Robot Control**

CauSyncVLA updates a frozen vision-language-action policy between visual observations. A learned updater starts from the latest published visual KV cache and conditions on the state and action history actually executed since that observation. The frozen action expert can refine the shifted tail of its previous plan with one flow step. KV prediction and action warm starting are independently configurable.

[Paper (arXiv)](https://arxiv.org/abs/2611.25667) · [Updater weights on Hugging Face](https://huggingface.co/SSSSphinx/CauSyncVLA) · [Experiment settings](docs/EXPERIMENTS.md) · [Reported results](docs/PAPER_RESULTS.md) · [Franka profile](franka/README.md)

## Installation

Use Linux, a CUDA GPU and Python 3.11. This tree includes the OpenPI code used by the method, its Python client, the dependency lock and the PyTorch transformer replacements. Install them in a dedicated environment:

```sh
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.11
source .venv/bin/activate
cp -r src/openpi/models_pytorch/transformers_replace/. .venv/lib/python3.11/site-packages/transformers/
```

For simulation, install the [LIBERO environment and assets](https://github.com/Lifelong-Robot-Learning/LIBERO) in that environment. The evaluations used robosuite 1.4.1 and MuJoCo 2.3.7. Configure LIBERO's BDDL, assets and initial-state paths for the checkout you use. For LIBERO-Plus, select its corresponding loader and assets in a separate invocation. The evaluator checks the expected task count.

## Weights

Download the LIBERO updater:

```sh
hf download SSSSphinx/CauSyncVLA --include 'libero/*' --local-dir weights
```

For LIBERO, use the official frozen `pi05_libero` checkpoint from [OpenPI](https://github.com/Physical-Intelligence/openpi). Convert it with the supplied converter and retain the original normalization assets:

```sh
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi05_libero --config-name pi05_libero \
  --output-path models/pi05_libero_pytorch --precision bfloat16
```

Check that `models/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json` is present. The released LIBERO adapter contains its original updater normalization buffers. The loader checks the adapter architecture, profile, and matching action normalization, then strictly loads the parameters. The Franka updater and its matching action expert are privately held.

## Native evaluation

Run from the repository root. This example uses the ten official tasks, fifty provided initial states per task, and seed 7:

```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m openpi.dual.eval_libero \
  --config configs/libero.json --benchmark native --tasks all --trials 50 \
  --seed 7 --max-steps 520 --action-metrics --out runs/libero-native
```

Set `--tasks 0 --trials 1` for a single episode. Each live policy session receives the action actually issued at the preceding step. Reset the session at episode boundaries. For ablations use the configs in `configs/`. Native success, paced closed-loop behavior and resident-input throughput measure different properties.

## Training

`openpi.dual.train` trains the updater while freezing the VLM and action expert. The teacher and student use the same flow noise. The trainer accepts a fixed train/validation split and supports DDP. The supplied recipes retain the submitted run's batch, learning-rate, history and loss settings. Supply the datasets, expert rollout recordings and split named by the recipes before training:

```sh
torchrun --standalone --nproc-per-node=2 scripts/train_updater.py configs/init_libero.json
torchrun --standalone --nproc-per-node=2 scripts/train_updater.py configs/train_libero.json
```

The reported LIBERO checkpoint is the 12,000-update milestone. The original training run continued to 18,000 updates. The simulator action chunk has H=10 and C=5, with a single expert step and PFWS restart time 0.5. The physical profile uses a separate time-conditioned updater and C=2.

## License and attribution

See [LICENSE](LICENSE), [NOTICE](NOTICE) and the accompanying [Gemma terms](LICENSE_GEMMA.txt). OpenPI, the datasets and the separately obtained foundation checkpoints retain their respective upstream terms.

## Citation

```bibtex
@misc{pan2026causyncvla,
  title         = {{CauSyncVLA}: Causal Latent World Modeling for Asynchronous Robot Control},
  author        = {Xu Pan and Chubin Zhang and Xingrui Yu and Runmin Lan and Zihao Wang and Wenkai Guo and Bofang Jia and Bo An and Ziwei Wang},
  year          = {2026},
  eprint        = {2611.25667},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.25667}
}
```
