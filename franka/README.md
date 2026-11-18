# Franka updater

The physical policy uses the 100 Hz feedback adapter and the frozen FR3 action expert. It consumes measured seven-dimensional state, the preceding issued action and measured time intervals. Its history has 24 slots and its condition vector has 383 components. The LIBERO and FR3 normalizations are different.

1. Prepare the runtime from the root of this checkout:

   ```sh
   python scripts/prepare_franka.py .runtime/franka
   export PYTHONPATH="$PWD/.runtime/franka:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
   ```

2. Place the matching privately distributed FR3 backbone, including its normalization assets, in `models/pi05_fr3_pytorch/`. If you have access to the private FR3 updater, place it at `weights/franka/updater.safetensors`. Use Python from the same OpenPI environment. Inspect the interface with `python -m openpi.robot.server --help`. The server default is loopback; the released code implements the policy side of the protocol. Hardware-specific drivers and the laboratory operator console are maintained separately.

3. The policy configuration is `configs/franka.json`. After a fresh real visual anchor, inference consumes the actual issued actions. The physical setup used the latest feedback, a 100 Hz host dispatch ceiling, H=10 and C=2. Physical hold ticks, new action nodes, and new plans have different rates.

For training on the user's own aligned recordings, `python scripts/prepare_feedback.py --help` describes the filtered source, measured timeline and split inputs. It holds the original action across synthetic substeps, preserves real/synthetic flags and segments gripper transitions. The adaptation recipe expects an initialized 6k FR3 updater with new time-input columns initialized to zero, the original frames, sidecars and split. The submitted checkpoint was selected after 2,500 additional updates from a run budgeted for 4,000. The private training recordings and matching FR3 expert remain separate prerequisites.

For the adaptation initialization, run `python scripts/initialize_franka.py PATH_TO_ORIGINAL_6K_TRAINING_CHECKPOINT runs/franka-init` with the Franka import profile active, then launch `torchrun --standalone --nproc-per-node=2 scripts/train_updater.py configs/adapt_franka.json`. The sidecar preparation writes a copy of the chosen split inside its output directory. Point the recipe at that split or provide the identical file at `data/franka-split.json`.
