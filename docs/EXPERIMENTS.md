# Experiment settings

| Profile | Base | Visual cache | Updater | Chunk H/C | Expert |
| --- | --- | --- | --- | --- | --- |
| LIBERO-10 | π0.5-LIBERO, state tokens disabled | Threaded real-prefix publication | Step 12,000; history 48; 32 compact text slots | 10/5 | 1 step, PFWS restart 0.5 |
| Franka FR3 | Lab-finetuned π0.5, state tokens enabled | Latest completed real prefix | Step 8,500; history 24; measured time intervals | 10/2 | 1 step, PFWS restart 0.5 |

**Training.** The simulation recipes specify two DDP ranks, two anchor snapshots per microbatch, four targets per snapshot and two gradient accumulation steps. The effective batch is 32. Recorded actions at indices [a,t) are paired with measured states at (a,t]. The real cache is replaced only on publication of a completed visual pass. Simulated and physical actions use their respective normalization assets.

**Evaluation.** The native LIBERO run uses seed 7, ten tasks, fifty initial states per task and an upper bound of 520 control steps. The timed protocol uses a separate scheduler. Record GPU sharing, rendering placement, and whether timestamps refer to observation, plan readiness or first command dispatch when comparing rates.

**Data access.** The release supplies the method code, the LIBERO updater checkpoint and portable recipes. The FR3 updater, its matching expert and the training recordings are privately held. The four expert sources in the LIBERO recipe are groups of 600/300/300/300 training trajectories. Use a training-only split for fitted normalization.
