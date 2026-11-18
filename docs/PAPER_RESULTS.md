# Reported results

| Evaluation | Controller | Success |
| --- | --- | ---: |
| LIBERO-10 native, 500 episodes | CauSyncVLA | 469/500 (93.80%) |
| Three physical tasks, 23 trials per task and controller | Frozen π0.5 | 50/69 (72.46%) |
| Three physical tasks, 23 trials per task and controller | CauSyncVLA | 60/69 (86.96%) |

The LIBERO native headline and the separately run module ablations use distinct run cohorts. Physical success is pooled across cup stacking, fruit placement and vegetable sorting. A dispatch ceiling is a configuration parameter. Measured new-plan rate counts first use of distinct plans; new-action rate counts first dispatch of distinct action nodes.
