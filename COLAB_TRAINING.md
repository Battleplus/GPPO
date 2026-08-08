# Colab training protocol

This workflow answers two different questions in order:

1. Does Colab GPU reduce **end-to-end** training time for this simulator?
2. Does the paper-faithful GPPO architecture pass the pre-registered mechanism checks?

The model is small and rollout collection is sequential, so a fast GPU is not automatically faster. Google also states that Colab GPU types and limits vary over time. Use the included benchmark instead of assuming that a premium GPU is necessary.

## 1. Create the runtime

Open Colab, select a GPU runtime only for the benchmark, and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone the branch and enter the repository:

```bash
!git clone --branch '8.8-GPPO无偏好' --single-branch https://github.com/Battleplus/GPPO.git /content/GPPO
%cd /content/GPPO
```

Do not put a GitHub token in a notebook. If the repository becomes private, use a Colab secret or upload a source archive.

## 2. Smoke test

```bash
!OUTPUT_ROOT=/content/drive/MyDrive/GPPO_colab \
  DEVICE=auto bash colab/run_colab_validation.sh smoke
```

This checks the formulas/resume tests and performs a two-iteration train/save run.

## 3. Decide whether GPU is necessary

```bash
!OUTPUT_ROOT=/content/drive/MyDrive/GPPO_colab \
  bash colab/run_colab_validation.sh benchmark
```

Read `MyDrive/GPPO_colab/DEVICE_BENCHMARK.json`. Use CUDA only if `recommended_device` is `cuda`; the frozen rule requires at least 1.15x end-to-end speedup. Otherwise switch Colab back to a standard CPU runtime and use `DEVICE=cpu`. GPU utilization alone is not a valid reason to keep paying for a GPU.

## 4. Single-seed mechanism gate

```bash
!OUTPUT_ROOT=/content/drive/MyDrive/GPPO_colab \
  DEVICE=cpu JOBS=2 bash colab/run_colab_validation.sh quick
```

For CUDA, use `DEVICE=cuda JOBS=1`. The output report is:

```text
MyDrive/GPPO_colab/quick_seed1_100/QUICK_MECHANISM_REPORT_ZH.md
```

Continue only if all of the following are true:

- GPPO-event has lower realized makespan than PPO-none and PPO-event.
- GPPO-event has lower realized makespan than Random.
- Adaptive is not clearly worse than both NoGate and SingleHead.
- Event is non-inferior to Full while reducing communication bytes.

This gate is directional evidence only, not a paper-level result.

## 5. Five-seed formal scale

Run one scale at a time. Re-running the same command resumes incomplete jobs from `resume_latest.pt` and skips completed checkpoints.

```bash
!OUTPUT_ROOT=/content/drive/MyDrive/GPPO_colab \
  SCALE=T5-10-48 DEVICE=cpu JOBS=2 \
  bash colab/run_colab_validation.sh formal-scale
```

Repeat with `T10-10-53`, `T15-8-66`, and `T20-10-92`. Use `DEVICE=cuda JOBS=1` only when the benchmark recommends CUDA.

## 6. Final matrix and interpretation

After all scales are complete:

```bash
!OUTPUT_ROOT=/content/drive/MyDrive/GPPO_colab \
  DEVICE=cpu JOBS=2 bash colab/run_colab_validation.sh formal-all
```

The run contains five training seeds for:

- PPO-none and PPO-event;
- GPPO-none and GPPO-event;
- GPPO-NoGate-event and GPPO-SingleHead-event;
- GPPO-periodic and GPPO-always.

The formal claim is supported only when seed-level confidence intervals show the graph model is stable across scales and the adaptive/event components pass their own controls. If adaptive does not beat NoGate/SingleHead, report that the paper's adaptive contribution was not independently reproduced; do not tune until the desired ordering appears.

## Practical recommendation

Start with the free/standard runtime. A 4090-class GPU is unlikely to be necessary for correctness and may not be faster because environment stepping and batch-1 policy inference dominate. The valuable part of Colab is uninterrupted clean reruns and portable checkpoints, not the GPU label itself.
