# D4RT-Lab

D4RT-Lab is an independent, research-oriented implementation of D4RT built
around Hydra and Lightning. It provides a typed model implementation, a
Lightning training system, structured data pipelines, and a small API for
encoding videos and evaluating point queries.

This project is not an official implementation from the D4RT paper authors.
It is a substantial engineering refactor of the public Open-D4RT reproduction.
See [Acknowledgements](#acknowledgements) before redistributing or publishing
results produced with this codebase.

## Project Structure

```text
d4rt_lab/
├── api/             # Video encoding and downstream query interface
├── config/          # Root Hydra schema and config registration
├── core/            # Cross-module tensor contracts
├── data/            # Datasets, sampling, augmentation, and DataModule
├── loss/            # D4RT training objectives
├── model/           # Pure model and model-weight lifecycle
└── system/          # LightningModule, experiment composition, and callbacks
configs/             # Hydra configuration groups
notebooks/           # Interactive examples
```

The installable distribution is named `d4rt-lab`. Python uses underscores in
module names, so imports use `d4rt_lab`.

## Installation

The project targets Python 3.10 and uses `uv` for environment management.

```bash
uv sync
```

Model weights and datasets are not downloaded automatically. Configure their
paths through the Hydra groups under `configs/`.

## Training

Inspect the fully composed training configuration without starting a run:

```bash
uv run d4rt-train --cfg job
```

Start the default experiment:

```bash
uv run d4rt-train
```

Hydra overrides can change individual values or entire configuration groups:

```bash
uv run d4rt-train \
  initialization=videomaev2_giant \
  runtime.devices=1 \
  schedule.total_steps=1000
```

The initialization source is explicit:

```yaml
initialization:
  source: artifact            # random | artifact | encoder_pretrained
  artifact_path: checkpoints/path/to/model.pt
```

Lightning training can be resumed independently:

```bash
uv run d4rt-train \
  launch.resume_checkpoint_path=output/experiment/checkpoints/last.ckpt
```

## Video API

Load a D4RT model artifact, encode an MP4, and query one pixel across frames:

```python
import torch

from d4rt_lab.api import D4RTAPI

api = D4RTAPI.from_checkpoint(
    "checkpoints/path/to/model.pt",
    device="cuda",
)
api.eval()

with torch.inference_mode():
    encoded = api.encode_path("video.mp4")
    output = api.query(
        encoded,
        uv=[[320.0, 180.0]],
        t_src=0,
        t_tgt=1,
        t_cam=0,
        coordinates="pixels",
    )
    display = api.for_display(output, encoded.original_size)
```

`encode_path()` decodes only the leading number of frames required by the
model configuration. Audio streams are ignored. For preprocessed tensors, use
`encode_tensor()` and `decode()` directly.

## Configuration

Configuration is distributed by ownership:

- model architecture: `d4rt_lab/model/config.py`
- model initialization: `d4rt_lab/model/weights/config.py`
- data and augmentation: `d4rt_lab/data/config.py`
- loss: `d4rt_lab/loss/config.py`
- training execution: `d4rt_lab/system/config.py`
- root composition: `d4rt_lab/config/schema.py`

YAML configuration groups remain under `configs/`. The root `D4RTConfig`
provides the structured boundary between Hydra and the runtime system.

## Design Notes

- [Loss design and formulas](docs/loss_design.md)

## Acknowledgements

D4RT-Lab builds on two separate bodies of work:

1. The D4RT CVPR paper, which introduced the underlying method.
2. The public Open-D4RT reproduction, from which this repository was
   refactored and from which compatible training recipes or weights may be
   derived.

The canonical paper citation, upstream repository URL, author list, and their
licenses must be copied from the original sources before this repository is
published. They are intentionally not guessed here because the current local
checkout contains no Git remote, citation file, license, or previous README.

This repository is not affiliated with or endorsed by the D4RT paper authors
or the Open-D4RT maintainers.
