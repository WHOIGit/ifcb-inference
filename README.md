# ifcb-inference

![Tests](https://github.com/WHOIGit/ifcb-inference/workflows/Tests/badge.svg)
![Lint](https://github.com/WHOIGit/ifcb-inference/workflows/Lint/badge.svg)

ONNX-based inference system for IFCB (Imaging FlowCytobot) bin data. This tool performs automated plankton classification on IFCB bin files using pre-trained ONNX models.

## Features

- **Flexible model support**: Works with both static and dynamic batch size ONNX models
- **Multiple data loading backends**: Supports both PyTorch and non-PyTorch data loading
- **Configurable output organization**: Choose between run-date or model-name subfolder organization
- **Directory structure preservation**: Maintains input directory hierarchies in output
- **Containerized deployment**: Docker/Podman support for consistent environments
- **GPU acceleration**: CUDA support for faster inference (automatic when available)

## Installation

| Extra | Installs | Use when |
|---|---|---|
| `[cpu]` | `onnxruntime` (CPU) | Lightweight/constrained environments — no GPU |
| `[cuda]` | `onnxruntime-gpu` | GPU inference via CUDA |
| `[torch]` | PyTorch + torchvision | Faster/more flexible data loading, but more dependancies |
| `[cuda,torch]` | Both of the above | Full-featured install |
| `[dev]` | pytest, black, isort, flake8 | Development and testing |

- One of `[cpu]` or `['cuda']` must be used to have the appropriate onnxruntime. They are mutually exclusive. If neither are included, at install, `ifcb-infer` will be unable to run. If in doubt, use `[cuda]`.
- Use of `[torch]` is optional. Without it, a basic data loader is used — suitable for constrained or lite environments where installing PyTorch is impractical (e.g. small containers, edge deployments). The `[torch]` data loader is recommended otherwise as it supports more image formats and is generally faster.

```bash
# Full featured install
pip install "ifcb-infer[cuda,torch] @ git+https://github.com/WHOIGit/ifcb-inference.git"

# Lightest install
pip install "ifcb-infer[cpu] @ git+https://github.com/WHOIGit/ifcb-inference.git"
```

If cloning the repo and developing locally:
```bash
# Full-featured install (gpu/CUDA + PyTorch)
pip install -e ".[cuda,torch,dev]"
```

## Usage

```bash
ifcb-infer [OPTIONS] MODEL BINS [BINS ...]
```
`MODEL` is the path to an onnx model file
`BINS` can be a directory, a bin path, or a `.txt`/`.list` file of bin paths.

### Options

```
--classes FILE                         Class list file; adds column headers to output CSVs.
                                       Accepts a line-delimited .txt or an index-keyed .json
                                       (e.g. {"0": "class_a", "1": "class_b"})
--batch N                              Required for models without a fixed input batch size
--outdir DIRPATH                       Output directory. Default: ./outputs
--outfile PATTERN                      Output filename pattern. Default: {MODEL_NAME}/{SUBPATH}/{BIN}.csv
                                       Tokens: {MODEL_NAME}, {RUN_DATE}, {SUBPATH} (relative dir), {BIN} (bin name)
--cpuonly                              Force CPU inference even if CUDA is available
--notorch                              Use non-PyTorch data loader even if torch is installed
```

- By default, CUDA is used automatically when available and falls back to CPU otherwise.
- By default, torch-dataloaders are used automatically when available and otherwise falls back to a simpler implementation
- For the output csv to have column names that correspond to human-readable class names, use `--classes` option
- If a model has a predefined input batch size, that batch size is automatically used and `--batch` is ignored. 
- If a model does NOT have a predefined input batch size, `--batch` must be specified.

### Output Organization Examples

The output path for each bin is controlled by the `--outfile PATTERN` option (default: `{MODEL_NAME}/{SUBPATH}/{BIN}.csv`), resolved relative to `--outdir`. The available tokens are:

| Token | Value |
|---|---|
| `{BIN}` | Bin name (e.g. `D20230108T145350_IFCB127`) |
| `{SUBPATH}` | Directory of the bin relative to the input folder |
| `{MODEL_NAME}` | Model filename without extension |
| `{RUN_DATE}` | Date the command was run (`YYYY-MM-DD`) |

`{SUBPATH}` mirrors the input directory hierarchy, so outputs reflect the same structure as the source data. Given:

```
example-data/bins/
├── MVCO/
│   ├── 2006/
│   │   └── IFCB1_2006_157/
│   │       ├── IFCB1_2006_157_181359   ← bin
│   │       ├── IFCB1_2006_157_183432   ← bin
│   │       └── IFCB1_2006_157_185616   ← bin
│   └── 2023/
│       └── D20230108/
│           ├── D20230108T145350_IFCB127   ← bin
│           ├── D20230108T151529_IFCB127   ← bin
│           └── D20230108T153615_IFCB127   ← bin
└── OTZ/
    └── 2019/
        ├── D20190722/
        │   └── D20190722T155753_IFCB127   ← bin
        └── D20190723/
            ├── D20190723T161602_IFCB127   ← bin
            └── D20190723T171832_IFCB127   ← bin
```

**Default (`{MODEL_NAME}/{SUBPATH}/{BIN}.csv`):**
```bash
ifcb-infer my_classifier.onnx example-data/bins/
```
```
outputs/
└── my_classifier/
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_181359.csv
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_183432.csv
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_185616.csv
    ├── MVCO/2023/D20230108/D20230108T145350_IFCB127.csv
    ├── MVCO/2023/D20230108/D20230108T151529_IFCB127.csv
    ├── MVCO/2023/D20230108/D20230108T153615_IFCB127.csv
    ├── OTZ/2019/D20190722/D20190722T155753_IFCB127.csv
    ├── OTZ/2019/D20190723/D20190723T161602_IFCB127.csv
    └── OTZ/2019/D20190723/D20190723T171832_IFCB127.csv
```

**Flat output — one folder, all bins (`--outfile "{BIN}.csv"`):**
```bash
ifcb-infer --outdir "my/custom/output" --outfile "{BIN}.csv" my_classifier.onnx example-data/bins/
```
```
my/custom/output/
├── IFCB1_2006_157_181359.csv
├── IFCB1_2006_157_183432.csv
├── IFCB1_2006_157_185616.csv
├── D20230108T145350_IFCB127.csv
├── D20230108T151529_IFCB127.csv
├── D20230108T153615_IFCB127.csv
├── D20190722T155753_IFCB127.csv
├── D20190723T161602_IFCB127.csv
└── D20190723T171832_IFCB127.csv
```

**Run-date prefix (`--outfile "{RUN_DATE}/{SUBPATH}/{BIN}.csv"`):**
```bash
ifcb-infer --outfile "{RUN_DATE}/{SUBPATH}/{BIN}.csv" my_classifier.onnx example-data/bins/
```
```
outputs/
└── 2025-01-15/
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_181359.csv
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_183432.csv
    ├── MVCO/2006/IFCB1_2006_157/IFCB1_2006_157_185616.csv
    ├── MVCO/2023/D20230108/D20230108T145350_IFCB127.csv
    ├── MVCO/2023/D20230108/D20230108T151529_IFCB127.csv
    ├── MVCO/2023/D20230108/D20230108T153615_IFCB127.csv
    ├── OTZ/2019/D20190722/D20190722T155753_IFCB127.csv
    ├── OTZ/2019/D20190723/D20190723T161602_IFCB127.csv
    └── OTZ/2019/D20190723/D20190723T171832_IFCB127.csv
```

## Container Use

The Dockerfile installs with `[cuda,torch]` for full GPU support.

Build:
```bash
# Podman
podman build . -t ifcb-infer:latest
podman run -it --rm -e CUDA_VISIBLE_DEVICES=1 \
       --device nvidia.com/gpu=all \
       -v $(pwd)/models:/app/models \
       -v $(pwd)/inputs:/app/inputs \
       -v $(pwd)/outputs:/app/outputs \
       ifcb-infer:latest models/classifier.onnx inputs/
```

To select a specific GPU, use `CUDA_VISIBLE_DEVICES`:

All `ifcb-infer` options can be appended after the image name. 
`MODEL` and `BINS` paths must refer to paths _inside_ the container as mapped by `-v`.


## Development

### Running Tests

First install with the `[dev]` extra:

```bash
pip install -e ".[dev]"
```

Then run:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

### Continuous Integration

The project includes GitHub Actions workflows that automatically:

- **Run tests** on Python 3.10, 3.11, and 3.12 when code is pushed or PRs are opened
- **Check code quality** with linting tools (flake8, black, isort)

Tests run automatically on pushes to `main` branch and on all pull requests.
