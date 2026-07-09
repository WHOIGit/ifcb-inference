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
| `[cuda,torch]` | Both of the above | GPU inference with the PyTorch data loader |
| `[cuda,torch,parquet,h5]` | CUDA, PyTorch, pyarrow, and h5py | Full-featured install, including Parquet and H5 output |
| `[parquet]` | pyarrow | Writing Parquet output — class scores (see [Output](#output-organization-examples)) and embedding vectors (see [Embeddings](#embeddings)) |
| `[h5]` | h5py | Writing `ifcb_classifier`-style H5 class score output |
| `[dev]` | pytest, black, isort, flake8, pyarrow, h5py | Development and testing |

- One of `[cpu]` or `[cuda]` must be used to have the appropriate onnxruntime. They are mutually exclusive. If neither are included, at install, `ifcb-infer` will be unable to run. If in doubt, use `[cuda]`.
- Use of `[torch]` is optional. Without it, a basic data loader is used - suitable for constrained or lite environments where installing PyTorch is impractical (e.g. small containers, edge deployments). The `[torch]` data loader is otherwise recommended as it supports more image formats and is generally faster. If `[torch]` is used, to avoid `cudnn` version conflicts from `[cuda]` (aka `onnxruntime[cuda,cudnn]`), and/or lighten install, pip commmand should include `--extra-index-url https://download.pytorch.org/whl/cpu`; this forces the cpu version of pytorch, which is fine since we're only using it for its dataloaders. 

```bash
# Full featured install
pip install "ifcb-infer[cuda,torch,parquet,h5] @ git+https://github.com/WHOIGit/ifcb-inference.git" --extra-index-url https://download.pytorch.org/whl/cpu

# GPU enabled, but without pytorch dependencies
pip install "ifcb-infer[cuda] @ git+https://github.com/WHOIGit/ifcb-inference.git"
export LD_LIBRARY_PATH=$(pip show nvidia-cudnn-cu12 | grep Location | awk '{print $2}')/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# see "cuDNN requirement for `[cuda]` without `[torch]`" LD_LIBRARY_PATH note below

# Lightest install
pip install "ifcb-infer[cpu] @ git+https://github.com/WHOIGit/ifcb-inference.git"
```

If cloning the repo and developing locally:
```bash
# Full-featured install (gpu/CUDA + PyTorch + Parquet + H5)
pip install -e ".[cuda,torch,parquet,h5,dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### cuDNN requirement for `[cuda]` without `[torch]`

`[cuda,torch] --extra-index-url https://download.pytorch.org/whl/cpu` works out of the box. CUDNN gets bundled with `[cuda']` (ie `onnxrutime[cuda,cudnn]`) and not `torch`. To avoid conflict `--extra-index-url https://download.pytorch.org/whl/cpu` is used such that pytorch (which is primarily used for its dataloaders) doesn't clobber onnxruntime's CUDNN libs.

`[cuda]` alone installs `nvidia-cudnn-cu12` via pip, but ORT cannot find it without help because the libraries land in `site-packages`, not a standard system path. If you don't have libcudnn9-cuda-12 installed globally/to a standard location, it must be explicitely set with `LD_LIBRARY_PATH`. 

Setting `LD_LIBRARY_PATH` to point to the pip-installed cuDNN:**
```bash
export LD_LIBRARY_PATH=$(pip show nvidia-cudnn-cu12 | grep Location | awk '{print $2}')/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
Add this to your environment profile (`.bashrc`, `.bash_profile`, venv/bin/activate script) to make it persistent.

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
                                       A .parquet extension writes scores as Parquet;
                                       a .h5 extension writes ifcb_classifier-style H5;
                                       otherwise CSV.
--cpuonly                              Force CPU inference even if CUDA is available
--notorch                              Use non-PyTorch data loader even if torch is installed
--embeddings                           Also emit penultimate-layer embedding vectors (see Embeddings)
--embeddings-only                      Emit only embeddings, skip the score CSV (implies --embeddings)
--embeddings-outfile PATTERN           Embedding filename pattern. Same tokens as --outfile.
                                       Default: {MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet
```

- By default, CUDA is used automatically when available/installed and otherwise falls back to using CPU.
- By default, torch-dataloaders are used automatically when available/installed and otherwise falls back to a simpler implementation.
- For the output csv to have column names that correspond to human-readable class names, use `--classes` option.
- Give `--outfile` a `.parquet` extension to write scores as Parquet (requires the `[parquet]` extra). Schema: a `pid` string column plus one `float32` column per class (class names from `--classes`, else `score_0`, `score_1`, …).
- Give `--outfile` a `.h5` extension to write `ifcb_classifier`-style H5 score files (requires the `[h5]` extra and `--classes`). The H5 contains `metadata` attrs (`version`, `model_id`, `timestamp`, `bin_id`) plus `output_classes`, `output_scores`, `class_labels`, and `roi_numbers` datasets.
- Any other `--outfile` extension writes CSV as before.
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

## Embeddings

In addition to class scores, `ifcb-infer` can emit the CNN's **penultimate-layer embedding** — the global-pooled feature vector that feeds the classification head. No retraining is needed: the embedding is an intermediate activation the trained model already computes on every forward pass; it just needs to be surfaced as a model output.

This is a two-step workflow:

**1. One-time graph surgery.** ONNX Runtime only returns tensors declared in the model's graph outputs. Add the embedding tensor as a second output:

```bash
python -m ifcb_infer.add_embedding_output classifier.onnx classifier_emb.onnx
```

The embedding tensor is auto-detected as the data input of the final `Gemm`/`MatMul` (the classification head). For a non-standard architecture, override it with `--tensor-name`. The resulting model returns `[scores, embedding]` from one forward pass and is otherwise identical to the original.

**2. Run inference with `--embeddings`** against the surgically-modified model:

```bash
# install the extra once: pip install -e ".[parquet]"
ifcb-infer --embeddings --classes classes.txt classifier_emb.onnx example-data/bins/
```

Each bin gets, alongside its `.csv` of scores, an `.emb.parquet` file with one row per ROI:

| Column | Type | Notes |
|---|---|---|
| `pid` | string | ROI identifier (aligned with the score CSV) |
| `embedding` | `fixed_size_list<float16>` | the feature vector (e.g. length 2048 for InceptionV3) |

Embeddings are stored at **float16** to halve on-disk size — ample precision for similarity, clustering, and visualization. The output path follows `--embeddings-outfile` (same tokens as `--outfile`). Use `--embeddings-only` to skip writing the score CSV. Running `--embeddings` against an unmodified (single-output) model raises an error pointing back to step 1.

## Container Use

The default Docker image installs with `[cuda,torch]` for GPU support and the PyTorch data loader. The GitHub workflow also publishes a separate embeddings image with `[cuda,torch,parquet,h5]` for Parquet output (class scores and embeddings) and H5 score output:

| Image | Extras |
|---|---|
| `ghcr.io/WHOIGit/ifcb-inference:latest` | `[cuda,torch]` |
| `ghcr.io/WHOIGit/ifcb-inference-embeddings:latest` | `[cuda,torch,parquet,h5]` |

Build:
```bash
# Podman
podman build . -t ifcb-infer:latest

# Embeddings image
podman build . \
       --build-arg IFCB_INFER_EXTRAS=cuda,torch,parquet,h5 \
       -t ifcb-infer-embeddings:latest

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

First install with the `[cpu,dev]` extras:

```bash
pip install -e ".[cpu,dev]"
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
