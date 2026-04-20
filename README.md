# amplify_onnx_inference

![Tests](https://github.com/WHOIGit/amplify_onnx_inference/workflows/Tests/badge.svg)
![Lint](https://github.com/WHOIGit/amplify_onnx_inference/workflows/Lint/badge.svg)

ONNX-based inference system for IFCB (Imaging FlowCytobot) data analysis. This tool performs automated plankton classification on IFCB bin files using pre-trained ONNX models.

## Features

- **Flexible model support**: Works with both static and dynamic batch size ONNX models
- **Multiple data loading backends**: Supports both PyTorch and non-PyTorch data loading
- **Configurable output organization**: Choose between run-date or model-name subfolder organization
- **Directory structure preservation**: Maintains input directory hierarchies in output
- **Containerized deployment**: Docker/Podman support for consistent environments
- **GPU acceleration**: CUDA support for faster inference (automatic when available)

## Installation

```bash
# CPU-only (default)
pip install -e .

# With CUDA/GPU support
pip install -e ".[cuda]"

# With PyTorch data loaders
pip install -e ".[torch]"

# All of the above
pip install -e ".[cuda,torch]"

# Development (includes pytest, black, isort, flake8)
pip install -e ".[dev]"
```

To include in a `requirements.txt`:

```
# CPU-only
ifcb-infer @ git+https://github.com/WHOIGit/amplify_onnx_inference.git

# With extras (e.g. CUDA + PyTorch)
ifcb-infer[cuda,torch] @ git+https://github.com/WHOIGit/amplify_onnx_inference.git
```

## Usage

```bash
ifcb-infer [OPTIONS] MODEL BINS [BINS ...]
```

`BINS` can be a directory, a bin path, or a `.txt`/`.list` file of bin paths.

### Options

```
--batch N                              Required for models without a fixed input batch size
--classes FILE                         Class list file; adds column headers to output CSVs.
                                       Accepts a line-delimited .txt or an index-keyed .json
                                       (e.g. {"0": "class_a", "1": "class_b"})
--outdir DIRPATH                       Output directory. Default: ./outputs
--outfile PATTERN                      Output filename pattern. Default: {MODEL_NAME}/{SUBPATH}.csv
                                       Tokens: {MODEL_NAME}, {RUN_DATE}, {SUBPATH}
--cpuonly                              Force CPU inference even if CUDA is available
--notorch                              Use non-PyTorch data loader even if torch is installed
```

By default, CUDA is used automatically when available and falls back to CPU otherwise.

### Output Organization Examples

**Model-name organization (default):**
```
outputs/
├── my_classifier/
│   ├── D20240301T123456_IFCB999.csv
│   └── D20240301T130000_IFCB999.csv
└── another_model/
    └── D20240302T090000_IFCB999.csv
```

**Run-date organization (`--outfile "{RUN_DATE}/{SUBPATH}.csv"`):**
```
outputs/
├── 2025-01-15/
│   ├── D20240301T123456_IFCB999.csv
│   └── D20240301T130000_IFCB999.csv
└── 2025-01-16/
    └── D20240302T090000_IFCB999.csv
```

## Container Use

```bash
podman build . -t onnx:latest
podman run -it --rm -e CUDA_VISIBLE_DEVICES=1 \
       --device nvidia.com/gpu=all \
       -v $(pwd)/models:/app/models \
       -v $(pwd)/inputs/:/app/inputs \
       -v $(pwd)/outputs:/app/outputs \
       onnx:latest models/PathToYourModel.onnx inputs/PathToBinDirectory
```

## Development

### Running Tests

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
