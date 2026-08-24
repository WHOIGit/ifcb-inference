# Use slim Python base image
FROM python:3.12-slim

# Set environment variables to prevent .pyc files and set UTF-8 encoding
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

ARG IFCB_INFER_EXTRAS=cuda,torch

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src

# Install git for ifcbkit git dependency, install package, then remove git to save space
RUN apt-get update && \
    apt-get install -y git && \
    pip install ".[${IFCB_INFER_EXTRAS}]" --extra-index-url https://download.pytorch.org/whl/cpu && \
    apt-get remove -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Register the CUDA/cuDNN shared libraries shipped by the nvidia-* pip wheels.
#
# Those wheels install their .so files under site-packages/nvidia/*/lib, which the
# dynamic linker does not search. Without this, libonnxruntime_providers_cuda.so
# fails to load ("libcublasLt.so.13: cannot open shared object file") and
# onnxruntime falls back to CPU. The fallback is quiet: it logs a warning to
# stderr, and ort.get_available_providers() still lists CUDAExecutionProvider,
# because that reports what was compiled in rather than what can load. Only
# InferenceSession(...).get_providers() reveals the real state.
#
# site-packages is resolved at build time and the glob covers every nvidia
# subpackage (currently cu13 and cudnn), so a Python or CUDA version bump does not
# silently drop us back to CPU. Building with the cpu extra produces an empty
# file, which ldconfig ignores.
RUN python -c "import site, glob, os; print('\n'.join(d for p in site.getsitepackages() for d in glob.glob(os.path.join(p, 'nvidia', '*', 'lib'))))" \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && cat /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

ENTRYPOINT ["sh", "-c", "umask 0002 && exec ifcb-infer \"$@\"", "--"]
