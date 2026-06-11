#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dino-sam-geco2}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -n "$ENV_NAME" -f environment.yml
else
  conda activate "$ENV_NAME"
  python -m pip install -r requirements.txt
fi

conda activate "$ENV_NAME"

# Install the vendored SAM3 package used by infer.py when --sam3-mask is enabled.
python -m pip install -e "$PROJECT_ROOT/src/models/backbones/sam3"

# Build the Deformable DETR CUDA extension. This requires a CUDA toolkit with nvcc.
if command -v nvcc >/dev/null 2>&1; then
  pushd "$PROJECT_ROOT/src/models/DeformableDETR/models/ops" >/dev/null
  python setup.py build install
  popd >/dev/null
else
  echo "Warning: nvcc was not found; skipped Deformable DETR CUDA extension build." >&2
  echo "Install a CUDA toolkit and rerun this script if your run imports MSDeformAttn." >&2
fi

echo "Environment '$ENV_NAME' is ready."
