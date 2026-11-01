#!/bin/bash
# CPU-queue setup job: downloads + env build. No GPU.
set -x
P=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg
CONDA=/data/rtxw01/nbak/robot_intel/kztrgg/miniconda3/bin/conda
export CONDA_PKGS_DIRS=/data/rtxw01/nbak/robot_intel/kztrgg/conda_pkgs
export PIP_CACHE_DIR=/data/rtxw01/nbak/robot_intel/kztrgg/pip_cache
export HF_HOME=/data/rtxw01/nbak/robot_intel/hf_cache
mkdir -p $P/third_party $P/data/raw $P/weights $P/logs
cd $P

# 1. dataset (plain git, ~1 GB)
[ -d data/raw/dataset-ErrP-HRI/.git ] || git clone https://github.com/stefan-ehrlich/dataset-ErrP-HRI.git data/raw/dataset-ErrP-HRI
git -C data/raw/dataset-ErrP-HRI rev-parse HEAD > data/raw/dataset_commit.txt
git -C data/raw/dataset-ErrP-HRI lfs ls-files 2>/dev/null | head
ls -la data/raw/dataset-ErrP-HRI/robot | head -40
# read-only raw data
find data/raw/dataset-ErrP-HRI -type f \( -name '*.set' -o -name '*.fdt' -o -name '*.pdf' \) -exec chmod a-w {} +

# 2. official model repos
[ -d third_party/CBraMod/.git ] || git clone https://github.com/wjq-learning/CBraMod.git third_party/CBraMod
[ -d third_party/arl-eegmodels/.git ] || git clone https://github.com/vlawhern/arl-eegmodels.git third_party/arl-eegmodels
for r in CBraMod arl-eegmodels; do echo "$r $(git -C third_party/$r rev-parse HEAD)"; done > third_party/commits.txt
cat third_party/commits.txt

# 3. env
if [ ! -x $P/env/bin/python ]; then
  $CONDA create -y -p $P/env python=3.11 pip
fi
PY=$P/env/bin/python
$PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install "numpy<2.3" scipy pandas scikit-learn mne einops pyyaml huggingface_hub safetensors \
    matplotlib pypdf tqdm h5py
$PY -m pip install braindecode
$PY -m pip freeze > $P/logs/pip_freeze_setup.txt
$PY -c "import torch,mne,braindecode,numpy,sklearn;print(torch.__version__,mne.__version__,braindecode.__version__,numpy.__version__,sklearn.__version__)"

# 4. CBraMod pretrained weights
$PY - <<'EOF'
from huggingface_hub import snapshot_download, HfApi
import json
api = HfApi()
info = api.model_info("weighting666/CBraMod", files_metadata=True)
print("sha", info.sha)
for s in info.siblings: print(s.rfilename, s.size)
p = snapshot_download("weighting666/CBraMod", revision=info.sha, local_dir="/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg/weights/CBraMod")
json.dump({"repo":"weighting666/CBraMod","revision":info.sha,"files":[[s.rfilename,s.size] for s in info.siblings]}, open(p+"/_hf_revision.json","w"), indent=1)
EOF
sha256sum $P/weights/CBraMod/* 2>/dev/null
echo SETUP_DONE
