#!/bin/bash
# =============================================================================
# Install SageAttention 2.2.0 from source
# Detects environment (WSL vs RunPod) via WSL_DISTRO_NAME
# =============================================================================

set -euo pipefail
trap 'echo "ERROR: Command failed at line $LINENO with exit code $?" >&2; exit 1' ERR

SAGE_VERSION="2.2.0"
SAGE_REPO="https://github.com/thu-ml/SageAttention.git"
SAGE_DIR="${HOME}/SageAttention"
MAX_JOBS=4

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

sage_check_installed() {
    echo ">>> Checking installed SageAttention version..."

    INSTALLED=$(pip show sageattention 2>/dev/null | grep "^Version:" | awk '{print $2}') || true

    if [ -z "$INSTALLED" ]; then
        echo -e "${GREEN}[INFO] SageAttention not installed. Proceeding.${NC}"
    elif [ "$INSTALLED" == "${SAGE_VERSION}" ]; then
        echo -e "${GREEN}[OK] SageAttention ${SAGE_VERSION} is already installed and up to date.${NC}"
        echo ""
        echo "Nothing to do. Exiting."
        exit 0
    else
        echo -e "${YELLOW}[INFO] Found older version: ${INSTALLED}. Uninstalling...${NC}"
        python -m pip uninstall sageattention -y
        echo -e "${GREEN}[OK] Old version removed.${NC}"
    fi

    echo ""
}

sage_check_build_deps() {
    echo ">>> Checking build dependencies..."

    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}[ERROR] nvcc not found. Please install CUDA Toolkit first (install-cuda-toolkit.sh).${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK] nvcc: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')${NC}"

    if ! command -v gcc &> /dev/null; then
        echo -e "${RED}[ERROR] gcc not found. Run: sudo apt-get install build-essential${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK] gcc: $(gcc --version | head -1)${NC}"

    python -m pip install -q ninja packaging
    echo -e "${GREEN}[OK] ninja and packaging are installed.${NC}"
    echo ""
}

sage_clone() {
    echo ">>> Cloning SageAttention repository..."

    if [ -d "${SAGE_DIR}" ]; then
        echo -e "${YELLOW}[INFO] Directory ${SAGE_DIR} already exists. Removing...${NC}"
        rm -rf "${SAGE_DIR}"
    fi

    git clone "${SAGE_REPO}" "${SAGE_DIR}"
    echo -e "${GREEN}[OK] Repository cloned.${NC}"
    echo ""
}

sage_build_install() {
    echo ">>> Building SageAttention (this may take several minutes)..."
    echo "    Using MAX_JOBS=${MAX_JOBS}"
    echo ""

    cd "${SAGE_DIR}"
    MAX_JOBS=${MAX_JOBS} python -m pip install . --no-build-isolation

    echo ""
    echo -e "${GREEN}[OK] Build and installation complete.${NC}"

    echo ""
    echo ">>> Moving out of repo directory before testing..."
    cd "${HOME}"
}

sage_test() {
    echo ">>> Running SageAttention functional test..."
    echo ""

    python3 -c "
import torch
from sageattention import sageattn
import sageattention

print(f'SageAttention version: {sageattention.__version__ if hasattr(sageattention, \"__version__\") else \"2.x (no __version__ attr)\"}'
)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print()

q = torch.randn(1, 8, 1024, 64, dtype=torch.float16).cuda()
k = torch.randn(1, 8, 1024, 64, dtype=torch.float16).cuda()
v = torch.randn(1, 8, 1024, 64, dtype=torch.float16).cuda()
out = sageattn(q, k, v)
print(f'Test output shape: {out.shape}')
print('SageAttention is working correctly!')
"
}

sage_cleanup() {
    echo ""
    echo ">>> Cleaning up repository folder..."
    rm -rf "${SAGE_DIR}"
    echo -e "${GREEN}[OK] ${SAGE_DIR} removed.${NC}"
}

sage_print_success() {
    echo ""
    echo "============================================="
    echo -e "${GREEN} SageAttention ${SAGE_VERSION} installed successfully!${NC}"
    echo "============================================="
    echo ""
    echo "To use in ComfyUI, launch with:"
    echo "  python main.py --use-sage-attention"
    echo ""
}

# Detect WSL using the WSL_DISTRO_NAME variable (set by WSL on every session)
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    IS_WSL=true
else
    IS_WSL=false
fi

echo ""
echo "============================================="
if $IS_WSL; then
    echo " SageAttention ${SAGE_VERSION} Installer (WSL)"
else
    echo " SageAttention ${SAGE_VERSION} Installer (RunPod)"
fi
echo "============================================="
echo ""

if $IS_WSL; then
    echo ">>> Detected environment: WSL (${WSL_DISTRO_NAME})"
    echo ""
    echo ">>> Checking virtual environment..."

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${RED}[ERROR] No virtual environment is active.${NC}"
        echo ""
        echo "On WSL, you must activate a venv before running this script:"
        echo "  source /path/to/your/.venv/bin/activate"
        echo ""
        exit 1
    fi

    echo -e "${GREEN}[OK] venv active: ${VIRTUAL_ENV}${NC}"
else
    echo ">>> Detected environment: RunPod (system Python)"
fi

echo ">>> Python: $(which python3)"
echo ""

sage_check_installed
sage_check_build_deps
sage_clone
sage_build_install
sage_test
sage_cleanup
sage_print_success
