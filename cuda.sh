#!/bin/bash

GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
BOLD="\033[1m"
NC="\033[0m"

CPU_ONLY="false"
CUDA_INSTALLED=false
NVCC_PATH=""
CUDA_PATH=""
CUDA_VERSION=""
DRIVER_VERSION=""

detect_environment() {
    IS_WSL=false
    IS_RENTED_SERVER=false
    
    if grep -q Microsoft /proc/version 2>/dev/null; then
        echo -e "${YELLOW}${BOLD}[!] WSL environment detected${NC}"
        IS_WSL=true
    fi
    
    if [ -d "/opt/deeplearning" ] || [ -d "/opt/aws" ] || [ -d "/opt/cloud" ] || [ -f "/.dockerenv" ]; then
        echo -e "${YELLOW}${BOLD}[!] Rented/Cloud server environment detected${NC}"
        IS_RENTED_SERVER=true
    fi
    
    UBUNTU_VERSION=""
    if [ -f /etc/lsb-release ]; then
        source /etc/lsb-release
        UBUNTU_VERSION=$DISTRIB_RELEASE
    elif [ -f /etc/os-release ]; then
        source /etc/os-release
        UBUNTU_VERSION=$(echo $VERSION_ID | tr -d '"')
    elif [ -f /etc/issue ]; then
        UBUNTU_VERSION=$(cat /etc/issue | grep -oP 'Ubuntu \K[0-9]+\.[0-9]+' | head -1)
    fi
    
    if [ -z "$UBUNTU_VERSION" ]; then
        if command -v lsb_release >/dev/null 2>&1; then
            UBUNTU_VERSION=$(lsb_release -rs)
        else
            apt-get update >/dev/null 2>&1
            apt-get install -y lsb-release >/dev/null 2>&1
            if command -v lsb_release >/dev/null 2>&1; then
                UBUNTU_VERSION=$(lsb_release -rs)
            else
                UBUNTU_VERSION="22.04"
            fi
        fi
    fi
    
    echo -e "${CYAN}${BOLD}[✓] System: Ubuntu ${UBUNTU_VERSION}, Architecture: $(uname -m)${NC}"
}

detect_gpu() {
    echo -e "\n${CYAN}${BOLD}[✓] Detecting NVIDIA GPU...${NC}"
    
    GPU_AVAILABLE=false
    
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo -e "${GREEN}${BOLD}[✓] NVIDIA GPU detected (via nvidia-smi)${NC}"
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo -e "${GREEN}${BOLD}[✓] NVIDIA driver version: ${DRIVER_VERSION}${NC}"
        
        # Get CUDA version directly from nvidia-smi
        DRIVER_CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null)
        if [ -n "$DRIVER_CUDA_VERSION" ]; then
            echo -e "${GREEN}${BOLD}[✓] NVIDIA driver supports CUDA ${DRIVER_CUDA_VERSION}${NC}"
        fi
        
        GPU_AVAILABLE=true
        return 0
    fi
    
    if command -v lspci &> /dev/null && lspci | grep -i nvidia &> /dev/null; then
        echo -e "${GREEN}${BOLD}[✓] NVIDIA GPU detected (via lspci)${NC}"
        GPU_AVAILABLE=true
        return 0
    fi
    
    if [ -d "/proc/driver/nvidia" ] || [ -d "/dev/nvidia0" ]; then
        echo -e "${GREEN}${BOLD}[✓] NVIDIA GPU detected (via system directories)${NC}"
        GPU_AVAILABLE=true
        return 0
    fi
    
    if [ "$IS_RENTED_SERVER" = true ]; then
        echo -e "${YELLOW}${BOLD}[!] Running on a cloud/rented server, assuming GPU is available${NC}"
        GPU_AVAILABLE=true
        return 0
    fi
    
    if [ "$IS_WSL" = true ] && grep -q "nvidia" /mnt/c/Windows/System32/drivers/etc/hosts 2>/dev/null; then
        echo -e "${YELLOW}${BOLD}[!] WSL environment with potential NVIDIA drivers on Windows host${NC}"
        GPU_AVAILABLE=true
        return 0
    fi
    
    echo -e "${YELLOW}${BOLD}[!] No NVIDIA GPU detected - using CPU-only mode${NC}"
    CPU_ONLY="true"
    return 1
}

detect_cuda() {
    echo -e "\n${CYAN}${BOLD}[✓] Checking for CUDA installation...${NC}"
    
    CUDA_AVAILABLE=false
    NVCC_AVAILABLE=false
    CUDA_INSTALLED=false
    
    # First check for CUDA in common locations
    for cuda_dir in /usr/local/cuda* /usr/local/cuda; do
        if [ -d "$cuda_dir" ] && [ -d "$cuda_dir/bin" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
            CUDA_PATH=$cuda_dir
            NVCC_PATH="$cuda_dir/bin/nvcc"
            
            if [ -x "$NVCC_PATH" ]; then
                CUDA_VERSION=$($NVCC_PATH --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1)
                [ -z "$CUDA_VERSION" ] && CUDA_VERSION=$(echo $cuda_dir | grep -oP 'cuda-\K[0-9.]+' || echo $(echo $cuda_dir | grep -oP 'cuda\K[0-9.]+'))
                echo -e "${GREEN}${BOLD}[✓] CUDA detected at ${CUDA_PATH} (version ${CUDA_VERSION})${NC}"
                CUDA_AVAILABLE=true
                CUDA_INSTALLED=true
                break
            fi
        fi
    done
    
    # If CUDA wasn't found in standard locations but nvcc is in PATH
    if [ "$CUDA_INSTALLED" = false ] && command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_PATH=$(dirname $(dirname $NVCC_PATH))
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
        echo -e "${GREEN}${BOLD}[✓] NVCC detected: ${NVCC_PATH} (version ${CUDA_VERSION})${NC}"
        NVCC_AVAILABLE=true
        CUDA_AVAILABLE=true
        CUDA_INSTALLED=true
    fi
    
    # Use CUDA version from nvidia-smi if available
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null)
        if [ -n "$DRIVER_CUDA_VERSION" ]; then
            # Use driver's CUDA version if we couldn't detect it through nvcc
            if [ -z "$CUDA_VERSION" ]; then
                CUDA_VERSION=$DRIVER_CUDA_VERSION
            fi
            CUDA_AVAILABLE=true
        fi
    fi
    
    # Check if environment paths are set up correctly
    if [ "$CUDA_INSTALLED" = true ]; then
        check_cuda_path
    fi
    
    return 0
}

check_cuda_path() {
    PATH_SET=false
    LD_LIBRARY_PATH_SET=false
    
    if [ -n "$CUDA_PATH" ]; then
        if [[ ":$PATH:" == *":$CUDA_PATH/bin:"* ]]; then
            PATH_SET=true
        fi
        
        if [[ ":$LD_LIBRARY_PATH:" == *":$CUDA_PATH/lib64:"* ]]; then
            LD_LIBRARY_PATH_SET=true
        fi
    fi
    
    if [ "$PATH_SET" = false ] || [ "$LD_LIBRARY_PATH_SET" = false ]; then
        echo -e "${YELLOW}${BOLD}[!] CUDA environment paths not properly set - auto-configuring now${NC}"
        setup_cuda_env
        return 1
    fi
    
    echo -e "${GREEN}${BOLD}[✓] CUDA environment paths are properly configured${NC}"
    return 0
}

setup_cuda_env() {
    echo -e "\n${CYAN}${BOLD}[✓] Setting up CUDA environment variables...${NC}"
    
    if [ -z "$CUDA_PATH" ]; then
        for cuda_dir in /usr/local/cuda* /usr/local/cuda; do
            if [ -d "$cuda_dir" ] && [ -d "$cuda_dir/bin" ]; then
                CUDA_PATH=$cuda_dir
                break
            fi
        done
    fi
    
    if [ -z "$CUDA_PATH" ] || [ ! -d "$CUDA_PATH" ]; then
        echo -e "${RED}${BOLD}[✗] Cannot find CUDA directory${NC}"
        return 1
    fi
    
    echo -e "${GREEN}${BOLD}[✓] Using CUDA path: ${CUDA_PATH}${NC}"
    
    # Create systemwide path setup
    cat > /etc/profile.d/cuda.sh <<EOL
#!/bin/bash
export PATH=${CUDA_PATH}/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOL
    chmod +x /etc/profile.d/cuda.sh
    
    # Update current session
    export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
    # Add to .bashrc if it's not already there
    if ! grep -q "CUDA_HOME=${CUDA_PATH}" ~/.bashrc 2>/dev/null; then
        echo -e "\n# CUDA Path" >> ~/.bashrc
        echo "export CUDA_HOME=${CUDA_PATH}" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
    fi
    
    # Source bashrc to apply changes in current session
    source ~/.bashrc 2>/dev/null || true
    
    echo -e "${GREEN}${BOLD}[✓] CUDA environment variables configured and applied${NC}"
    return 0
}

determine_compatible_cuda() {
    echo -e "\n${CYAN}${BOLD}[✓] Determining compatible CUDA version...${NC}"
    
    local compatible_version=""
    
    if command -v nvidia-smi &> /dev/null; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        
        # First try to get CUDA version directly from nvidia-smi
        DRIVER_CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null)
        if [ -n "$DRIVER_CUDA_VERSION" ]; then
            compatible_version=$DRIVER_CUDA_VERSION
            echo -e "${GREEN}${BOLD}[✓] Compatible CUDA version detected: ${compatible_version}${NC}"
            return 0
        fi
        
        # If direct detection failed, estimate based on driver version
        if [ -n "$driver_version" ]; then
            local major_version=$(echo $driver_version | cut -d '.' -f 1)
            
            if [ "$major_version" -ge 545 ]; then
                compatible_version="12.6"
            elif [ "$major_version" -ge 535 ]; then
                compatible_version="12.2"
            elif [ "$major_version" -ge 525 ]; then
                compatible_version="12.1"
            elif [ "$major_version" -ge 520 ]; then
                compatible_version="12.0"
            elif [ "$major_version" -ge 510 ]; then
                compatible_version="11.6"
            elif [ "$major_version" -ge 470 ]; then
                compatible_version="11.4"
            elif [ "$major_version" -ge 450 ]; then
                compatible_version="11.0"
            elif [ "$major_version" -ge 440 ]; then
                compatible_version="10.2"
            elif [ "$major_version" -ge 418 ]; then
                compatible_version="10.1"
            elif [ "$major_version" -ge 410 ]; then
                compatible_version="10.0"
            else
                compatible_version="11.4" # Default fallback for older drivers
            fi
            
            echo -e "${GREEN}${BOLD}[✓] Driver version ${driver_version} is compatible with CUDA ${compatible_version}${NC}"
            return 0
        fi
    fi
    
    # Fallback to a safe version if detection fails
    compatible_version="12.6"
    echo -e "${YELLOW}${BOLD}[!] Could not determine driver version, defaulting to CUDA ${compatible_version}${NC}"
    return 0
}

install_cuda_toolkit() {
    echo -e "\n${CYAN}${BOLD}[✓] Installing CUDA Toolkit...${NC}"
    
    COMPATIBLE_CUDA_VERSION=""
    determine_compatible_cuda
    
    local install_success=false
    
    # Try method 1: Using apt repository
    install_cuda_apt_repo
    if [ $? -eq 0 ]; then
        install_success=true
    else
        echo -e "${YELLOW}${BOLD}[!] Repository installation failed, trying local package method...${NC}"
        install_cuda_local_package
        if [ $? -eq 0 ]; then
            install_success=true
        fi
    fi
    
    if [ "$install_success" = false ]; then
        echo -e "${RED}${BOLD}[✗] All CUDA installation methods failed${NC}"
        echo -e "${YELLOW}${BOLD}[!] Proceeding with CPU-only mode${NC}"
        CPU_ONLY="true"
        return 1
    fi
    
    setup_cuda_env
    detect_cuda
    verify_cuda_installation
    
    return 0
}

install_cuda_apt_repo() {
    local repo_url=""
    local keyring_url=""
    
    case $UBUNTU_VERSION in
        24.04|"24.04")
            repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
            keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        22.04|"22.04")
            repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
            keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        20.04|"20.04")
            repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/"
            keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        18.04|"18.04")
            repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/"
            keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        *)
            repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
            keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
    esac
    
    if [ "$IS_WSL" = true ]; then
        repo_url="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/"
        keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
    fi
    
    local keyring_file="cuda-keyring.deb"
    echo -e "${CYAN}${BOLD}[✓] Downloading CUDA keyring from ${keyring_url}${NC}"
    
    wget --quiet "$keyring_url" -O "$keyring_file" || {
        if ! command -v curl &> /dev/null; then
            apt-get install -y curl >/dev/null 2>&1
        fi
        curl -L "$keyring_url" -o "$keyring_file" --progress-bar
    }
    
    if [ ! -f "$keyring_file" ] || [ ! -s "$keyring_file" ]; then
        echo -e "${RED}${BOLD}[✗] Failed to download keyring file${NC}"
        return 1
    fi
    
    dpkg -i "$keyring_file" || {
        echo -e "${RED}${BOLD}[✗] Failed to install CUDA keyring${NC}"
        rm -f "$keyring_file"
        return 1
    }
    
    echo -e "${CYAN}${BOLD}[✓] Updating package lists...${NC}"
    apt-get update -qq
    
    echo -e "${CYAN}${BOLD}[✓] Installing CUDA packages...${NC}"
    
    # Try to install the specific CUDA version based on driver support
    if [ -n "$DRIVER_CUDA_VERSION" ]; then
        local major_version=$(echo $DRIVER_CUDA_VERSION | cut -d '.' -f 1)
        local minor_version=$(echo $DRIVER_CUDA_VERSION | cut -d '.' -f 2)
        
        # Try specific version first, then fall back to more generic versions
        apt-get install -y cuda-toolkit-${major_version}-${minor_version} || 
        apt-get install -y cuda-toolkit-${major_version} ||
        apt-get install -y cuda || {
            echo -e "${RED}${BOLD}[✗] Failed to install CUDA packages${NC}"
            rm -f "$keyring_file"
            return 1
        }
    else
        # Try generic installation
        apt-get install -y cuda || {
            echo -e "${RED}${BOLD}[✗] Failed to install CUDA packages${NC}"
            rm -f "$keyring_file"
            return 1
        }
    fi
    
    rm -f "$keyring_file"
    echo -e "${GREEN}${BOLD}[✓] CUDA installed via repository method!${NC}"
    return 0
}

install_cuda_local_package() {
    echo -e "\n${CYAN}${BOLD}[✓] Installing CUDA using local package method...${NC}"
    
    local pin_file=""
    local pin_url=""
    local deb_file=""
    local deb_url=""
    local cuda_version="12.6"
    

    if [ -n "$DRIVER_CUDA_VERSION" ]; then
        cuda_version=$DRIVER_CUDA_VERSION
    fi
    
    local major_version=$(echo $cuda_version | cut -d '.' -f 1)
    local minor_version=$(echo $cuda_version | cut -d '.' -f 2)
    
    if [ "$IS_WSL" = true ]; then
        pin_file="cuda-wsl-ubuntu.pin"
        pin_url="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
        deb_file="cuda-repo-wsl-ubuntu-${major_version}-${minor_version}-local_${major_version}.${minor_version}.0-1_amd64.deb"
        deb_url="https://developer.download.nvidia.com/compute/cuda/${major_version}.${minor_version}.0/local_installers/${deb_file}"
    else
        local ubuntu_ver_suffix=""
        case $UBUNTU_VERSION in
            24.04|"24.04") ubuntu_ver_suffix="2404" ;;
            22.04|"22.04") ubuntu_ver_suffix="2204" ;;
            20.04|"20.04") ubuntu_ver_suffix="2004" ;;
            18.04|"18.04") ubuntu_ver_suffix="1804" ;;
            *) ubuntu_ver_suffix="2204" ;;
        esac
        
        pin_file="cuda-ubuntu${ubuntu_ver_suffix}.pin"
        pin_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_ver_suffix}/x86_64/cuda-ubuntu${ubuntu_ver_suffix}.pin"
        deb_file="cuda-repo-ubuntu${ubuntu_ver_suffix}-${major_version}-${minor_version}-local_${major_version}.${minor_version}.0-1_amd64.deb"
        deb_url="https://developer.download.nvidia.com/compute/cuda/${major_version}.${minor_version}.0/local_installers/${deb_file}"
    fi
    
    wget --quiet "$pin_url" -O "$pin_file" 2>/dev/null || {
        if ! command -v curl &> /dev/null; then
            apt-get install -y curl >/dev/null 2>&1
        fi
        curl -sL "$pin_url" -o "$pin_file" 2>/dev/null
    }
    
    if [ -f "$pin_file" ] && [ -s "$pin_file" ]; then
        cp "$pin_file" /etc/apt/preferences.d/cuda-repository-pin-600
    else
        echo -e "${YELLOW}${BOLD}[!] Failed to download pin file, continuing without it${NC}"
    fi
    
    echo -e "${CYAN}${BOLD}[✓] Downloading CUDA repository package...${NC}"
    wget --progress=bar:force "$deb_url" -O "$deb_file" || {
        if ! command -v curl &> /dev/null; then
            apt-get install -y curl >/dev/null 2>&1
        fi
        curl -L "$deb_url" -o "$deb_file" --progress-bar
    }
    
    if [ ! -f "$deb_file" ] || [ ! -s "$deb_file" ]; then
        echo -e "${RED}${BOLD}[✗] Failed to download repository package${NC}"
        rm -f "$pin_file" "$deb_file"
        return 1
    fi
        
    echo -e "${CYAN}${BOLD}[✓] Installing CUDA repository package...${NC}"
    if ! dpkg -i "$deb_file"; then
        echo -e "${RED}${BOLD}[✗] Failed to install repository package${NC}"
        rm -f "$pin_file" "$deb_file"
        return 1
    fi
    
    if [ -f /var/cuda-repo-*/cuda-*-keyring.gpg ]; then
        cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/
    fi
    
    echo -e "${CYAN}${BOLD}[✓] Updating package lists...${NC}"
    apt-get update -qq
    
    echo -e "${CYAN}${BOLD}[✓] Installing CUDA Toolkit...${NC}"
    apt-get install -y cuda || {
        echo -e "${RED}${BOLD}[✗] Failed to install CUDA${NC}"
        rm -f "$pin_file" "$deb_file"
        return 1
    }
    
    rm -f "$pin_file" "$deb_file"
    echo -e "${GREEN}${BOLD}[✓] CUDA installed via local package method!${NC}"
    return 0
}

verify_cuda_installation() {
    echo -e "\n${CYAN}${BOLD}[✓] Verifying CUDA installation...${NC}"
    
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
        echo -e "${GREEN}${BOLD}[✓] NVCC compiler detected (version $NVCC_VERSION)${NC}"
        
        if [ "$IS_RENTED_SERVER" = true ] || [ "$IS_WSL" = true ]; then
            echo -e "${YELLOW}${BOLD}[!] Skipping CUDA test on rented/WSL environment${NC}"
            return 0
        fi
        
        if [ "$GPU_AVAILABLE" = true ] && command -v nvidia-smi &> /dev/null; then
            TEMP_DIR=$(mktemp -d)
            cd "$TEMP_DIR"
            
            echo -e "${CYAN}${BOLD}[✓] Running a simple CUDA test...${NC}"
            cat > cuda_test.cu << 'EOL'
#include <stdio.h>

__global__ void testKernel() {
    printf("GPU kernel executed successfully!\n");
}

int main() {
    printf("Testing CUDA setup...\n");
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CUDA test complete!\n");
    return 0;
}
EOL
            
            if nvcc cuda_test.cu -o cuda_test &>/dev/null; then
                echo -e "${GREEN}${BOLD}[✓] CUDA test compiled successfully${NC}"
                if ./cuda_test 2>/dev/null; then
                    echo -e "${GREEN}${BOLD}[✓] CUDA test executed successfully${NC}"
                else
                    echo -e "${YELLOW}${BOLD}[!] CUDA test execution failed, but compilation was successful${NC}"
                fi
            else
                echo -e "${YELLOW}${BOLD}[!] CUDA test compilation failed${NC}"
            fi
            
            rm -rf "$TEMP_DIR"
        fi
    else
        echo -e "${YELLOW}${BOLD}[!] NVCC compiler not detected in PATH${NC}"
        return 1
    fi
    
    return 0
}

check_cuda_installation() {
    echo -e "\n${CYAN}${BOLD}[✓] Checking CUDA installation status...${NC}"
    
    detect_environment
    detect_gpu
    detect_cuda
    
    if [ "$CUDA_INSTALLED" = true ] && command -v nvcc &> /dev/null; then
        echo -e "${GREEN}${BOLD}[✓] CUDA is properly installed and available${NC}"
        if ! check_cuda_path; then
            :
        fi
        CPU_ONLY="false"
    elif [ "$GPU_AVAILABLE" = true ]; then
        echo -e "${YELLOW}${BOLD}[!] NVIDIA GPU detected but CUDA environment not fully configured${NC}"
        echo -e "${CYAN}${BOLD}[✓] Installing and configuring CUDA automatically...${NC}"
        install_cuda_toolkit
    else
        echo -e "${YELLOW}${BOLD}[!] No NVIDIA GPU detected - using CPU-only mode${NC}"
        CPU_ONLY="true"
    fi
    
    if [ "$CPU_ONLY" = "true" ]; then
        echo -e "\n${YELLOW}${BOLD}[✓] Running in CPU-only mode${NC}"
    else
        echo -e "\n${GREEN}${BOLD}[✓] Running with GPU acceleration${NC}"
        
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${CYAN}${BOLD}[✓] GPU information:${NC}"
            nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv,noheader
        fi
    fi
    
    export CPU_ONLY
    return 0
}

check_cuda_installation
