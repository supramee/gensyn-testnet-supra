#!/bin/bash

GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
BOLD="\033[1m"
NC="\033[0m"

CPU_ONLY="false"
REQUIRED_CUDA_VERSION="12.8"
CUDA_INSTALLED=false
CUDA_COMPATIBLE=false
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
    
    if [ "$CUDA_INSTALLED" = false ] && command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_PATH=$(dirname $(dirname $NVCC_PATH))
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
        echo -e "${GREEN}${BOLD}[✓] NVCC detected: ${NVCC_PATH} (version ${CUDA_VERSION})${NC}"
        NVCC_AVAILABLE=true
        CUDA_AVAILABLE=true
        CUDA_INSTALLED=true
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null)
        if [ -n "$DRIVER_CUDA_VERSION" ]; then
            if [ -z "$CUDA_VERSION" ]; then
                CUDA_VERSION=$DRIVER_CUDA_VERSION
            fi
            CUDA_AVAILABLE=true
        fi
    fi
    
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
    
    cat > /etc/profile.d/cuda.sh <<EOL
#!/bin/bash
export PATH=${CUDA_PATH}/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOL
    chmod +x /etc/profile.d/cuda.sh
    
    export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
    if ! grep -q "CUDA_HOME=${CUDA_PATH}" ~/.bashrc 2>/dev/null; then
        echo -e "\n# CUDA Path" >> ~/.bashrc
        echo "export CUDA_HOME=${CUDA_PATH}" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
    fi
    
    source ~/.bashrc 2>/dev/null || true
    
    echo -e "${GREEN}${BOLD}[✓] CUDA environment variables configured and applied${NC}"
    return 0
}

check_cuda_compatibility() {
    echo -e "\n${CYAN}${BOLD}[✓] Checking compatibility for CUDA ${REQUIRED_CUDA_VERSION}...${NC}"
    
    if [ "$GPU_AVAILABLE" = false ]; then
        echo -e "${RED}${BOLD}[✗] No NVIDIA GPU detected - CUDA ${REQUIRED_CUDA_VERSION} is not supported${NC}"
        show_cpu_option
        return 1
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$DRIVER_VERSION" ]; then
            local major_version=$(echo $DRIVER_VERSION | cut -d '.' -f 1)
            
            if [ "$major_version" -ge 550 ]; then
                echo -e "${GREEN}${BOLD}[✓] Driver version ${DRIVER_VERSION} is compatible with CUDA ${REQUIRED_CUDA_VERSION}${NC}"
                CUDA_COMPATIBLE=true
                return 0
            else
                echo -e "${RED}${BOLD}[✗] Driver version ${DRIVER_VERSION} is NOT compatible with CUDA ${REQUIRED_CUDA_VERSION}${NC}"
                echo -e "${RED}${BOLD}[✗] CUDA ${REQUIRED_CUDA_VERSION} requires NVIDIA driver version 550 or higher${NC}"
                show_cpu_option
                return 1
            fi
        else
            echo -e "${RED}${BOLD}[✗] Could not determine driver version - CUDA ${REQUIRED_CUDA_VERSION} compatibility unknown${NC}"
            show_cpu_option
            return 1
        fi
    else
        echo -e "${RED}${BOLD}[✗] Could not run nvidia-smi - CUDA ${REQUIRED_CUDA_VERSION} compatibility unknown${NC}"
        show_cpu_option
        return 1
    fi
}

show_cpu_option() {
    echo -e "\n${YELLOW}${BOLD}[!] CUDA ${REQUIRED_CUDA_VERSION} is not supported on your system${NC}"
    echo -e "${YELLOW}${BOLD}[!] Would you like to run on CPU instead of GPU? (yes/no)${NC}"
    read -p "> " choice
    
    case "$choice" in
        yes|y|YES|Y )
            CPU_ONLY="true"
            echo -e "${GREEN}${BOLD}[✓] Setting CPU_ONLY=true${NC}"
            export CPU_ONLY=true
            ;;
        * )
            echo -e "${RED}${BOLD}[✗] Exiting as CUDA ${REQUIRED_CUDA_VERSION} is not supported${NC}"
            exit 1
            ;;
    esac
}

install_cuda_toolkit() {
    if [ "$CUDA_COMPATIBLE" = false ]; then
        echo -e "${RED}${BOLD}[✗] Cannot install CUDA ${REQUIRED_CUDA_VERSION} as your system is not compatible${NC}"
        return 1
    fi
    
    echo -e "\n${CYAN}${BOLD}[✓] Installing CUDA ${REQUIRED_CUDA_VERSION} Toolkit...${NC}"
    
    local install_success=false
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
        show_cpu_option
        return 1
    fi
    
    setup_cuda_env
    detect_cuda
    
    echo -e "${GREEN}${BOLD}[✓] CUDA ${REQUIRED_CUDA_VERSION} installed successfully${NC}"
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
    
    echo -e "${CYAN}${BOLD}[✓] Installing CUDA ${REQUIRED_CUDA_VERSION} packages...${NC}"
    
    local major_version=$(echo $REQUIRED_CUDA_VERSION | cut -d '.' -f 1)
    local minor_version=$(echo $REQUIRED_CUDA_VERSION | cut -d '.' -f 2)
    
    apt-get install -y cuda-toolkit-${major_version}-${minor_version} || 
    apt-get install -y cuda-${major_version}-${minor_version} ||
    apt-get install -y cuda || {
        echo -e "${RED}${BOLD}[✗] Failed to install CUDA packages${NC}"
        rm -f "$keyring_file"
        return 1
    }
    
    rm -f "$keyring_file"
    echo -e "${GREEN}${BOLD}[✓] CUDA ${REQUIRED_CUDA_VERSION} installed via repository method!${NC}"
    return 0
}

install_cuda_local_package() {
    echo -e "\n${CYAN}${BOLD}[✓] Installing CUDA ${REQUIRED_CUDA_VERSION} using local package method...${NC}"
    
    local pin_file=""
    local pin_url=""
    local deb_file=""
    local deb_url=""
    
    local major_version=$(echo $REQUIRED_CUDA_VERSION | cut -d '.' -f 1)
    local minor_version=$(echo $REQUIRED_CUDA_VERSION | cut -d '.' -f 2)
    
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
    
    echo -e "${CYAN}${BOLD}[✓] Installing CUDA ${REQUIRED_CUDA_VERSION} Toolkit...${NC}"
    apt-get install -y cuda-${major_version}-${minor_version} || 
    apt-get install -y cuda || {
        echo -e "${RED}${BOLD}[✗] Failed to install CUDA${NC}"
        rm -f "$pin_file" "$deb_file"
        return 1
    }
    
    rm -f "$pin_file" "$deb_file"
    echo -e "${GREEN}${BOLD}[✓] CUDA ${REQUIRED_CUDA_VERSION} installed via local package method!${NC}"
    return 0
}

main() { 
    detect_environment
    detect_gpu
    detect_cuda
    
    if [ "$CUDA_INSTALLED" = true ] && [ -n "$CUDA_VERSION" ]; then
        if [ "$CUDA_VERSION" = "$REQUIRED_CUDA_VERSION" ]; then
            echo -e "${GREEN}${BOLD}[✓] CUDA ${REQUIRED_CUDA_VERSION} is already installed!${NC}"
            check_cuda_path
            CPU_ONLY="false"
            return 0
        else
            echo -e "${YELLOW}${BOLD}[!] CUDA ${CUDA_VERSION} is installed, but CUDA ${REQUIRED_CUDA_VERSION} is required${NC}"
        fi
    fi
    
    if [ "$GPU_AVAILABLE" = true ]; then
        check_cuda_compatibility
        if [ "$CUDA_COMPATIBLE" = true ] && [ "$CPU_ONLY" = "false" ]; then
            install_cuda_toolkit
        fi
    else
        show_cpu_option
    fi
    
    if [ "$CPU_ONLY" = "true" ]; then
        echo -e "\n${YELLOW}${BOLD}[✓] Running in CPU-only mode${NC}"
        export CPU_ONLY=true
    else
        echo -e "\n${GREEN}${BOLD}[✓] Running with GPU acceleration using CUDA ${REQUIRED_CUDA_VERSION}${NC}"
        
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${CYAN}${BOLD}[✓] GPU information:${NC}"
            nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv,noheader
        fi
    fi
    
    return 0
}

main
