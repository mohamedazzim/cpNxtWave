#!/bin/bash
# CpSpech V2 Installation Script
# Installs all dependencies and sets up the system

set -e

echo "=========================================="
echo "  CpSpech V2 Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    echo -e "${RED}[ERROR]${NC} Unsupported OS: $OSTYPE"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Detected OS: $OS"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}[INFO]${NC} Python version: $PYTHON_VERSION"

# Update system packages
echo ""
echo -e "${GREEN}[STEP 1/6]${NC} Updating system packages..."
if [ "$OS" == "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv git
    
    # Install audio dependencies
    echo -e "${GREEN}[INFO]${NC} Installing audio dependencies..."
    sudo apt-get install -y portaudio19-dev libsndfile1 ffmpeg mpg123
    sudo apt-get install -y espeak alsa-utils
    
    # Raspberry Pi specific
    if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        echo -e "${GREEN}[INFO]${NC} Raspberry Pi detected"
        sudo apt-get install -y i2c-tools python3-rpi.gpio
        
        # Enable I2C
        if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
            echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
            echo -e "${YELLOW}[WARNING]${NC} I2C enabled. Reboot required."
        fi
    fi
elif [ "$OS" == "mac" ]; then
    brew install portaudio ffmpeg espeak
fi

# Create virtual environment
echo ""
echo -e "${GREEN}[STEP 2/6]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}[✓]${NC} Virtual environment created"
else
    echo -e "${YELLOW}[INFO]${NC} Virtual environment already exists"
fi

# Activate virtual environment
echo -e "${GREEN}[INFO]${NC} Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo -e "${GREEN}[STEP 3/6]${NC} Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo -e "${GREEN}[STEP 4/6]${NC} Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo -e "${GREEN}[STEP 5/6]${NC} Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p dataset/noise
mkdir -p manifests
mkdir -p temp
mkdir -p data/raw data/processed

echo -e "${GREEN}[✓]${NC} Directories created"

# Setup environment file
echo ""
echo -e "${GREEN}[STEP 6/6]${NC} Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo -e "${YELLOW}[ACTION REQUIRED]${NC} Please edit .env file and add your OpenAI API key"
    echo -e "${YELLOW}[INFO]${NC} Run: nano .env"
else
    echo -e "${GREEN}[✓]${NC} .env file already exists"
fi

# Test installation
echo ""
echo -e "${GREEN}[INFO]${NC} Testing installation..."
python3 -c "import torch; import torchaudio; import transformers; print('✓ Core dependencies OK')"
python3 -c "import openai; import flask; print('✓ API dependencies OK')"

# Print summary
echo ""
echo "=========================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Add your OpenAI API key to .env file:"
echo "     nano .env"
echo ""
echo "  3. Prepare your dataset:"
echo "     - Place audio files in dataset/"
echo "     - Run: python data_processing/preprocess.py"
echo "     - Run: python data_processing/split_manifest.py"
echo ""
echo "  4. Train the model:"
echo "     python model/train.py"
echo ""
echo "  5. Test inference:"
echo "     python main.py --interactive --test"
echo ""
echo "  6. Run the system:"
echo "     python main.py --interactive"
echo ""
echo "=========================================="
