#!/bin/bash
# Setup systemd service for auto-start

set -e

echo "=========================================="
echo "  CpSpech Systemd Service Setup"
echo "=========================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "[ERROR] Systemd is only available on Linux"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Project directory: $PROJECT_DIR"

# Update service file with correct paths
SERVICE_FILE="$PROJECT_DIR/systemd/cpspeech.service"
TEMP_SERVICE="/tmp/cpspeech.service"

echo "[INFO] Updating service file paths..."
sed "s|/home/pi/CpSpech|$PROJECT_DIR|g" "$SERVICE_FILE" > "$TEMP_SERVICE"
sed -i "s|User=pi|User=$USER|g" "$TEMP_SERVICE"
sed -i "s|Group=pi|Group=$USER|g" "$TEMP_SERVICE"

# Copy service file
echo "[INFO] Installing service file..."
sudo cp "$TEMP_SERVICE" /etc/systemd/system/cpspeech.service
sudo chmod 644 /etc/systemd/system/cpspeech.service

# Reload systemd
echo "[INFO] Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service
echo "[INFO] Enabling service..."
sudo systemctl enable cpspeech.service

echo ""
echo "=========================================="
echo "  Service Installed Successfully!"
echo "=========================================="
echo ""
echo "Service commands:"
echo "  Start:   sudo systemctl start cpspeech"
echo "  Stop:    sudo systemctl stop cpspeech"
echo "  Restart: sudo systemctl restart cpspeech"
echo "  Status:  sudo systemctl status cpspeech"
echo "  Logs:    sudo journalctl -u cpspeech -f"
echo ""
echo "The service will auto-start on boot."
echo ""
