#!/bin/bash
# Install and configure hostapd and dnsmasq for WiFi AP mode

set -e

echo "Installing AP mode dependencies..."
sudo apt-get update
sudo apt-get install -y hostapd dnsmasq iptables-persistent

# Stop services
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq

# Configure hostapd
sudo tee /etc/hostapd/hostapd.conf > /dev/null <<EOF
interface=wlan0
driver=nl80211
ssid=CpSpeech-Setup
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=setup123
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# Configure dnsmasq
sudo tee /etc/dnsmasq.conf > /dev/null <<EOF
interface=wlan0
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
address=/#/192.168.4.1
EOF

# Configure wlan0 static IP
sudo tee -a /etc/dhcpcd.conf > /dev/null <<EOF

# Static IP for AP mode
interface wlan0
static ip_address=192.168.4.1/24
nohook wpa_supplicant
EOF

echo "✓ AP mode configuration complete"
echo "Run 'sudo systemctl start hostapd' and 'sudo systemctl start dnsmasq' to enable AP"
