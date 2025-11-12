"""
WiFi Provisioning System
Implements captive portal for headless device WiFi configuration
"""

import subprocess
import os
import sys
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify
import threading
import time


class WiFiProvisioning:
    """WiFi provisioning with captive portal"""
    
    def __init__(self, ap_ssid="CpSpeech-Setup", ap_password="setup123"):
        self.ap_ssid = ap_ssid
        self.ap_password = ap_password
        self.app = Flask(__name__)
        self.configured = False
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main configuration page"""
            return render_template_string(CAPTIVE_PORTAL_HTML)
        
        @self.app.route('/scan')
        def scan_networks():
            """Scan for available WiFi networks"""
            try:
                networks = self._scan_wifi()
                return jsonify({"success": True, "networks": networks})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/connect', methods=['POST'])
        def connect():
            """Connect to WiFi network"""
            ssid = request.form.get('ssid')
            password = request.form.get('password')
            
            if not ssid:
                return jsonify({"success": False, "error": "SSID required"})
            
            try:
                self._configure_wifi(ssid, password)
                self.configured = True
                return jsonify({
                    "success": True,
                    "message": "WiFi configured. Device will reboot in 5 seconds..."
                })
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/status')
        def status():
            """Check connection status"""
            return jsonify({
                "ap_mode": True,
                "ssid": self.ap_ssid,
                "configured": self.configured
            })
    
    def _scan_wifi(self):
        """Scan for available WiFi networks"""
        try:
            # Use nmcli to scan networks
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 'dev', 'wifi', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            networks = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) >= 2 and parts[0]:
                        networks.append({
                            'ssid': parts[0],
                            'signal': parts[1] if len(parts) > 1 else '0',
                            'secure': 'WPA' in parts[2] if len(parts) > 2 else False
                        })
            
            # Remove duplicates and sort by signal strength
            unique_networks = {n['ssid']: n for n in networks}.values()
            return sorted(unique_networks, key=lambda x: int(x['signal']), reverse=True)
            
        except Exception as e:
            print(f"[ERROR] WiFi scan failed: {e}")
            return []
    
    def _configure_wifi(self, ssid, password):
        """Configure WiFi credentials"""
        try:
            # Create wpa_supplicant configuration
            wpa_config = f'''
country=IN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
'''
            
            # Write configuration
            config_path = '/etc/wpa_supplicant/wpa_supplicant.conf'
            with open(config_path, 'w') as f:
                f.write(wpa_config)
            
            print(f"[INFO] WiFi configured for: {ssid}")
            
            # Schedule reboot
            threading.Thread(target=self._delayed_reboot, daemon=True).start()
            
        except Exception as e:
            raise Exception(f"WiFi configuration failed: {e}")
    
    def _delayed_reboot(self):
        """Reboot device after delay"""
        time.sleep(5)
        print("[INFO] Rebooting to apply WiFi configuration...")
        subprocess.run(['sudo', 'reboot'])
    
    def start_ap_mode(self):
        """Start Access Point mode"""
        try:
            print(f"[INFO] Starting AP mode: {self.ap_ssid}")
            
            # Configure hostapd
            hostapd_config = f'''
interface=wlan0
driver=nl80211
ssid={self.ap_ssid}
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase={self.ap_password}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
'''
            
            with open('/tmp/hostapd.conf', 'w') as f:
                f.write(hostapd_config)
            
            # Start hostapd
            subprocess.run(['sudo', 'systemctl', 'start', 'hostapd'])
            
            # Start dnsmasq
            subprocess.run(['sudo', 'systemctl', 'start', 'dnsmasq'])
            
            print(f"[✓] AP mode started: {self.ap_ssid}")
            print(f"[INFO] Connect to WiFi: {self.ap_ssid}")
            print(f"[INFO] Password: {self.ap_password}")
            print(f"[INFO] Navigate to: http://192.168.4.1")
            
        except Exception as e:
            print(f"[ERROR] Failed to start AP mode: {e}")
            raise
    
    def run(self, host='0.0.0.0', port=80):
        """Run the provisioning server"""
        print(f"[INFO] Starting provisioning server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)


# HTML template for captive portal
CAPTIVE_PORTAL_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>CpSpech WiFi Setup</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 500px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .scan-btn {
            background: #f0f0f0;
            color: #333;
            margin-bottom: 15px;
        }
        .scan-btn:hover {
            background: #e0e0e0;
        }
        .message {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .network-list {
            max-height: 200px;
            overflow-y: auto;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .network-item {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background 0.2s;
        }
        .network-item:hover {
            background: #f8f9fa;
        }
        .network-item:last-child {
            border-bottom: none;
        }
        .network-ssid {
            font-weight: 500;
            color: #333;
        }
        .network-signal {
            font-size: 12px;
            color: #666;
            margin-left: 10px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 CpSpech WiFi Setup</h1>
        <p class="subtitle">Configure your device's WiFi connection</p>
        
        <div id="message" class="message"></div>
        
        <form id="wifiForm">
            <div class="form-group">
                <button type="button" class="scan-btn" onclick="scanNetworks()">
                    📡 Scan for Networks
                </button>
            </div>
            
            <div id="networkList" style="display: none;">
                <label>Available Networks:</label>
                <div class="network-list" id="networks"></div>
            </div>
            
            <div class="form-group">
                <label for="ssid">WiFi Network Name (SSID)</label>
                <input type="text" id="ssid" name="ssid" required placeholder="Enter network name">
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="Enter password (leave blank if open)">
            </div>
            
            <button type="submit" id="connectBtn">Connect to WiFi</button>
        </form>
    </div>
    
    <script>
        function showMessage(text, type) {
            const msg = document.getElementById('message');
            msg.textContent = text;
            msg.className = 'message ' + type;
            msg.style.display = 'block';
        }
        
        function hideMessage() {
            document.getElementById('message').style.display = 'none';
        }
        
        async function scanNetworks() {
            hideMessage();
            const networkList = document.getElementById('networkList');
            const networksDiv = document.getElementById('networks');
            
            networksDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Scanning...</div>';
            networkList.style.display = 'block';
            
            try {
                const response = await fetch('/scan');
                const data = await response.json();
                
                if (data.success && data.networks.length > 0) {
                    networksDiv.innerHTML = '';
                    data.networks.forEach(network => {
                        const div = document.createElement('div');
                        div.className = 'network-item';
                        div.innerHTML = `
                            <span class="network-ssid">${network.ssid}</span>
                            <span class="network-signal">📶 ${network.signal}%</span>
                            ${network.secure ? '🔒' : ''}
                        `;
                        div.onclick = () => {
                            document.getElementById('ssid').value = network.ssid;
                            if (network.secure) {
                                document.getElementById('password').focus();
                            }
                        };
                        networksDiv.appendChild(div);
                    });
                } else {
                    networksDiv.innerHTML = '<div class="loading">No networks found</div>';
                }
            } catch (error) {
                networksDiv.innerHTML = '<div class="loading">Scan failed</div>';
                showMessage('Failed to scan networks', 'error');
            }
        }
        
        document.getElementById('wifiForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            hideMessage();
            
            const ssid = document.getElementById('ssid').value;
            const password = document.getElementById('password').value;
            const btn = document.getElementById('connectBtn');
            
            btn.disabled = true;
            btn.textContent = 'Connecting...';
            
            try {
                const formData = new FormData();
                formData.append('ssid', ssid);
                formData.append('password', password);
                
                const response = await fetch('/connect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showMessage(data.message, 'success');
                    btn.textContent = 'Connected! Rebooting...';
                } else {
                    showMessage('Connection failed: ' + data.error, 'error');
                    btn.disabled = false;
                    btn.textContent = 'Connect to WiFi';
                }
            } catch (error) {
                showMessage('Connection failed: ' + error.message, 'error');
                btn.disabled = false;
                btn.textContent = 'Connect to WiFi';
            }
        });
    </script>
</body>
</html>
'''


def main():
    """Main entry point for WiFi provisioning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WiFi Provisioning System")
    parser.add_argument("--ssid", default="CpSpeech-Setup", help="AP SSID")
    parser.add_argument("--password", default="setup123", help="AP password")
    parser.add_argument("--port", type=int, default=80, help="Web server port")
    
    args = parser.parse_args()
    
    try:
        provisioning = WiFiProvisioning(ap_ssid=args.ssid, ap_password=args.password)
        provisioning.start_ap_mode()
        provisioning.run(port=args.port)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

