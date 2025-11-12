# CpSpech V2 - Hybrid Speech Recognition System

**Production-ready speech recognition system for children with cerebral palsy**

## 🎯 Features

- **Hybrid Architecture**: Offline custom ASR + OpenAI TTS API
- **Real-time Recognition**: Low-latency phrase classification
- **Hardware Integration**: Raspberry Pi GPIO support
- **WiFi Provisioning**: Captive portal for headless setup
- **Auto-failover**: Automatic fallback to offline TTS
- **Production Ready**: Systemd service, error handling, logging

---

## 📋 Requirements

### Hardware (Recommended)
- Raspberry Pi 4 (4GB+ RAM) or similar SBC
- I²S MEMS microphone (INMP441) or USB audio dongle
- Tactile button for user input
- Speaker with audio amplifier (PAM8403)
- 2x LEDs (status and recording indicator)
- MicroSD card (32GB+)

### Software
- Python 3.8+
- PyTorch 2.0+
- OpenAI API key
- Internet connection (for initial setup and TTS)

---

## 🚀 Quick Start

### 1. Installation

