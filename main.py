#!/usr/bin/env python3
"""
CpSpech V2 - Main Device Entry Point
Production-ready speech recognition system with hybrid TTS
"""

import os
import sys
import time
import signal
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core modules
import torch
from transformers import Wav2Vec2FeatureExtractor
from model.dataset import SpeechPhraseDataset
from model.wav2vec_classifier import Wav2Vec2Classifier
from inference.tts_engine import TTSEngine
import pyaudio
import torchaudio
import tempfile
import wave

# Optional: GPIO for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARNING] RPi.GPIO not available. GPIO features disabled.")


class CpSpeechDevice:
    """Main device controller for CpSpech system"""
    
    def __init__(self, config_path="configs/project_config.yaml"):
        """Initialize the device"""
        
        # Load configuration
        print("[INFO] Loading configuration...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.device = torch.device(
            self.config['performance']['device_type']
            if torch.cuda.is_available() and self.config['performance']['device_type'] == 'cuda'
            else 'cpu'
        )
        print(f"[INFO] Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Initialize TTS
        self.tts_engine = TTSEngine(prefer_online=True)
        print(f"[INFO] TTS initialized (Online: {self.tts_engine.is_online()})")
        
        # Initialize GPIO if available
        if GPIO_AVAILABLE and self.config['hardware']['enable_gpio']:
            self._setup_gpio()
        
        # System state
        self.running = True
        self.recording = False
        
        print("[✓] Device initialized successfully")
    
    def _load_model(self):
        """Load trained model"""
        model_path = self.config['model']['checkpoint_path']
        manifest_path = self.config['model']['manifest_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        print("[INFO] Loading model...")
        
        # Load feature extractor
        self.feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config['model']['pretrained_model']
        )
        
        # Load dataset (for label mapping)
        self.dataset = SpeechPhraseDataset(manifest_path, self.feat_extractor)
        self.num_classes = self.dataset.num_classes()
        
        # Load model
        self.model = Wav2Vec2Classifier(num_classes=self.num_classes)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        
        print(f"[✓] Model loaded ({self.num_classes} classes)")
    
    def _setup_gpio(self):
        """Setup GPIO pins for button and LEDs"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            button_pin = self.config['hardware']['button_pin']
            led_status_pin = self.config['hardware']['led_status_pin']
            led_recording_pin = self.config['hardware']['led_recording_pin']
            
            # Setup button
            GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Setup LEDs
            GPIO.setup(led_status_pin, GPIO.OUT)
            GPIO.setup(led_recording_pin, GPIO.OUT)
            
            # Turn on status LED
            GPIO.output(led_status_pin, GPIO.HIGH)
            
            # Add button event detection
            GPIO.add_event_detect(
                button_pin,
                GPIO.FALLING,
                callback=self._on_button_press,
                bouncetime=300
            )
            
            self.gpio_initialized = True
            print("[✓] GPIO initialized")
            
        except Exception as e:
            print(f"[ERROR] GPIO setup failed: {e}")
            self.gpio_initialized = False
    
    def _on_button_press(self, channel):
        """Callback for button press"""
        if not self.recording:
            print("\n[BUTTON] Button pressed - starting recognition...")
            self.process_speech()
    
    def record_audio(self, duration=None):
        """Record audio from microphone"""
        if duration is None:
            duration = self.config['audio']['record_seconds']
        
        sample_rate = self.config['audio']['sample_rate']
        channels = self.config['audio']['channels']
        chunk = self.config['audio']['chunk_size']
        
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            # Turn on recording LED
            if hasattr(self, 'gpio_initialized') and self.gpio_initialized:
                GPIO.output(self.config['hardware']['le
