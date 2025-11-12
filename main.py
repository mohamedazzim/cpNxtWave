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
                        # Turn on recording LED
            if hasattr(self, 'gpio_initialized') and self.gpio_initialized:
                GPIO.output(self.config['hardware']['led_recording_pin'], GPIO.HIGH)
            
            self.recording = True
            
            # Open audio stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            print(f"[🎤] Recording for {duration} seconds...")
            
            frames = []
            for _ in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            
            print("[✓] Recording complete")
            
            # Save to temporary file
            temp_path = os.path.join(tempfile.gettempdir(), f"recording_{os.getpid()}.wav")
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return temp_path
            
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            return None
            
        finally:
            self.recording = False
            
            # Turn off recording LED
            if hasattr(self, 'gpio_initialized') and self.gpio_initialized:
                GPIO.output(self.config['hardware']['led_recording_pin'], GPIO.LOW)
            
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    def predict(self, audio_path):
        """Predict phrase from audio file"""
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.config['audio']['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr, self.config['audio']['sample_rate'])
                wav = resampler(wav)
            
            # Pad or trim
            x = wav[0]
            target_length = self.config['audio']['sample_rate'] * self.config['audio']['record_seconds']
            if len(x) < target_length:
                x = torch.nn.functional.pad(x, (0, target_length - len(x)))
            else:
                x = x[:target_length]
            
            # Extract features
            inputs = self.feat_extractor(
                x.numpy(),
                sampling_rate=self.config['audio']['sample_rate'],
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(inputs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)
            
            phrase = self.dataset.label2phrase.get(pred.item(), "Unknown")
            confidence_score = confidence.item()
            
            return phrase, confidence_score
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None, 0.0
    
    def process_speech(self):
        """Complete speech processing workflow"""
        try:
            # Record audio
            audio_path = self.record_audio()
            
            if audio_path is None:
                print("[ERROR] Recording failed")
                return
            
            # Predict
            print("[🤖] Processing...")
            phrase, confidence = self.predict(audio_path)
            
            # Clean up audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            if phrase is None:
                print("[ERROR] Recognition failed")
                self.tts_engine.speak("Sorry, I could not understand.")
                return
            
            # Check confidence threshold
            threshold = self.config['recognition']['confidence_threshold']
            
            print(f"\n[✓] Recognized: \"{phrase}\"")
            print(f"[✓] Confidence: {confidence:.2%}")
            
            if confidence < threshold:
                print(f"[⚠] Low confidence (< {threshold:.0%})")
                self.tts_engine.speak("I'm not sure. Please repeat.")
                return
            
            # Speak the result
            print(f"[🔊] Speaking: {phrase}")
            self.tts_engine.speak(phrase)
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_interactive_mode(self):
        """Run in interactive keyboard mode (for testing without GPIO)"""
        print("\n" + "="*60)
        print("  CP SPEECH RECOGNITION SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Press ENTER to record and recognize speech")
        print("Press 'q' + ENTER to quit")
        print("="*60 + "\n")
        
        try:
            while self.running:
                user_input = input("\n[Press ENTER to start] ").strip().lower()
                
                if user_input == 'q':
                    print("\n[INFO] Shutting down...")
                    break
                
                self.process_speech()
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
    
    def run_button_mode(self):
        """Run in button trigger mode (for GPIO)"""
        if not hasattr(self, 'gpio_initialized') or not self.gpio_initialized:
            print("[ERROR] GPIO not initialized. Use --interactive mode instead.")
            return
        
        print("\n" + "="*60)
        print("  CP SPEECH RECOGNITION SYSTEM - BUTTON MODE")
        print("="*60)
        print("Press the physical button to record and recognize speech")
        print("Press Ctrl+C to quit")
        print("="*60 + "\n")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[INFO] Cleaning up...")
        
        if hasattr(self, 'gpio_initialized') and self.gpio_initialized:
            GPIO.cleanup()
            print("[✓] GPIO cleaned up")
        
        print("[✓] Shutdown complete")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n[INFO] Shutdown signal received")
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CpSpech V2 - Speech Recognition Device",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive keyboard mode (no GPIO required)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single test recognition"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize device
        device = CpSpeechDevice(config_path=args.config)
        
        # Test mode
        if args.test:
            print("\n[TEST MODE] Running single recognition test...")
            device.process_speech()
            return
        
        # Run appropriate mode
        if args.interactive or not GPIO_AVAILABLE:
            device.run_interactive_mode()
        else:
            device.run_button_mode()
        
        # Cleanup
        device.cleanup()
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

