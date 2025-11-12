"""
Live Speech Recognition with Hybrid TTS
Records audio from microphone, recognizes phrase, and speaks output
"""

import argparse
import torch
import torchaudio
import pyaudio
import wave
from transformers import Wav2Vec2FeatureExtractor
from model.dataset import SpeechPhraseDataset
from model.wav2vec_classifier import Wav2Vec2Classifier
from inference.tts_engine import speak, TTSEngine
import tempfile
import os
import sys
from pathlib import Path

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 3


class LiveSpeechRecognizer:
    """Production-ready live speech recognition system"""
    
    def __init__(self, model_ckpt, manifest_json, use_tts=True):
        """
        Initialize the recognizer
        
        Args:
            model_ckpt: Path to trained model checkpoint
            manifest_json: Path to manifest JSON
            use_tts: Enable text-to-speech output
        """
        self.use_tts = use_tts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Initialize TTS engine
        if self.use_tts:
            self.tts_engine = TTSEngine(prefer_online=True)
            print(f"[INFO] TTS enabled (Internet: {'Online' if self.tts_engine.is_online() else 'Offline'})")
        
        # Load model and dataset
        print("[INFO] Loading model and dataset...")
        self.feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.dataset = SpeechPhraseDataset(manifest_json, self.feat_extractor)
        self.num_classes = self.dataset.num_classes()
        
        # Load model
        self.model = Wav2Vec2Classifier(num_classes=self.num_classes)
        try:
            state_dict = torch.load(model_ckpt, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device).eval()
            print(f"[INFO] Model loaded successfully ({self.num_classes} classes)")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def record_audio(self, duration=RECORD_SECONDS):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Path to saved WAV file or None if failed
        """
        p = pyaudio.PyAudio()
        stream = None
        temp_wav_path = None
        
        try:
            # Open audio stream
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print(f"\n[🎤] Recording for {duration} seconds. Please speak now...")
            
            frames = []
            for i in range(0, int(SAMPLE_RATE / CHUNK * duration)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"[WARNING] Audio read error: {e}")
                    continue
            
            print("[✓] Recording finished.")
            
            # Save to temporary file
            temp_wav_path = os.path.join(tempfile.gettempdir(), f"recording_{os.getpid()}.wav")
            wf = wave.open(temp_wav_path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
            wf.close()
            
            return temp_wav_path
            
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            return None
            
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            p.terminate()
    
    def predict_from_wav(self, wav_path):
        """
        Predict phrase from WAV file
        
        Args:
            wav_path: Path to WAV file
            
        Returns:
            Tuple of (predicted_phrase, confidence_score) or (None, 0.0) if failed
        """
        try:
            # Load audio
            wav, sr = torchaudio.load(wav_path)
            
            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resampler(wav)
            
            # Pad or trim to fixed length
            x = wav[0]
            length = SAMPLE_RATE * RECORD_SECONDS
            if len(x) < length:
                x = torch.nn.functional.pad(x, (0, length - len(x)))
            else:
                x = x[:length]
            
            # Extract features
            inputs = self.feat_extractor(
                x.numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(inputs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)
                pred_idx = pred.item()
                confidence_score = confidence.item()
            
            # Get phrase label
            phrase = self.dataset.label2phrase.get(pred_idx, "Unknown")
            
            return phrase, confidence_score
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None, 0.0
    
    def run_inference(self, repeat=1, confidence_threshold=0.5):
        """
        Run live inference loop
        
        Args:
            repeat: Number of inference iterations
            confidence_threshold: Minimum confidence for acceptance
        """
        print("\n" + "="*50)
        print("  LIVE SPEECH RECOGNITION SYSTEM")
        print("="*50)
        print(f"Model: {self.num_classes} phrase classes")
        print(f"TTS: {'Enabled' if self.use_tts else 'Disabled'}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("="*50 + "\n")
        
        successful_predictions = 0
        
        for i in range(repeat):
            print(f"\n--- Iteration {i+1}/{repeat} ---")
            
            # Record audio
            temp_wav_path = self.record_audio(duration=RECORD_SECONDS)
            
            if temp_wav_path is None:
                print("[ERROR] Recording failed. Skipping...")
                continue
            
            try:
                # Predict
                phrase, confidence = self.predict_from_wav(temp_wav_path)
                
                if phrase is None:
                    print("[ERROR] Prediction failed.")
                    continue
                
                # Display result
                print(f"\n[✓] Predicted phrase: \"{phrase}\"")
                print(f"[✓] Confidence: {confidence:.2%}")
                
                # Check confidence threshold
                if confidence < confidence_threshold:
                    print(f"[⚠] Low confidence (< {confidence_threshold:.0%}). Please try again.")
                    if self.use_tts:
                        speak("I'm not sure. Please repeat.")
                    continue
                
                successful_predictions += 1
                
                # Speak the result
                if self.use_tts:
                    print(f"[🔊] Speaking: {phrase}")
                    speak(phrase)
                
            finally:
                # Clean up temporary file
                            # Clean up temporary file
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except:
                        pass
        
        # Summary
        print("\n" + "="*50)
        print(f"  SUMMARY: {successful_predictions}/{repeat} successful predictions")
        print("="*50 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Live Speech Phrase Recognition with Hybrid TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once with TTS
  python inference/live_inference.py --model_ckpt models/best_model.pth --manifest_json manifests/train_manifest.json --speak

  # Run 5 times without TTS
  python inference/live_inference.py --model_ckpt models/best_model.pth --manifest_json manifests/train_manifest.json --repeat 5

  # Custom confidence threshold
  python inference/live_inference.py --model_ckpt models/best_model.pth --manifest_json manifests/train_manifest.json --speak --confidence 0.7
        """
    )
    
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--manifest_json",
        type=str,
        required=True,
        help="Path to manifest JSON file"
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Enable text-to-speech output"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of live predictions to run (default: 1)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3,
        help="Recording duration in seconds (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_ckpt):
        print(f"[ERROR] Model checkpoint not found: {args.model_ckpt}")
        sys.exit(1)
    
    if not os.path.exists(args.manifest_json):
        print(f"[ERROR] Manifest file not found: {args.manifest_json}")
        sys.exit(1)
    
    if not 0.0 <= args.confidence <= 1.0:
        print(f"[ERROR] Confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Update global recording duration if specified
    global RECORD_SECONDS
    RECORD_SECONDS = args.duration
    
    try:
        # Initialize recognizer
        recognizer = LiveSpeechRecognizer(
            model_ckpt=args.model_ckpt,
            manifest_json=args.manifest_json,
            use_tts=args.speak
        )
        
        # Run inference
        recognizer.run_inference(
            repeat=args.repeat,
            confidence_threshold=args.confidence
        )
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

