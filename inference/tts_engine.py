"""
Hybrid Text-to-Speech Engine
Supports OpenAI TTS API (online) with pyttsx3 fallback (offline)
"""

import pyttsx3
import requests
import subprocess
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TTSEngine:
    """Hybrid TTS engine with OpenAI API and offline fallback"""
    
    def __init__(self, prefer_online=True):
        self.prefer_online = prefer_online
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.temp_dir = tempfile.gettempdir()
        
    def is_online(self, timeout=3):
        """Check if internet connection is available"""
        try:
            response = requests.get("https://api.openai.com", timeout=timeout)
            return response.status_code in [200, 403]  # 403 means server is reachable
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def speak_openai(self, text, voice="nova", model="tts-1"):
        """
        Use OpenAI TTS API to generate speech
        
        Args:
            text: Text to speak
            voice: Voice model (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model (tts-1 or tts-1-hd)
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            # Use OpenAI API via requests (compatible with openai>=1.0)
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Generate speech
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Save to temporary file
            audio_path = os.path.join(self.temp_dir, f"tts_output_{os.getpid()}.mp3")
            response.stream_to_file(audio_path)
            
            # Play audio
            self._play_audio(audio_path)
            
            # Clean up
            try:
                os.remove(audio_path)
            except:
                pass
                
            return True
            
        except Exception as e:
            print(f"[ERROR] OpenAI TTS failed: {e}")
            return False
    
    def speak_offline(self, text, rate=160):
        """
        Use pyttsx3 for offline TTS
        
        Args:
            text: Text to speak
            rate: Speech rate (words per minute)
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", 1.0)
            
            # Optional: Set voice (uncomment if needed)
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[0].id)  # 0=male, 1=female
            
            engine.say(text)
            engine.runAndWait()
            return True
            
        except Exception as e:
            print(f"[ERROR] Offline TTS failed: {e}")
            return False
    
    def _play_audio(self, audio_path):
        """Play audio file using system command"""
        try:
            # Try different audio players
            players = ['mpg123', 'ffplay', 'aplay', 'play']
            
            for player in players:
                try:
                    if player == 'ffplay':
                        subprocess.run([player, '-nodisp', '-autoexit', audio_path], 
                                     check=True, stderr=subprocess.DEVNULL)
                    else:
                        subprocess.run([player, audio_path], 
                                     check=True, stderr=subprocess.DEVNULL)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            # If no player works, use pygame as last resort
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            except:
                pass
                
            print("[WARNING] No audio player available. Audio generated but not played.")
            return False
            
        except Exception as e:
            print(f"[ERROR] Audio playback failed: {e}")
            return False
    
    def speak(self, text):
        """
        Main speak function with hybrid logic
        
        Args:
            text: Text to speak
        
        Returns:
            bool: True if speech was successful
        """
        if not text or not text.strip():
            print("[WARNING] Empty text provided to TTS engine")
            return False
        
        print(f"[TTS] Speaking: {text}")
        
        # Try OpenAI TTS if online and preferred
        if self.prefer_online and self.is_online():
            print("[TTS] Using OpenAI TTS API...")
            if self.speak_openai(text):
                return True
            print("[TTS] OpenAI TTS failed, falling back to offline TTS...")
        
        # Fallback to offline TTS
        print("[TTS] Using offline TTS (pyttsx3)...")
        return self.speak_offline(text)


# Global TTS engine instance
_tts_engine = None

def get_tts_engine():
    """Get or create global TTS engine instance"""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine(prefer_online=True)
    return _tts_engine

def speak(text):
    """
    Convenience function for quick TTS
    
    Args:
        text: Text to speak
    """
    engine = get_tts_engine()
    return engine.speak(text)


if __name__ == "__main__":
    # Test the TTS engine
    import sys
    
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = "Hello! This is a test of the hybrid text to speech engine."
    
    print("Testing TTS Engine...")
    print(f"Text: {test_text}")
    
    engine = TTSEngine(prefer_online=True)
    
    # Test online status
    online = engine.is_online()
    print(f"Internet: {'Online' if online else 'Offline'}")
    
    # Test speech
    success = engine.speak(test_text)
    print(f"Result: {'Success' if success else 'Failed'}")
