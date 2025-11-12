"""
GPIO Control Module for Raspberry Pi
Handles button input, LED status indicators, and hardware interfacing
"""

import time
import threading

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARNING] RPi.GPIO not available")


class GPIOController:
    """GPIO controller for hardware interfacing"""
    
    def __init__(self, button_pin=17, led_status_pin=27, led_recording_pin=22):
        """
        Initialize GPIO controller
        
        Args:
            button_pin: GPIO pin for tactile button
            led_status_pin: GPIO pin for status LED
            led_recording_pin: GPIO pin for recording indicator LED
        """
        if not GPIO_AVAILABLE:
            raise ImportError("RPi.GPIO not available. Install with: pip install RPi.GPIO")
        
        self.button_pin = button_pin
        self.led_status_pin = led_status_pin
        self.led_recording_pin = led_recording_pin
        
        self.button_callback = None
        self.initialized = False
        
        self._setup()
    
    def _setup(self):
        """Setup GPIO pins"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup button with pull-up resistor
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Setup LEDs as outputs
            GPIO.setup(self.led_status_pin, GPIO.OUT)
            GPIO.setup(self.led_recording_pin, GPIO.OUT)
            
            # Initialize LEDs to OFF
            GPIO.output(self.led_status_pin, GPIO.LOW)
            GPIO.output(self.led_recording_pin, GPIO.LOW)
            
            self.initialized = True
            print("[✓] GPIO initialized successfully")
            print(f"    Button: GPIO{self.button_pin}")
            print(f"    Status LED: GPIO{self.led_status_pin}")
            print(f"    Recording LED: GPIO{self.led_recording_pin}")
            
        except Exception as e:
            print(f"[ERROR] GPIO setup failed: {e}")
            raise
    
    def set_button_callback(self, callback, bouncetime=300):
        """
        Set callback function for button press
        
        Args:
            callback: Function to call on button press
            bouncetime: Debounce time in milliseconds
        """
        if not self.initialized:
            print("[ERROR] GPIO not initialized")
            return
        
        self.button_callback = callback
        
        try:
            GPIO.add_event_detect(
                self.button_pin,
                GPIO.FALLING,
                callback=self._button_handler,
                bouncetime=bouncetime
            )
            print(f"[✓] Button callback registered (debounce: {bouncetime}ms)")
        except Exception as e:
            print(f"[ERROR] Failed to register button callback: {e}")
    
    def _button_handler(self, channel):
        """Internal button handler with debouncing"""
        if self.button_callback:
            self.button_callback(channel)
    
    def led_status_on(self):
        """Turn on status LED"""
        if self.initialized:
            GPIO.output(self.led_status_pin, GPIO.HIGH)
    
    def led_status_off(self):
        """Turn off status LED"""
        if self.initialized:
            GPIO.output(self.led_status_pin, GPIO.LOW)
    
    def led_status_blink(self, times=3, interval=0.2):
        """
        Blink status LED
        
        Args:
            times: Number of blinks
            interval: Delay between blinks (seconds)
        """
        def blink():
            for _ in range(times):
                self.led_status_on()
                time.sleep(interval)
                self.led_status_off()
                time.sleep(interval)
        
        threading.Thread(target=blink, daemon=True).start()
    
    def led_recording_on(self):
        """Turn on recording LED"""
        if self.initialized:
            GPIO.output(self.led_recording_pin, GPIO.HIGH)
    
    def led_recording_off(self):
        """Turn off recording LED"""
        if self.initialized:
            GPIO.output(self.led_recording_pin, GPIO.LOW)
    
    def led_recording_pulse(self, duration=3.0):
        """
        Pulse recording LED for duration
        
        Args:
            duration: Pulse duration in seconds
        """
        def pulse():
            self.led_recording_on()
            time.sleep(duration)
            self.led_recording_off()
        
        threading.Thread(target=pulse, daemon=True).start()
    
    def is_button_pressed(self):
        """Check if button is currently pressed"""
        if self.initialized:
            return GPIO.input(self.button_pin) == GPIO.LOW
        return False
        def wait_for_button_press(self, timeout=None):
        """
        Wait for button press (blocking)
        
        Args:
            timeout: Maximum wait time in seconds (None for infinite)
            
        Returns:
            True if button was pressed, False if timeout
        """
        if not self.initialized:
            return False
        
        try:
            channel = GPIO.wait_for_edge(
                self.button_pin,
                GPIO.FALLING,
                timeout=int(timeout * 1000) if timeout else None
            )
            return channel is not None
        except Exception as e:
            print(f"[ERROR] Wait for button failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.initialized:
            try:
                GPIO.cleanup()
                print("[✓] GPIO cleaned up")
            except:
                pass


class MockGPIOController:
    """Mock GPIO controller for development without hardware"""
    
    def __init__(self, *args, **kwargs):
        print("[INFO] Using mock GPIO controller (no hardware)")
        self.initialized = True
    
    def set_button_callback(self, callback, bouncetime=300):
        print(f"[MOCK] Button callback registered")
    
    def led_status_on(self):
        print("[MOCK] Status LED: ON")
    
    def led_status_off(self):
        print("[MOCK] Status LED: OFF")
    
    def led_status_blink(self, times=3, interval=0.2):
        print(f"[MOCK] Status LED: Blinking {times} times")
    
    def led_recording_on(self):
        print("[MOCK] Recording LED: ON")
    
    def led_recording_off(self):
        print("[MOCK] Recording LED: OFF")
    
    def led_recording_pulse(self, duration=3.0):
        print(f"[MOCK] Recording LED: Pulsing for {duration}s")
    
    def is_button_pressed(self):
        return False
    
    def wait_for_button_press(self, timeout=None):
        print("[MOCK] Waiting for button press...")
        return False
    
    def cleanup(self):
        print("[MOCK] GPIO cleanup")


def get_gpio_controller(*args, **kwargs):
    """Factory function to get appropriate GPIO controller"""
    if GPIO_AVAILABLE:
        return GPIOController(*args, **kwargs)
    else:
        return MockGPIOController(*args, **kwargs)


# Test function
def test_gpio():
    """Test GPIO functionality"""
    print("\n=== GPIO Controller Test ===\n")
    
    try:
        gpio = get_gpio_controller()
        
        # Test status LED
        print("Testing status LED...")
        gpio.led_status_on()
        time.sleep(1)
        gpio.led_status_off()
        time.sleep(0.5)
        gpio.led_status_blink(times=3)
        time.sleep(2)
        
        # Test recording LED
        print("Testing recording LED...")
        gpio.led_recording_pulse(duration=2)
        time.sleep(3)
        
        # Test button
        def on_button_press(channel):
            print(f"\n[BUTTON] Button pressed on GPIO{channel}!")
            gpio.led_status_blink(times=2, interval=0.1)
        
        gpio.set_button_callback(on_button_press)
        
        print("\nPress the button to test (Ctrl+C to exit)...")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    finally:
        gpio.cleanup()


if __name__ == "__main__":
    test_gpio()

    
