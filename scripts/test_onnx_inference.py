"""
Test ONNX Model Inference
Simple script to test exported ONNX models
"""

import onnxruntime as ort
import numpy as np
import torchaudio
import argparse
import sys
from pathlib import Path
import time


def test_onnx_model(model_path, audio_path=None, num_runs=10):
    """
    Test ONNX model inference
    
    Args:
        model_path: Path to ONNX model
        audio_path: Optional audio file to test
        num_runs: Number of inference runs for benchmarking
    """
    print("\n" + "="*60)
    print("  ONNX MODEL INFERENCE TEST")
    print("="*60)
    
    # Load model
    print(f"\n[INFO] Loading model: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get model info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"[✓] Model loaded")
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output: {output_name}")
    
    # Prepare input
    if audio_path and Path(audio_path).exists():
        print(f"\n[INFO] Loading audio: {audio_path}")
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        
        # Pad or trim
        target_length = 48000  # 3 seconds at 16kHz
        if wav.shape[1] < target_length:
            wav = torch.nn.functional.pad(wav, (0, target_length - wav.shape[1]))
        else:
            wav = wav[:, :target_length]
        
        audio_input = wav.numpy().astype(np.float32)
    else:
        print(f"\n[INFO] Using random audio input")
        audio_input = np.random.randn(1, 48000).astype(np.float32)
    
    # Warm-up run
    print(f"\n[INFO] Running warm-up inference...")
    _ = session.run([output_name], {input_name: audio_input})
    
    # Benchmark
    print(f"\n[INFO] Running {num_runs} inference iterations...")
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        outputs = session.run([output_name], {input_name: audio_input})
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(inference_time)
        
        if i == 0:
            # Print first result
            logits = outputs[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            confidence = np.max(np.exp(logits) / np.sum(np.exp(logits)))
            
            print(f"\n[INFO] First inference result:")
            print(f"  Predicted class: {predicted_class}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Output shape: {logits.shape}")
    
    # Print statistics
    print(f"\n[INFO] Performance Statistics:")
    print(f"  Mean inference time: {np.mean(times):.2f} ms")
    print(f"  Std deviation: {np.std(times):.2f} ms")
    print(f"  Min time: {np.min(times):.2f} ms")
    print(f"  Max time: {np.max(times):.2f} ms")
    
    # Check real-time capability
    audio_duration_ms = 3000  # 3 seconds
    mean_time = np.mean(times)
    
    print(f"\n[INFO] Real-time Analysis:")
    print(f"  Audio duration: {audio_duration_ms} ms")
    print(f"  Processing time: {mean_time:.2f} ms")
    print(f"  Real-time factor: {audio_duration_ms / mean_time:.2f}x")
    
    if mean_time < audio_duration_ms:
        print(f"  ✓ Real-time capable!")
    else:
        print(f"  ⚠ Not real-time (processing slower than audio)")
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--audio", type=str, help="Optional audio file to test")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for benchmarking")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
    
    if args.audio and not Path(args.audio).exists():
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)
    
    try:
        test_onnx_model(args.model, args.audio, args.num_runs)
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
