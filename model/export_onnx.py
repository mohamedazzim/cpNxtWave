"""
Model Export to ONNX Format
Exports trained PyTorch model to ONNX for optimized inference
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.wav2vec_classifier import Wav2Vec2Classifier
from transformers import Wav2Vec2FeatureExtractor


class ModelExporter:
    """Export PyTorch models to ONNX format"""
    
    def __init__(self, model_path, num_classes, device='cpu'):
        """
        Initialize exporter
        
        Args:
            model_path: Path to trained PyTorch model (.pth)
            num_classes: Number of output classes
            device: Device to load model on
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load trained PyTorch model"""
        print(f"[INFO] Loading model from: {self.model_path}")
        
        try:
            model = Wav2Vec2Classifier(num_classes=self.num_classes)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.device)
            
            print(f"[✓] Model loaded successfully ({self.num_classes} classes)")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def export_to_onnx(self, output_path, input_length=48000, opset_version=14):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            input_length: Input audio length (samples)
            opset_version: ONNX opset version
        """
        print(f"\n[INFO] Exporting to ONNX...")
        print(f"  Output: {output_path}")
        print(f"  Input length: {input_length} samples")
        print(f"  Opset version: {opset_version}")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, input_length).to(self.device)
            
            # Export
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['audio_input'],
                output_names=['logits'],
                dynamic_axes={
                    'audio_input': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                verbose=False
            )
            
            print(f"[✓] Model exported to: {output_path}")
            
            # Verify exported model
            self._verify_onnx_model(output_path)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_onnx_model(self, onnx_path):
        """Verify exported ONNX model"""
        print("\n[INFO] Verifying ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            print("[✓] ONNX model is valid")
            
            # Print model info
            print("\n[INFO] Model Information:")
            print(f"  IR Version: {onnx_model.ir_version}")
            print(f"  Producer: {onnx_model.producer_name}")
            print(f"  Graph name: {onnx_model.graph.name}")
            
            # Print input/output info
            print("\n[INFO] Model Inputs:")
            for input_tensor in onnx_model.graph.input:
                print(f"  - {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
            
            print("\n[INFO] Model Outputs:")
            for output_tensor in onnx_model.graph.output:
                print(f"  - {output_tensor.name}: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
            
            # Test inference
            self._test_onnx_inference(onnx_path)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
            return False
    
    def _test_onnx_inference(self, onnx_path):
        """Test ONNX model inference"""
        print("\n[INFO] Testing ONNX inference...")
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            
            # Get input name and shape
            input_name = session.get_inputs()[0].name
            
            # Create dummy input
            dummy_input = np.random.randn(1, 48000).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: dummy_input})
            
            # Check output
            logits = outputs[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            
            print(f"[✓] ONNX inference successful")
            print(f"  Output shape: {logits.shape}")
            print(f"  Predicted class: {predicted_class}")
            
            # Compare with PyTorch
            self._compare_pytorch_onnx(dummy_input, logits)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] ONNX inference test failed: {e}")
            return False
    
    def _compare_pytorch_onnx(self, test_input, onnx_output):
        """Compare PyTorch and ONNX outputs"""
        print("\n[INFO] Comparing PyTorch vs ONNX outputs...")
        
        try:
            # PyTorch inference
            with torch.no_grad():
                torch_input = torch.from_numpy(test_input).to(self.device)
                torch_output = self.model(torch_input).cpu().numpy()
            
            # Calculate difference
            diff = np.abs(torch_output - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"[✓] Output comparison:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print("  ✓ Outputs match (difference < 1e-4)")
            elif max_diff < 1e-3:
                print("  ⚠ Outputs nearly match (difference < 1e-3)")
            else:
                print("  ⚠ Warning: Significant difference detected")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            return False
    
    def optimize_onnx(self, onnx_path, output_path):
        """
        Optimize ONNX model
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output optimized model path
        """
        print(f"\n[INFO] Optimizing ONNX model...")
        
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Optimize
            from onnxruntime.transformers import optimizer
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',  # Wav2Vec2 is similar to BERT
                num_heads=12,
                hidden_size=768
            )
            
            # Save
            optimized_model.save_model_to_file(output_path)
            print(f"[✓] Optimized model saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"[WARNING] Optimization failed (optional): {e}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Export trained model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python model/export_onnx.py --model_path models/best_model.pth --num_classes 10

  # Export with custom output path
  python model/export_onnx.py --model_path models/best_model.pth --num_classes 10 --output models/model.onnx

  # Export with optimization
  python model/export_onnx.py --model_path models/best_model.pth --num_classes 10 --optimize
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained PyTorch model (.pth)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of output classes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.onnx",
        help="Output ONNX model path (default: models/model.onnx)"
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=48000,
        help="Input audio length in samples (default: 48000 = 3 seconds at 16kHz)"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX model after export"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda, default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  ONNX MODEL EXPORT")
    print("="*60)
    
    try:
        # Initialize exporter
        exporter = ModelExporter(
            model_path=args.model_path,
            num_classes=args.num_classes,
            device=args.device
        )
        
        # Export to ONNX
        success = exporter.export_to_onnx(
            output_path=args.output,
                        input_length=args.input_length,
            opset_version=args.opset_version
        )
        
        if not success:
            print("\n[ERROR] Export failed")
            sys.exit(1)
        
        # Optimize if requested
        if args.optimize:
            optimized_path = args.output.replace('.onnx', '_optimized.onnx')
            exporter.optimize_onnx(args.output, optimized_path)
        
        print("\n" + "="*60)
        print("  EXPORT COMPLETE")
        print("="*60)
        print(f"\n✓ ONNX model: {args.output}")
        
        # Print file size
        file_size = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"✓ File size: {file_size:.2f} MB")
        
        print("\nNext steps:")
        print(f"  1. Test inference: python scripts/test_onnx_inference.py --model {args.output}")
        print(f"  2. Deploy to device: scp {args.output} pi@device:/path/to/models/")
        print()
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

