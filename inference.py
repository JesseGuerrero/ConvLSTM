#!/usr/bin/env python3
"""
Satellite ConvLSTM Inference CLI
==============================

Command-line interface for running satellite imagery predictions with temporal encoding.

Usage:
    python inference_cli.py --input_dir ./San_Antonio --model_path model.ckpt --days_ahead 30
    python inference_cli.py --input_dir ./San_Antonio --days_ahead 7 --output_dir ./weekly_preds
    python inference_cli.py --help
"""

import argparse
import os
import sys
from pathlib import Path

# Import the inference functions (assuming they're in the same file or module)
from real_tif_inference import (
    real_satellite_inference,
    save_predictions_as_tif,
    save_predictions_as_png,
    extract_date_from_filename,
    calculate_future_dates
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Satellite ConvLSTM Inference with Temporal Encoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with 30-day intervals
  python inference_cli.py --input_dir ./San_Antonio --model_path model.ckpt

  # Weekly predictions
  python inference_cli.py --input_dir ./San_Antonio --model_path model.ckpt --days_ahead 7

  # Seasonal predictions with custom output
  python inference_cli.py --input_dir ./San_Antonio --model_path model.ckpt \\
                         --days_ahead 90 --output_dir ./seasonal_predictions

  # Use CPU and save only PNG
  python inference_cli.py --input_dir ./San_Antonio --model_path model.ckpt \\
                         --device cpu --format png
        """
    )
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input .tif satellite images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ConvLSTM model checkpoint')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Directory to save predictions (default: ./inference_output)')
    parser.add_argument('--days_ahead', type=int, default=30,
                       help='Days between each prediction (default: 30)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on (default: auto)')
    parser.add_argument('--format', type=str, default='both',
                       choices=['tif', 'png', 'both'],
                       help='Output format: tif (georeferenced), png, or both (default: both)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without running inference')
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input arguments and files."""
    errors = []
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        errors.append(f"Input directory not found: {args.input_dir}")
    else:
        # Check for .tif files
        tif_files = list(Path(args.input_dir).glob("*.tif"))
        if len(tif_files) < 10:
            errors.append(f"Need at least 10 .tif files, found {len(tif_files)} in {args.input_dir}")
    
    # Check model path
    if not os.path.exists(args.model_path):
        errors.append(f"Model checkpoint not found: {args.model_path}")
    
    # Check days_ahead
    if args.days_ahead <= 0:
        errors.append(f"days_ahead must be positive, got {args.days_ahead}")
    
    # Check batch_size
    if args.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {args.batch_size}")
    
    return errors

def setup_device(device_arg):
    """Setup computing device."""
    import torch
    
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    return device

def load_model(model_path, device):
    """Load the trained model."""
    print(f"🤖 Loading model from {model_path}")
    
    try:
        # This would be the actual model loading
        # from your_training_module import SatelliteConvLSTMPredictor
        # model = SatelliteConvLSTMPredictor.load_from_checkpoint(model_path)
        # model.to(device)
        # model.eval()
        
        # For demonstration, create a dummy model
        import torch
        
        class DummyModel:
            def __init__(self, device):
                self.device = torch.device(device)
            
            def eval(self):
                pass
            
            def parameters(self):
                return [torch.tensor([1.0], device=self.device)]
            
            def __call__(self, input_seq, input_temporal, target_temporal):
                batch_size, seq_len, channels, height, width = input_seq.shape
                target_len = target_temporal.shape[1]
                predictions = torch.randn(batch_size, target_len, channels, height, width, 
                                        device=self.device)
                predictions = torch.sigmoid(predictions)
                return predictions
        
        model = DummyModel(device)
        print(f"   ✅ Model loaded successfully (dummy model for demo)")
        return model
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        raise

def preview_inference(args):
    """Show what files would be processed without running inference."""
    import glob
    
    print(f"🔍 DRY RUN - Preview of inference pipeline")
    print(f"=" * 50)
    
    # Show input files
    tif_files = glob.glob(os.path.join(args.input_dir, "*.tif"))
    sorted_files = sorted(tif_files, key=lambda f: extract_date_from_filename(os.path.basename(f)))
    recent_files = sorted_files[-10:]  # Most recent 10
    
    print(f"📂 Input directory: {args.input_dir}")
    print(f"📸 Found {len(tif_files)} .tif files, using 10 most recent:")
    
    input_dates = []
    for i, filepath in enumerate(recent_files):
        date = extract_date_from_filename(os.path.basename(filepath))
        input_dates.append(date)
        print(f"   {i+1:2d}. {os.path.basename(filepath)} → {date[0]}-{date[1]:02d}-{date[2]:02d}")
    
    # Show prediction dates
    last_date = input_dates[-1]
    future_dates = calculate_future_dates(last_date, args.days_ahead)
    
    print(f"\n🔮 Would predict for dates (every {args.days_ahead} days):")
    for i, date in enumerate(future_dates):
        days_from_last = args.days_ahead * (i + 1)
        print(f"   {i+1}. {date[0]}-{date[1]:02d}-{date[2]:02d} (+{days_from_last} days)")
    
    # Show output plan
    print(f"\n💾 Output plan:")
    print(f"   Directory: {args.output_dir}")
    print(f"   Format: {args.format}")
    
    if args.format in ['tif', 'both']:
        print(f"   Georeferenced .tif files:")
        for date in future_dates:
            filename = f"prediction_{date[0]:04d}-{date[1]:02d}-{date[2]:02d}.tif"
            print(f"     📄 {filename}")
    
    if args.format in ['png', 'both']:
        png_dir = os.path.join(args.output_dir, 'png_versions') if args.format == 'both' else args.output_dir
        print(f"   PNG files (in {png_dir}):")
        for date in future_dates:
            filename = f"prediction_{date[0]:04d}-{date[1]:02d}-{date[2]:02d}.png"
            print(f"     🖼️  {filename}")
    
    print(f"\n⚙️  Processing settings:")
    print(f"   Device: {setup_device(args.device)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Model: {args.model_path}")

def main():
    """Main CLI function."""
    print("🛰️  Satellite ConvLSTM Inference CLI")
    print("=" * 40)
    
    # Parse arguments
    args = parse_arguments()
    
    if args.verbose:
        print(f"📋 Arguments: {vars(args)}")
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        preview_inference(args)
        print(f"\n🏁 Dry run completed. Add --no-dry-run to run actual inference.")
        return
    
    # Setup device
    device = setup_device(args.device)
    print(f"💻 Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    print(f"\n🚀 Starting inference...")
    try:
        predictions, future_dates, geo_metadata = real_satellite_inference(
            model=model,
            input_dir=args.input_dir,
            days_ahead=args.days_ahead
        )
        
        print(f"\n✅ Inference completed!")
        print(f"   Predictions: {predictions.shape}")
        print(f"   Future dates: {future_dates}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        if args.verbose:
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    # Save predictions
    print(f"\n💾 Saving predictions...")
    try:
        if args.format in ['tif', 'both']:
            save_predictions_as_tif(
                predictions=predictions,
                future_dates=future_dates,
                geo_metadata=geo_metadata,
                output_dir=args.output_dir
            )
        
        if args.format in ['png', 'both']:
            png_dir = os.path.join(args.output_dir, 'png_versions') if args.format == 'both' else args.output_dir
            save_predictions_as_png(
                predictions=predictions,
                future_dates=future_dates,
                output_dir=png_dir
            )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Show generated files
        if os.path.exists(args.output_dir):
            print(f"\n📋 Generated files:")
            for root, dirs, files in os.walk(args.output_dir):
                for file in sorted(files):
                    rel_path = os.path.relpath(os.path.join(root, file), args.output_dir)
                    size_mb = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                    print(f"   📄 {rel_path} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")
        if args.verbose:
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()