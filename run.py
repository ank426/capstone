#!/usr/bin/env python3
"""
Inference script for Web Attack Detector
Usage:
  python run.py --model_path models/web_attack_model.pt --payload "Your payload here"
  python run.py --model_path models/web_attack_model.pt --interactive
  python run.py --model_path models/web_attack_model.pt --file payloads.txt
"""

import argparse
import json
from pathlib import Path
import sys

from web_attack_detector import WebAttackTrainer

class WebAttackInference:
    """Inference class for web attack detection."""

    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load configuration
        if config_path is None:
            config_path = self.model_path.parent / 'config.json'

        self.config = self.load_config(config_path)

        # Initialize trainer for inference
        print(f"Loading model from {model_path}...")
        self.trainer = WebAttackTrainer(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            device='auto'
        )

        # Load trained model
        try:
            self.trainer.load_model(str(model_path))
            self.trainer.class_names = self.config['class_names']
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try retraining the model with the latest version of the code")
            raise

    def load_config(self, config_path):
        """Load model configuration."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  Config file not found at {config_path}, using defaults")
            return {
                'model_name': 'distilbert-base-uncased',
                'num_classes': 4,
                'class_names': ['benign', 'xss', 'csrf', 'sqli']
            }

    def predict_single(self, payload: str, verbose: bool = True):
        """Predict attack type for a single payload."""
        prediction, probabilities, confidence = self.trainer.predict(payload)

        result = {
            'payload': payload,
            'prediction': self.trainer.class_names[prediction],
            'prediction_id': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.trainer.class_names, probabilities))
        }

        if verbose:
            self.print_prediction(result)

        return result

    def predict_batch(self, payloads: list, verbose: bool = True):
        """Predict attack types for multiple payloads."""
        results = []

        for i, payload in enumerate(payloads):
            if verbose:
                print(f"\n--- Prediction {i+1}/{len(payloads)} ---")

            try:
                result = self.predict_single(payload.strip(), verbose)
                results.append(result)
            except Exception as e:
                error_result = {
                    'payload': payload,
                    'error': str(e),
                    'prediction': 'error'
                }
                results.append(error_result)
                if verbose:
                    print(f"❌ Error: {e}")

        return results

    def print_prediction(self, result):
        """Print prediction in a formatted way."""
        print(f"Payload: {result['payload']}")
        print(f"🎯 Prediction: {result['prediction'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.4f}")

        # Sort probabilities by value for better display
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print("📈 Probabilities:")
        for name, prob in sorted_probs:
            bar_length = int(prob * 20)  # Scale to 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {name:8s}: {prob:.4f} |{bar}|")

    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🔍 Interactive Web Attack Detection")
        print("=" * 50)
        print("Enter payloads to analyze (type 'quit' to exit)")
        print("Example payloads:")
        print("  <script>alert('xss')</script>")
        print("  GET /api/users?id=123")
        print("  admin' OR '1'='1' --")
        print("-" * 50)

        while True:
            try:
                payload = input("\n💻 Enter payload: ").strip()

                if payload.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not payload:
                    print("⚠️  Empty payload, please try again.")
                    continue

                print("\n🔄 Analyzing...")
                self.predict_single(payload, verbose=True)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Web Attack Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config file (optional)')

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--payload', type=str,
                            help='Single payload to analyze')
    input_group.add_argument('--file', type=str,
                            help='File containing payloads (one per line)')
    input_group.add_argument('--interactive', action='store_true',
                            help='Run in interactive mode')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON format)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    return parser.parse_args()

def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Results saved to {output_path}")

def main():
    args = parse_args()

    try:
        # Initialize inference
        inference = WebAttackInference(args.model_path, args.config_path)

        if args.interactive:
            # Interactive mode
            inference.interactive_mode()

        elif args.payload:
            # Single payload mode
            print("\n🔍 Analyzing single payload...")
            result = inference.predict_single(args.payload, verbose=not args.quiet)

            if args.output:
                save_results([result], args.output)

        elif args.file:
            # Batch file mode
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {args.file}")
                sys.exit(1)

            print(f"📂 Loading payloads from {args.file}...")
            with open(file_path, 'r') as f:
                payloads = f.readlines()

            print(f"🔍 Analyzing {len(payloads)} payloads...")
            results = inference.predict_batch(payloads, verbose=not args.quiet)

            if args.output:
                save_results(results, args.output)

            # Summary
            if not args.quiet:
                predictions = [r.get('prediction', 'error') for r in results]
                summary = {}
                for pred in predictions:
                    summary[pred] = summary.get(pred, 0) + 1

                print("\n📊 Summary:")
                for pred, count in summary.items():
                    print(f"   {pred}: {count}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
