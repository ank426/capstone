#!/usr/bin/env python3
"""
Training script for Web Attack Detector
Usage: python train.py [--epochs 10] [--batch_size 16] [--learning_rate 2e-5]
"""

import argparse
import pandas as pd
from pathlib import Path
import json

from web_attack_detector import WebAttackTrainer, create_sample_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train Web Attack Detection Model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training data CSV file (optional, uses sample data if not provided)')
    parser.add_argument('--payload_col', type=str, default='payload',
                        help='Name of payload column in CSV')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Name of label column in CSV')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='HuggingFace model name')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--save_path', type=str, default='models/web_attack_model.pt',
                        help='Path to save trained model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')

    return parser.parse_args()

def load_data(data_path: str, payload_col: str, label_col: str):
    """Load training data from CSV file."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if payload_col not in df.columns:
        raise ValueError(f"Payload column '{payload_col}' not found in data")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    print(f"Loaded {len(df)} samples from {data_path}")
    print(f"Label distribution: {df[label_col].value_counts().to_dict()}")

    return df

def save_training_config(args, save_dir: Path):
    """Save training configuration for inference."""
    config = {
        'model_name': args.model_name,
        'num_classes': 4,
        'class_names': ['benign', 'xss', 'csrf', 'sqli'],
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_size': args.test_size
        }
    }

    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to {config_path}")

def main():
    args = parse_args()

    # Create output directory
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or create data
    if args.data_path:
        print("Loading data from file...")
        df = load_data(args.data_path, args.payload_col, args.label_col)
    else:
        print("Using sample data for demonstration...")
        df = create_sample_data()
        # Replicate data for better training
        df = pd.concat([df] * 50, ignore_index=True)
        print(f"Created sample dataset with {len(df)} samples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # Initialize trainer
    print(f"\nInitializing trainer with {args.model_name}...")
    trainer = WebAttackTrainer(
        model_name=args.model_name,
        num_classes=4,
        device=args.device,
        learning_rate=args.learning_rate
    )

    # Prepare data
    print("Preparing training data...")
    train_loader, test_loader = trainer.prepare_data(
        df, args.payload_col, args.label_col,
        test_size=args.test_size,
        batch_size=args.batch_size
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, save_path=str(save_path))

    # Save configuration
    save_training_config(args, save_path.parent)

    # Test on sample payloads
    print("\n" + "="*60)
    print("TESTING ON SAMPLE PAYLOADS")
    print("="*60)

    test_payloads = [
        ('<script>alert("XSS Attack")</script>', 'Expected: XSS'),
        ('GET /api/users?id=123', 'Expected: Benign'),
        ("admin' OR '1'='1' --", 'Expected: SQL Injection'),
        ('POST /admin/delete_user?id=456', 'Expected: CSRF'),
        ('<img src=x onerror=alert(1)>', 'Expected: XSS'),
        ('SELECT * FROM users WHERE id=1', 'Expected: SQL Injection')
    ]

    for payload, expected in test_payloads:
        try:
            prediction, probabilities, confidence = trainer.predict(payload)
            print(f"\nPayload: {payload}")
            print(f"Prediction: {trainer.class_names[prediction]} ({expected})")
            print(f"Confidence: {confidence:.4f}")
            prob_dict = {name: f"{prob:.4f}" for name, prob in zip(trainer.class_names, probabilities)}
            print(f"Probabilities: {prob_dict}")
        except Exception as e:
            print(f"Error predicting payload: {e}")

    print(f"\n✅ Training completed! Model saved to {save_path}")
    print(f"💡 Use 'python run.py --model_path {save_path}' to run inference")

if __name__ == "__main__":
    main()
