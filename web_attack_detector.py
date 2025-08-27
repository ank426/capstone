import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

class WebAttackDataset(Dataset):
    """Dataset class for web attack detection."""

    def __init__(self, payloads, labels, tokenizer, max_length=512):
        self.payloads = payloads
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, idx):
        payload = str(self.payloads[idx])
        label = self.labels[idx]

        # Tokenize payload
        encoding = self.tokenizer(
            payload,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'payload': payload
        }

class FeatureExtractor:
    """Extracts rule-based features from web requests."""

    @staticmethod
    def extract_xss_features(payload: str) -> np.ndarray:
        """Extract XSS-specific features."""
        features = {}

        try:
            soup = BeautifulSoup(payload, 'html.parser')
            features['malformed'] = 0
        except:
            soup = BeautifulSoup('', 'html.parser')
            features['malformed'] = 1

        js_code = "".join(script.get_text(strip=True) for script in soup.find_all('script'))

        # URL features
        url_match = re.search(r'https?://[^\s\'"]+', payload)
        url = url_match.group(0) if url_match else 'http://example.com'
        # urlparse(url)

        features['url_length'] = len(url)
        features['url_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
        features['url_script_tag'] = 1 if re.search(r'<script', url, re.IGNORECASE) else 0
        features['url_iframe_tag'] = 1 if re.search(r'<iframe', url, re.IGNORECASE) else 0

        # HTML tag counts
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'svg', 'form']
        for tag in dangerous_tags:
            features[f'html_{tag}_count'] = len(soup.find_all(tag))

        # Event handlers
        event_handlers = ['onclick', 'onload', 'onerror', 'onmouseover']
        for event in event_handlers:
            features[f'event_{event}'] = len(soup.find_all(attrs={event: True}))

        # JavaScript features
        js_keywords = ['eval', 'alert', 'document.cookie', 'window.location']
        features['js_keywords'] = sum(payload.lower().count(kw) for kw in js_keywords)
        features['js_length'] = len(js_code)

        # Convert to array
        return np.array(list(features.values()), dtype=np.float32)

    @staticmethod
    def extract_csrf_features(payload: str) -> np.ndarray:
        """Extract CSRF-specific features."""
        features = {}
        payload_lower = payload.lower()

        features['request_length'] = len(payload)

        # Extract URL and parameters
        url_match = re.search(r'https?://[^\'\"`\s<>]+', payload)
        all_params = {}

        if url_match:
            parsed_url = urlparse(url_match.group(0))
            parsed_url.path.lower()
            all_params.update(parse_qs(parsed_url.query))

        features['num_params'] = len(all_params)

        # Sensitive keywords in path/params
        sensitive_keywords = ['password', 'token', 'login', 'delete', 'update', 'create']
        for keyword in sensitive_keywords:
            features[f'{keyword}_in_request'] = 1 if keyword in payload_lower else 0

        # HTTP methods
        http_methods = ['post', 'put', 'delete', 'get']
        for method in http_methods:
            features[f'is_{method}'] = 1 if method in payload_lower else 0

        return np.array(list(features.values()), dtype=np.float32)

    @staticmethod
    def extract_sqli_features(payload: str) -> np.ndarray:
        """Extract SQL injection features."""
        features = {}
        payload_lower = payload.lower()

        # SQL keywords
        sql_keywords = ['select', 'union', 'insert', 'update', 'delete', 'drop',
                       'where', 'and', 'or', 'exec', 'sleep', 'benchmark']
        for keyword in sql_keywords:
            features[f'sql_{keyword}'] = payload_lower.count(keyword)

        # SQL comment patterns
        features['sql_comments'] = payload.count('--') + payload.count('#') + payload.count('/*')

        # Common injection patterns
        injection_patterns = ["1=1", "'='", "or 1=1", "' or '1'='1"]
        features['injection_patterns'] = sum(payload_lower.count(pattern) for pattern in injection_patterns)

        # Special characters
        special_chars = ['=', "'", '"', ';', '(', ')']
        features['special_chars'] = sum(payload.count(char) for char in special_chars)

        features['payload_length'] = len(payload)

        return np.array(list(features.values()), dtype=np.float32)

class HybridAttackDetector(nn.Module):
    """Hybrid model combining transformer embeddings with rule-based features."""

    def __init__(self, model_name='distilbert-base-uncased', num_classes=4,
                 rule_feature_size=50, hidden_size=256):
        super().__init__()

        # Transformer for semantic understanding
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_hidden_size = self.transformer.config.hidden_size

        # Don't freeze transformer layers - allow fine-tuning
        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        # Rule-based feature processing
        self.rule_processor = nn.Sequential(
            nn.Linear(rule_feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

        # Combined feature processing
        combined_size = transformer_hidden_size + hidden_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask, rule_features):
        # Get transformer embeddings
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # For DistilBERT, use last_hidden_state and apply mean pooling
        last_hidden_state = transformer_output.last_hidden_state
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Process rule-based features
        rule_output = self.rule_processor(rule_features)

        # Combine features
        combined = torch.cat([pooled_output, rule_output], dim=1)

        # Classification
        logits = self.classifier(combined)
        return logits

class WebAttackTrainer:
    """Trainer class for the web attack detection model."""

    def __init__(self, model_name='distilbert-base-uncased', num_classes=4,
                 device='auto', learning_rate=2e-5):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = FeatureExtractor()

        # Initialize model with proper feature size
        rule_feature_size = self._calculate_feature_size()
        self.model = HybridAttackDetector(
            model_name=model_name,
            num_classes=num_classes,
            rule_feature_size=rule_feature_size
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # Train all parameters now
            lr=learning_rate,
            weight_decay=0.01
        )

        self.criterion = nn.CrossEntropyLoss()
        self.class_names = ['benign', 'xss', 'csrf', 'sqli']

    def _calculate_feature_size(self):
        """Calculate the size of rule-based features."""
        dummy_payload = "test"
        xss_features = self.feature_extractor.extract_xss_features(dummy_payload)
        csrf_features = self.feature_extractor.extract_csrf_features(dummy_payload)
        sqli_features = self.feature_extractor.extract_sqli_features(dummy_payload)
        return len(xss_features) + len(csrf_features) + len(sqli_features)

    def extract_features(self, payload: str) -> np.ndarray:
        """Extract combined rule-based features."""
        xss_features = self.feature_extractor.extract_xss_features(payload)
        csrf_features = self.feature_extractor.extract_csrf_features(payload)
        sqli_features = self.feature_extractor.extract_sqli_features(payload)
        return np.hstack([xss_features, csrf_features, sqli_features])

    def prepare_data(self, df: pd.DataFrame, payload_col: str, label_col: str,
                    test_size: float = 0.2, batch_size: int = 32):
        """Prepare data loaders for training."""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[payload_col].values, df[label_col].values,
            test_size=test_size, stratify=df[label_col].values, random_state=42
        )

        # Create datasets
        train_dataset = WebAttackDataset(X_train, y_train, self.tokenizer)
        test_dataset = WebAttackDataset(X_test, y_test, self.tokenizer)

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return self.train_loader, self.test_loader

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Extract rule-based features for the batch
            rule_features = []
            for payload in batch['payload']:
                features = self.extract_features(payload)
                rule_features.append(features)
            rule_features = torch.tensor(np.array(rule_features), dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask, rule_features)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Extract rule-based features
                rule_features = []
                for payload in batch['payload']:
                    features = self.extract_features(payload)
                    rule_features.append(features)
                rule_features = torch.tensor(np.array(rule_features), dtype=torch.float32).to(self.device)

                logits = self.model(input_ids, attention_mask, rule_features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self, epochs: int = 10, save_path: str = 'web_attack_model.pt'):
        """Train the model."""
        best_accuracy = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Evaluation
            val_loss, val_acc, predictions, labels = self.evaluate()
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                # Save with weights_only=False for compatibility
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'class_names': self.class_names
                }, save_path, _use_new_zipfile_serialization=False)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")

        # Final evaluation
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=self.class_names))

    def predict(self, payload: str) -> Tuple[int, np.ndarray, float]:
        """Predict attack type for a single payload."""
        self.model.eval()

        # Tokenize payload
        encoding = self.tokenizer(
            payload,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # Extract rule-based features
        rule_features = self.extract_features(payload)
        rule_features = torch.tensor(rule_features, dtype=torch.float32).unsqueeze(0)

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        rule_features = rule_features.to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, rule_features)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]

        return (
            prediction.cpu().item(),
            probabilities.cpu().numpy().flatten(),
            confidence.cpu().item()
        )

    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            # Try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Failed to load with weights_only=False: {e}")
            # Fallback: try with safe globals
            try:
                import numpy as np
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except:
                # Final fallback: load with weights_only=False and suppress warning
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.class_names = checkpoint.get('class_names', self.class_names)
        print(f"Model loaded from {model_path}")

def create_sample_data():
    """Create sample data for demonstration."""
    # More diverse and realistic sample data
    sample_data = {
        'payload': [
            # Benign requests
            'GET /index.php?id=1',
            'GET /search?q=python programming',
            'POST /login HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\nusername=user&password=pass',
            'GET /api/users/profile',
            'GET /static/css/style.css',
            'POST /contact HTTP/1.1\nContent-Type: application/json\n\n{"name":"John","email":"john@example.com"}',

            # XSS attacks
            '<script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            '<iframe src="javascript:alert(1)"></iframe>',
            'javascript:alert(document.cookie)',
            '<svg onload=alert("XSS")>',
            '<input type="text" value="" onfocus="alert(\'XSS\')" autofocus>',

            # CSRF attacks
            'POST /transfer HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\namount=1000&to=attacker',
            'GET /admin/delete_user?id=123',
            'POST /settings/change_password HTTP/1.1\n\nnew_password=hacked&confirm=hacked',
            'DELETE /api/user/456 HTTP/1.1',
            'PUT /admin/users/789 HTTP/1.1\n\nrole=admin',
            'POST /payment/process HTTP/1.1\n\namount=999&account=evil',

            # SQL injection attacks
            "' OR '1'='1' --",
            "admin' AND (SELECT COUNT(*) FROM users) > 0 --",
            "1' UNION SELECT username,password FROM users--",
            "'; DROP TABLE users; --",
            "1' OR SLEEP(5)--",
            "' OR 1=1 LIMIT 1 OFFSET 0 --"
        ],
        'label': [
            # Benign (0)
            0, 0, 0, 0, 0, 0,
            # XSS (1)
            1, 1, 1, 1, 1, 1,
            # CSRF (2)
            2, 2, 2, 2, 2, 2,
            # SQLi (3)
            3, 3, 3, 3, 3, 3
        ],
        'attack_type': [
            # Benign
            'benign', 'benign', 'benign', 'benign', 'benign', 'benign',
            # XSS
            'xss', 'xss', 'xss', 'xss', 'xss', 'xss',
            # CSRF
            'csrf', 'csrf', 'csrf', 'csrf', 'csrf', 'csrf',
            # SQLi
            'sqli', 'sqli', 'sqli', 'sqli', 'sqli', 'sqli'
        ]
    }
    return pd.DataFrame(sample_data)
