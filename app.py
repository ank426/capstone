from gensim.models import KeyedVectors
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any
import torch.nn.functional as F
import streamlit as st
import plotly.express as px

WORD2VEC_DIM = 100
FASTTEXT_DIM = 300

INPUT_SIZE = 1069
NUM_CLASSES = 4
LEARNING_RATE = 0.001
EPOCHS = 200
device = "cuda"


def get_word_embedding(payload, model, dim):
    tokens = payload.lower().split()
    vectors = [model[word] for word in tokens if word in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

def get_sentence_embedding(payload, model):
    return model([payload]).numpy().flatten()

def extract_uniembed_features(payload, word2vec_model, fasttext_model, use_model):
    cleaned = re.sub(r'<[^>]+>', ' ', payload).strip()
    w2v = get_word_embedding(cleaned, word2vec_model, WORD2VEC_DIM)
    ft = get_word_embedding(cleaned, fasttext_model, FASTTEXT_DIM)
    use = get_sentence_embedding(payload, use_model)
    return np.hstack([w2v, ft, use])

def extract_xss_68_features(payload):
    features = {}

    # --- Pre-processing ---
    # The raw payload is used for both parsing and direct string analysis.
    html_content = payload
    try:
      soup = BeautifulSoup(html_content, 'html.parser')
      features['malformed']=0
    except Exception as e:
      soup = BeautifulSoup('', 'html.parser')
      features['malformed']=1

    js_code = "".join(script.get_text(strip=True) for script in soup.find_all('script'))

    # Attempt to find a URL within the payload, otherwise use a default
    url_match = re.search(r'https?://[^\s\'"]+', html_content)
    url = url_match.group(0) if url_match else 'http://example.com'
    parsed_url = urlparse(url)

    # Category 1: URL Features
    features['url_length'] = len(url)
    features['url_special_characters'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['url_tag_script'] = 1 if re.search(r'<script', url, re.IGNORECASE) else 0
    features['url_tag_iframe'] = 1 if re.search(r'<iframe', url, re.IGNORECASE) else 0
    features['url_attr_src'] = 1 if re.search(r'src=', url, re.IGNORECASE) else 0
    features['url_event_onload'] = 1 if re.search(r'onload=', url, re.IGNORECASE) else 0
    features['url_event_onmouseover'] = 1 if re.search(r'onmouseover=', url, re.IGNORECASE) else 0
    features['url_cookie'] = 1 if 'cookie' in url.lower() else 0
    features['url_number_keywords_param'] = len(re.findall(r'(alert|script|onerror|onload|eval)', parsed_url.query, re.IGNORECASE))
    features['url_number_domain'] = len(parsed_url.hostname.split('.')) if parsed_url.hostname else 0

    # Category 2: HTML Tag Features (from parsed payload)
    tags = ['script', 'iframe', 'meta', 'object', 'embed', 'link', 'svg', 'frame', 'form', 'div', 'style', 'img', 'input', 'textarea']
    for tag in tags:
        features[f'html_tag_{tag}'] = len(soup.find_all(tag))

    # Category 3: HTML Attribute Features (from parsed payload)
    attrs = ['action', 'background', 'classid', 'codebase', 'href', 'longdesc', 'profile', 'src', 'usemap']
    for attr in attrs:
        features[f'html_attr_{attr}'] = len(soup.find_all(attrs={attr: True}))
    features['html_attr_http-equiv'] = len(soup.find_all(attrs={'http-equiv': True}))

    # Category 4: HTML Event Handler Features (from parsed payload)
    events = ['onblur', 'onchange', 'onclick', 'onerror', 'onfocus', 'onkeydown', 'onkeypress', 'onkeyup', 'onload', 'onmousedown', 'onmouseout', 'onmouseover', 'onmouseup', 'onsubmit']
    for event in events:
        features[f'html_event_{event}'] = len(soup.find_all(attrs={event: True}))

    # Category 5: JavaScript & Content Features (from raw payload and extracted JS)
    evil_keywords = ['eval', 'alert', 'prompt', 'confirm', 'document.cookie', 'window.location', 'unescape']
    features['html_number_keywords_evil'] = sum(html_content.lower().count(kw) for kw in evil_keywords)
    features['js_file'] = len(soup.find_all('script', src=True))
    features['js_pseudo_protocol'] = len(re.findall(r'javascript:', html_content, re.IGNORECASE))
    features['js_dom_location'] = js_code.lower().count('location')
    features['js_dom_document'] = js_code.lower().count('document')
    features['js_prop_cookie'] = js_code.lower().count('.cookie')
    features['js_prop_referrer'] = js_code.lower().count('.referrer')

    js_methods = ['write', 'getElementsByTagName', 'getElementById', 'alert', 'eval', 'fromCharCode', 'confirm']
    for method in js_methods:
        features[f'js_method_{method}'] = len(re.findall(rf'[\b\.]({method})\s*\(', js_code, re.IGNORECASE))

    features['js_min_length'] = len(js_code)
    features['js_min_define_function'] = len(re.findall(r'function\s*[\w]*\s*\(', js_code))
    total_calls = len(re.findall(r'\w+\s*\(', js_code))
    features['js_min_function_calls'] = max(0, total_calls - features['js_min_define_function'])
    js_strings = re.findall(r'["\'](.*?)["\']', js_code)
    features['js_string_max_length'] = max(len(s) for s in js_strings) if js_strings else 0

    # Category 6: Final General Feature
    features['html_length'] = len(html_content)

    # Return as a numpy array in a consistent order
    feature_order = sorted(features.keys())
    return np.array([features[k] for k in feature_order])

def extract_rule_based_csrf_features(payload: str) -> np.ndarray:
    # This is your provided function for CSRF rule-based features
    feature_names=[
        'numOfParams', 'numOfBools', 'numOfIds', 'numOfBlobs', 'reqLen', 'createInPath', 'createInParams', 'addInPath', 'addInParams', 'setInPath', 'setInParams', 'deleteInPath', 'deleteInParams', 'updateInPath', 'updateInParams', 'removeInPath', 'removeInParams', 'friendInPath', 'friendInParams', 'settingInPath', 'settingInParams', 'passwordInPath', 'passwordInParams', 'tokenInPath', 'tokenInParams', 'changeInPath', 'changeInParams', 'actionInPath', 'actionInParams', 'payInPath', 'payInParams', 'loginInPath', 'loginInParams', 'logoutInPath', 'logoutInParams', 'postInPath', 'postInParams', 'commentInPath', 'commentInParams', 'followInPath', 'followInParams', 'subscribeInPath', 'subscribeInParams', 'signInPath', 'signInParams', 'viewInPath', 'viewInParams', 'isPUT', 'isDELETE', 'isPOST', 'isGET', 'isOPTIONS'
    ]
    keywords=[
        'create', 'add', 'set', 'delete', 'update', 'remove', 'friend', 'setting', 'password', 'token', 'change', 'action', 'pay', 'login', 'logout', 'post', 'comment', 'follow', 'subscribe', 'sign', 'view'
    ]
    features = {name: 0 for name in feature_names}
    payload_lower = payload.lower()
    features['reqLen'] = len(payload)
    url_match = re.search(r'https?://[^\'\"`\s<>]+', payload)
    path = ""
    all_params = {}
    if url_match:
        url = url_match.group(0)
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        all_params.update(parse_qs(parsed_url.query))
    form_param_match = re.search(r"send\('([^']+)'\)", payload)
    if form_param_match:
        all_params.update(parse_qs(form_param_match.group(1)))
    form_inputs = re.findall(r"<input[^>]+name=['\"]([^'\"]+)['\"][^>]+value=['\"]([^'\"]+)['\"]", payload)
    for name, value in form_inputs:
        if name not in all_params:
            all_params[name] = []
        all_params[name].append(value)
    features['numOfParams'] = len(all_params)
    for key, values in all_params.items():
        if 'id' in key.lower():
            features['numOfIds'] += 1
        for value in values:
            if value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                features['numOfBools'] += 1
    features['numOfBlobs'] = 0
    params_str = str(all_params.keys()).lower()
    for keyword in keywords:
        if keyword in path:
            features[f'{keyword}InPath'] = 1
        if keyword in params_str:
            features[f'{keyword}InParams'] = 1
    if 'method' in payload_lower and 'post' in payload_lower or "xhr.open('post'" in payload_lower:
        features['isPOST'] = 1
    elif 'method' in payload_lower and 'put' in payload_lower or "xhr.open('put'" in payload_lower:
        features['isPUT'] = 1
    elif 'method' in payload_lower and 'delete' in payload_lower or "xhr.open('delete'" in payload_lower:
        features['isDELETE'] = 1
    elif 'method' in payload_lower and 'options' in payload_lower or "xhr.open('options'" in payload_lower:
        features['isOPTIONS'] = 1
    else:
        features['isGET'] = 1
    return np.array([features[name] for name in feature_names])

def extract_sqli_features(payload: str) -> np.ndarray:
    features = []
    payload_lower = payload.lower()

    # 1. SQL Keywords
    sql_keywords = [
        'select', 'union', 'insert', 'update', 'delete', 'drop', 'from',
        'where', 'and', 'or', 'limit', 'order by', 'group by', 'exec',
        'declare', 'cast', 'convert', 'sleep', 'benchmark', 'waitfor'
    ]
    features.extend([payload_lower.count(kw) for kw in sql_keywords])

    # 2. SQL Comment Characters
    features.append(payload.count('--'))
    features.append(payload.count('#'))
    features.append(payload.count('/*'))
    features.append(payload.count('*/'))

    # 3. Tautologies and Common Bypass Patterns
    tautologies = ["1=1", "'='", "or 1=1", "or '1'='1'"]
    features.extend([payload_lower.count(t) for t in tautologies])

    # 4. Special Characters and Operators
    special_chars = ['=', "'", "\"", ";", "(", ")"]
    features.extend([payload.count(char) for char in special_chars])

    # 5. Hex Encoded Characters
    features.append(len(re.findall(r'%[0-9a-f]{2}', payload_lower)))

    # 6. Overall Payload Length
    features.append(len(payload))

    # 7. Whitespace characters (often used for obfuscation)
    features.append(len(re.findall(r'\\s', payload)))

    return np.array(features, dtype=np.float32)

def create_unified_feature_vector(row):
    payload = row['payload']
    label = row['class']
    
    nlp_features = extract_uniembed_features(payload)
    xss_rules = extract_xss_68_features(payload)
    csrf_rules = extract_rule_based_csrf_features(payload)
    sqli_rules = extract_sqli_features(payload)
        
    return np.hstack([xss_rules, csrf_rules, sqli_rules, nlp_features])

def classify_request_mlp(payload, mlp_model=None, device='cpu'):
    mlp_model.eval()

    try:
        xss_rules = extract_xss_68_features(payload)
        csrf_rules = extract_rule_based_csrf_features(payload)
        sqli_rules = extract_sqli_features(payload)
        nlp_features = extract_uniembed_features(payload)
        combined_features = np.hstack([xss_rules, csrf_rules, sqli_rules, nlp_features])
        
        feature_tensor = torch.tensor(combined_features, dtype=torch.float32)
        feature_tensor = feature_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = mlp_model(feature_tensor)
            
            # This line requires 'import torch.nn.functional as F'
            probabilities_tensor = F.softmax(logits, dim=1)
            
            confidence_tensor, prediction_tensor = torch.max(probabilities_tensor, 1)

        prediction = prediction_tensor.cpu().item()
        probabilities = probabilities_tensor.cpu().numpy().flatten()
        confidence = confidence_tensor.cpu().item()
        
        return prediction, probabilities, confidence

    except Exception as e:
        print(f"An error occurred during MLP classification: {e}")

class MLP(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(MLP, self).__init__()
        # Layer 1: 1032 input features -> 256 output features
        self.layer1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 256 input features -> 128 output features
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        # Layer 3: 128 input features -> 64 output features
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()

        # Output Layer: 64 input features -> 3 output features (for 3 classes)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.output_layer(x)
        return x

@st.cache_resource
def load_all_resources():
    """
    Loads all models and the inference class once and caches them.
    This function will only run the first time the app is started.
    """
    print("--- Running load_all_resources() ---")
    print("Loading pre-trained NLP models...")
    
    # --- OPTIMIZATION 2: Load GloVe from the FAST binary format ---
    # Make sure you've converted your .word2vec file to .bin first!
    GLOVE_PATH_BIN = "/kaggle/input/nlp-models/tensorflow2/default/3/glove-wiki-gigaword-100.bin" # UPDATE THIS PATH
    FASTTEXT_PATH = "/kaggle/input/nlp-models/tensorflow2/default/3/fasttext.model"
    USE_PATH = "/kaggle/input/nlp-models/tensorflow2/default/3/universal_sentence_encoder"

    word2vec_model = KeyedVectors.load_word2vec_format(GLOVE_PATH_BIN, binary=True)
    fasttext_model = KeyedVectors.load(FASTTEXT_PATH)
    use_model = hub.load(USE_PATH)
    print("✅ NLP Models loaded.")

    print("Loading MLP model...")
    
    mlp_model = MLP(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    MLP_MODEL_FILENAME = '/kaggle/input/xss_csrf_trained_models/other/default/4/mlp_xss_csrf_sqli_detector.pth'
    mlp_model.load_state_dict(torch.load(MLP_MODEL_FILENAME, map_location=device))
    mlp_model.eval() # Set model to evaluation mode
    print("✅ MLP Model loaded.")

    # Pass all loaded resources to the inference class
    inference_handler = WebAttackInference(mlp_model, word2vec_model, fasttext_model, use_model, device)
    
    return inference_handler

class WebAttackInference:
    def __init__(self, mlp_model, w2v_model, ft_model, use_model, device):
        """
        Initialize with pre-loaded models to avoid loading them inside methods.
        """
        # Models
        self.mlp_model = mlp_model
        self.word2vec_model = w2v_model
        self.fasttext_model = ft_model
        self.use_model = use_model
        self.device = device
        
        # Class and config info
        self.class_names = ['benign', 'xss', 'csrf', 'sqli']
        self.config = {'model_name': 'Custom MLP (Optimized)', 'class_names': self.class_names}

    # --- OPTIMIZATION 3: Cache the results of the prediction function ---
    @st.cache_data
    def predict_single(_self, payload: str) -> Dict[str, Any]:
        """
        The '_self' is used because st.cache_data doesn't hash 'self' objects correctly.
        This function now uses the pre-loaded models from the instance.
        """
        try:
            # --- Feature Extraction ---
            # You would need to refactor your feature extraction functions to accept the models as arguments
            # For simplicity here, we assume they are globally accessible or passed correctly.
            # Example: nlp_features = extract_uniembed_features(payload, _self.word2vec_model, _self.fasttext_model, _self.use_model)
            xss_rules = extract_xss_68_features(payload)
            csrf_rules = extract_rule_based_csrf_features(payload)
            sqli_rules = extract_sqli_features(payload)
            nlp_features = extract_uniembed_features(payload, _self.word2vec_model, _self.fasttext_model, _self.use_model) # This needs access to global models or pass them in
            
            combined_features = np.hstack([xss_rules, csrf_rules, sqli_rules, nlp_features])
            feature_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(_self.device)

            with torch.no_grad():
                logits = _self.mlp_model(feature_tensor)
                probabilities_tensor = F.softmax(logits, dim=1)
                confidence_tensor, prediction_tensor = torch.max(probabilities_tensor, 1)

            prediction = prediction_tensor.cpu().item()
            probabilities = probabilities_tensor.cpu().numpy().flatten()
            confidence = confidence_tensor.cpu().item()

            return {
                'payload': payload,
                'prediction': _self.class_names[prediction],
                'prediction_id': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(_self.class_names, probabilities.tolist()))
            }
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return {'payload': payload, 'prediction': 'Error'}


def create_probability_chart(probabilities):
    df = pd.DataFrame(list(probabilities.items()), columns=['Attack Type', 'Probability'])
    df = df.sort_values('Probability', ascending=True)
    fig = px.bar(df, x='Probability', y='Attack Type', orientation='h', title='Attack Type Probabilities', text='Probability')
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(showlegend=False, height=300, xaxis=dict(range=[0, 1]))
    return fig

def create_batch_summary_chart(results):
    predictions = [r.get('prediction', 'error') for r in results]
    summary = pd.Series(predictions).value_counts()
    fig = px.pie(values=summary.values, names=summary.index, title='Batch Analysis Summary')
    return fig


def main():
    st.set_page_config(page_title="Web Attack Detector", layout="wide")

    st.title("🛡️ Web Attack Detector")
    st.markdown("Detect and classify web attacks using a custom MLP model.")

    with st.spinner("🚀 Initializing models... (this will be slow only on first startup)"):
        inference = load_all_resources()

    with st.sidebar.expander("📊 Model Info", expanded=True):
        st.write(f"**Model:** {inference.config['model_name']}")
        st.write(f"**Classes:** {inference.config['class_names']}")

    tab1 = st.tabs(["🔍 Single Analysis"])[0]

    with tab1:
        st.header("Single Payload Analysis")
        payload = st.text_area("Enter payload to analyze:", placeholder="e.g., <script>alert('xss')</script>", height=150)
        if st.button("🎯 Analyze Payload", type="primary", use_container_width=True):
            if payload.strip():
                with st.spinner("Analyzing..."):
                    result = inference.predict_single(payload)
                prediction, confidence = result['prediction'], result['confidence']
                st.subheader("Analysis Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", prediction.upper(), "Benign" if prediction == 'benign' else "Malicious")
                    st.metric("Confidence Score", f"{confidence:.2%}")
                with col2:
                    st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
            else:
                st.warning("Please enter a payload.")

if __name__ == "__main__":
    main()