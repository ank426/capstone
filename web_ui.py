#!/usr/bin/env python3
"""
Streamlit Web UI for Web Attack Detector
Usage: streamlit run web_ui.py
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys
import os

# Add the current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web_attack_detector import WebAttackTrainer
except ImportError:
    st.error("⚠️ Could not import WebAttackTrainer. Make sure web_attack_detector.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Web Attack Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WebAttackInference:
    """Inference class for web attack detection (Streamlit version)."""

    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load configuration
        if config_path is None:
            config_path = self.model_path.parent / 'config.json'

        self.config = self.load_config(config_path)

        # Initialize trainer for inference
        self.trainer = WebAttackTrainer(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            device='auto'
        )

        # Load trained model
        self.trainer.load_model(str(model_path))
        self.trainer.class_names = self.config['class_names']

    def load_config(self, config_path):
        """Load model configuration."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'model_name': 'distilbert-base-uncased',
                'num_classes': 4,
                'class_names': ['benign', 'xss', 'csrf', 'sqli']
            }

    def predict_single(self, payload: str):
        """Predict attack type for a single payload."""
        prediction, probabilities, confidence = self.trainer.predict(payload)

        result = {
            'payload': payload,
            'prediction': self.trainer.class_names[prediction],
            'prediction_id': prediction,
            'confidence': confidence,
            'probabilities': {
                name: prob for name, prob in zip(self.trainer.class_names, probabilities)
            }
        }

        return result

    def predict_batch(self, payloads: list):
        """Predict attack types for multiple payloads."""
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, payload in enumerate(payloads):
            try:
                result = self.predict_single(payload.strip())
                results.append(result)
            except Exception as e:
                error_result = {
                    'payload': payload,
                    'error': str(e),
                    'prediction': 'error'
                }
                results.append(error_result)

            # Update progress
            progress = (i + 1) / len(payloads)
            progress_bar.progress(progress)
            status_text.text(f'Processed {i+1}/{len(payloads)} payloads')

        progress_bar.empty()
        status_text.empty()

        return results

@st.cache_resource
def load_model(model_path, config_path=None):
    """Load model with caching."""
    try:
        return WebAttackInference(model_path, config_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def create_probability_chart(probabilities):
    """Create a horizontal bar chart for probabilities."""
    df = pd.DataFrame(list(probabilities.items()), columns=['Attack Type', 'Probability'])
    df = df.sort_values('Probability', ascending=True)

    # Create color mapping
    colors = ['#ff4444', '#ff8800', '#ffdd00', '#44ff44']  # red to green
    color_map = dict(zip(sorted(probabilities.keys()), colors))
    df['Color'] = df['Attack Type'].map(color_map)

    fig = px.bar(df, x='Probability', y='Attack Type', orientation='h',
                 color='Attack Type', color_discrete_map=color_map,
                 title='Attack Type Probabilities')

    fig.update_layout(showlegend=False, height=300, xaxis=dict(range=[0, 1]))

    return fig

def create_batch_summary_chart(results):
    """Create a pie chart for batch results summary."""
    predictions = [r.get('prediction', 'error') for r in results]
    summary = {}
    for pred in predictions:
        summary[pred] = summary.get(pred, 0) + 1

    fig = px.pie(values=list(summary.values()), names=list(summary.keys()),
                 title='Batch Analysis Summary')
    return fig

def main():
    # Header
    st.title("🛡️ Web Attack Detector")
    st.markdown("Detect and classify web attacks using machine learning")

    # Sidebar for model configuration
    st.sidebar.header("⚙️ Model Configuration")

    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/web_attack_model.pt",
        help="Path to your trained model file"
    )

    config_path = st.sidebar.text_input(
        "Config Path (optional)",
        value="",
        help="Path to model config file. Leave empty to use default location."
    )

    # Load model button
    if st.sidebar.button("🔄 Load Model"):
        st.session_state.model_loaded = False
        st.session_state.inference = None

    # Load model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            config_path_val = config_path if config_path else None
            inference = load_model(model_path, config_path_val)
            if inference:
                st.session_state.inference = inference
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model path.")
                st.stop()
    else:
        inference = st.session_state.inference

    # Display model info
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Ready")
        with st.sidebar.expander("📊 Model Info"):
            st.write(f"**Model:** {inference.config['model_name']}")
            st.write(f"**Classes:** {len(inference.config['class_names'])}")
            st.write("**Class Names:**")
            for i, name in enumerate(inference.config['class_names']):
                st.write(f"  {i}: {name}")

    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📂 Batch Analysis", "ℹ️ Examples"])

    with tab1:
        st.header("Single Payload Analysis")

        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Text Area (for long payloads)"],
            horizontal=True
        )

        if input_method == "Text Input":
            payload = st.text_input(
                "Enter payload to analyze:",
                placeholder="e.g., <script>alert('xss')</script>",
                help="Enter a web request or payload to analyze"
            )
        else:
            payload = st.text_area(
                "Enter payload to analyze:",
                placeholder="Enter your payload here...\nMultiple lines are supported.",
                height=150,
                help="Enter a web request or payload to analyze"
            )

        if st.button("🎯 Analyze Payload", type="primary"):
            if payload.strip():
                with st.spinner("Analyzing..."):
                    result = inference.predict_single(payload)

                # Display results
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🎯 Results")

                    # Prediction with colored badge
                    prediction = result['prediction'].upper()
                    confidence = result['confidence']

                    if prediction == 'BENIGN':
                        st.success(f"**Prediction:** {prediction}")
                    elif prediction in ['XSS', 'CSRF', 'SQLI']:
                        st.error(f"**Prediction:** {prediction}")
                    else:
                        st.warning(f"**Prediction:** {prediction}")

                    st.metric("Confidence Score", f"{confidence:.4f}")

                    # Probabilities table
                    st.subheader("📊 Detailed Probabilities")
                    prob_df = pd.DataFrame(
                        list(result['probabilities'].items()),
                        columns=['Attack Type', 'Probability']
                    ).sort_values('Probability', ascending=False)
                    prob_df['Probability'] = prob_df['Probability'].round(4)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                with col2:
                    # Probability chart
                    fig = create_probability_chart(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)

                # Payload display
                st.subheader("💻 Analyzed Payload")
                st.code(payload, language="text")

            else:
                st.warning("Please enter a payload to analyze.")

    with tab2:
        st.header("Batch Payload Analysis")

        # Input methods for batch
        batch_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"],
            horizontal=True
        )

        payloads = []

        if batch_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload file with payloads (one per line)",
                type=['txt', 'csv'],
                help="Upload a text file with one payload per line"
            )

            if uploaded_file is not None:
                content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                payloads = [line.strip() for line in content.split('\n') if line.strip()]
                st.success(f"📂 Loaded {len(payloads)} payloads from file")

        else:
            batch_text = st.text_area(
                "Paste payloads (one per line):",
                height=200,
                placeholder="<script>alert('xss')</script>\nGET /api/users?id=123\nadmin' OR '1'='1' --",
                help="Enter multiple payloads, one per line"
            )

            if batch_text:
                payloads = [line.strip() for line in batch_text.split('\n') if line.strip()]
                st.info(f"📝 Found {len(payloads)} payloads")

        if payloads and st.button("🚀 Analyze All Payloads", type="primary"):
            st.subheader("🔄 Processing...")
            results = inference.predict_batch(payloads)

            # Display summary
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 Summary")
                predictions = [r.get('prediction', 'error') for r in results]
                summary = {}
                for pred in predictions:
                    summary[pred] = summary.get(pred, 0) + 1

                summary_df = pd.DataFrame(
                    list(summary.items()),
                    columns=['Attack Type', 'Count']
                ).sort_values('Count', ascending=False)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            with col2:
                # Summary chart
                fig = create_batch_summary_chart(results)
                st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            st.subheader("📋 Detailed Results")

            # Create results dataframe
            results_data = []
            for i, result in enumerate(results):
                if 'error' not in result:
                    results_data.append({
                        '#': i + 1,
                        'Payload': result['payload'][:100] + '...' if len(result['payload']) > 100 else result['payload'],
                        'Prediction': result['prediction'].upper(),
                        'Confidence': round(result['confidence'], 4),
                        'Full Payload': result['payload']
                    })
                else:
                    results_data.append({
                        '#': i + 1,
                        'Payload': result['payload'][:100] + '...' if len(result['payload']) > 100 else result['payload'],
                        'Prediction': 'ERROR',
                        'Confidence': 0.0,
                        'Full Payload': result['payload']
                    })

            results_df = pd.DataFrame(results_data)

            # Display with filtering
            attack_filter = st.multiselect(
                "Filter by prediction:",
                options=results_df['Prediction'].unique(),
                default=results_df['Prediction'].unique()
            )

            filtered_df = results_df[results_df['Prediction'].isin(attack_filter)]
            st.dataframe(
                filtered_df[['#', 'Payload', 'Prediction', 'Confidence']],
                use_container_width=True,
                hide_index=True
            )

            # Download results
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="💾 Download Results (JSON)",
                data=results_json,
                file_name="web_attack_results.json",
                mime="application/json"
            )

    with tab3:
        st.header("💡 Example Payloads")
        st.markdown("Here are some example payloads you can use to test the detector:")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟢 Benign Examples")
            benign_examples = [
                "GET /api/users HTTP/1.1",
                "POST /login username=user&password=pass",
                "SELECT * FROM users WHERE active = 1",
                "Hello, how are you today?"
            ]

            for example in benign_examples:
                if st.button(f"Try: {example[:30]}...", key=f"benign_{example}"):
                    st.session_state.example_payload = example

        with col2:
            st.subheader("🔴 Attack Examples")

            st.markdown("**XSS (Cross-Site Scripting):**")
            xss_examples = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>"
            ]

            for example in xss_examples:
                if st.button(f"Try: {example}", key=f"xss_{example}"):
                    st.session_state.example_payload = example

            st.markdown("**SQL Injection:**")
            sqli_examples = [
                "admin' OR '1'='1' --",
                "1; DROP TABLE users; --",
                "' UNION SELECT * FROM passwords --"
            ]

            for example in sqli_examples:
                if st.button(f"Try: {example}", key=f"sqli_{example}"):
                    st.session_state.example_payload = example

        # If an example was selected, show it
        if 'example_payload' in st.session_state:
            st.info(f"Selected example: {st.session_state.example_payload}")
            st.markdown("👆 Go to the **Single Analysis** tab and paste this payload!")

    # Footer
    st.markdown("---")
    st.markdown(
        "🛡️ **Web Attack Detector** - Built with Streamlit | "
        "💡 Use responsibly for security testing only"
    )

if __name__ == "__main__":
    main()
