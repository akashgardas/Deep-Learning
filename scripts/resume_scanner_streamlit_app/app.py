import os
import re
import pickle
import numpy as np
import tensorflow as tf
from pypdf import PdfReader
import streamlit as st
from pathlib import Path

# ==============================================================================
# 🎨 1. STREAMLIT PAGE CONFIG & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep tech/modern UI styling via CSS
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #e2e8f0; }
    h1 { color: #38bdf8 !important; font-weight: 700 !important; }
    h2, h3 { color: #f1f5f9 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
        color: white; border: none; padding: 0.6rem 1.5rem;
        border-radius: 8px; font-weight: 600; width: 100%;
        transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20deg rgba(56, 189, 248, 0.4);
        color: white;
    }
    .metric-card {
        background-color: #1e293b; border: 1px solid #334155;
        padding: 1.5rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .cleaned-box {
        background-color: #0b0f17; border-left: 4px solid #38bdf8;
        padding: 1rem; font-family: monospace; font-size: 0.85rem;
        max-height: 250px; overflow-y: auto; border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 2. CUSTOM LAYER & MODEL LOADING FUNCTIONS
# ==============================================================================
# Keras needs this exact class layout available globally to map the saved model structure
@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, name="token_embed"
        )
        
        positions = tf.range(start=0, limit=sequence_length, delta=1, dtype=tf.float32)[:, tf.newaxis]
        indices = tf.range(start=0, limit=embed_dim, delta=1, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (indices // 2.0)) / tf.cast(embed_dim, tf.float32))
        angle_rads = positions * angle_rates
        
        sin_mask = tf.cast(tf.equal(tf.math.mod(tf.range(embed_dim), 2), 0), tf.float32)
        cos_mask = tf.cast(tf.equal(tf.math.mod(tf.range(embed_dim), 2), 1), tf.float32)
        
        pos_matrix = (tf.sin(angle_rads) * sin_mask) + (tf.cos(angle_rads) * cos_mask)
        self.pos_encoding = pos_matrix[tf.newaxis, ...] 

    def call(self, inputs):
        tokens = self.token_embeddings(inputs)
        return tokens + tf.cast(self.pos_encoding, dtype=tokens.dtype)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


@st.cache_resource
def load_deployment_artifacts():
    """Loads and caches all trained modeling layers to preserve high performance."""

    BASE_DIR = Path(__file__).resolve().parent
    ARTIFACTS_DIR = BASE_DIR / "models"

    config_path = ARTIFACTS_DIR / "config.pkl"
    tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"
    label_encoder_path = ARTIFACTS_DIR / "label_encoder.pkl"
    model_path = ARTIFACTS_DIR / "transformer_model.keras"
    
    # Load configuration parameters
    with open(os.path.join(config_path), "rb") as f:
        config = pickle.load(f)
        
    # Load tokenizers and target text encoders
    with open(os.path.join(tokenizer_path), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(label_encoder_path), "rb") as f:
        label_encoder = pickle.load(f)
        
    # Load Keras Deep Learning Transformer Architecture
    model_path = os.path.join(model_path)
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )
    
    return model, tokenizer, label_encoder, config

# ==============================================================================
# 🧹 3. PIPELINE PREPROCESSING HELPER FUNCTIONS
# ==============================================================================
def clean_resume_text(text):
    """Exact pipeline cleaning replica preserving technology markers (+, #, .)"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[－•\t\n\r]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    """Safely extracts linear stream context characters from byte objects."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    except Exception as e:
        st.error(f"Error parsing PDF document layout structure: {e}")
        return None

# ==============================================================================
# 🚀 4. MAIN USER INTERFACE APPLICATION CONTEXT
# ==============================================================================
def main():
    # Sidebar Metadata Panel
    st.sidebar.title("🛠️ Model Architecture")
    st.sidebar.markdown("""
    This inference engine uses a custom **Transformer Encoder Block** paired with **Sinusoidal Positional Embeddings** to preserve spatial dependencies across text structures.
    """)
    
    try:
        model, tokenizer, label_encoder, config = load_deployment_artifacts()
        st.sidebar.success("✅ Deployment artifacts successfully initialized.")
        
        # Display model stats inside the sidebar layout
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Max Sequence Length:** `{config['MAX_LEN']}` tokens")
        st.sidebar.markdown(f"**Vocab Limit:** `{config['MAX_WORDS']}` keywords")
        st.sidebar.markdown(f"**Attention Heads:** `{config['num_heads']}` blocks")
    except Exception as e:
        st.sidebar.error(f"🔴 Pipeline Error loading artifacts from `/models`: {e}")
        st.error("Please verify that your training artifacts exist inside the `models/` folder relative to this directory.")
        return

    # Application Header Title Block
    st.title("📄 AI-Driven Spatial Resume Classifier")
    st.markdown("Automated text classification across 24 industry categories using a custom multi-head self-attention network.")
    st.markdown("---")

    # Main Grid split window interface layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Input Data Stream")
        input_type = st.radio("Select Input Method:", ["Upload PDF Document", "Paste Raw Text Profile"], horizontal=True)
        
        raw_text = ""
        if input_type == "Upload PDF Document":
            uploaded_file = st.file_uploader("Drop candidate resume PDF below:", type=["pdf"])
            if uploaded_file is not None:
                with st.spinner("Parsing document structure layout..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                if raw_text:
                    st.success(f"Successfully processed: '{uploaded_file.name}'")
        else:
            raw_text = st.text_area("Paste text content here:", height=300, placeholder="John Doe\nSoftware Engineer...")

    with col2:
        st.subheader("📊 Analytical Predictions")
        
        if not raw_text.strip():
            st.info("Provide a data source profile inside Input Stream to execute spatial taxonomy tracking.")
            return

        # Inference Processing Triggers
        if st.button("🚀 Run Classification Network"):
            with st.spinner("Applying cleaning matrices and running transformer inference..."):
                # 1. Clean raw context
                cleaned_text = clean_resume_text(raw_text)
                
                # 2. Tokenize using training constraints 
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                
                # 3. Handle zero boundaries padding metrics
                padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                    sequences,
                    maxlen=config["MAX_LEN"],
                    padding="post"
                )
                
                # 4. Predict probabilities using cached model layers
                predictions = model.predict(padded_sequence, verbose=0)[0]
                top_idx = np.argmax(predictions)
                
                # 5. Reverse target encoding array mapping indices to labels
                predicted_category = label_encoder.inverse_transform([top_idx])[0]
                confidence_score = predictions[top_idx] * 100

                # Render custom CSS UI metric banner blocks
                st.markdown(f"""
                <div class="metric-card">
                    <p style="margin:0; font-size:1.1rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;">Predicted Domain Classification</p>
                    <h2 style="margin:0.5rem 0; color:#38bdf8; font-size:2.4rem; font-weight:800;">{predicted_category}</h2>
                    <p style="margin:0; font-size:1.2rem; color:#4ade80; font-weight:600;">{confidence_score:.2f}% System Confidence</p>
                </div>
                """, unsafe_allow_html=True)

                # Expandable view breaking down other potential categories
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📈 Complete Distribution Breakdown"):
                    # Zip classifications cleanly to construct sorted visual distributions
                    all_labels = label_encoder.classes_
                    chart_data = sorted(zip(all_labels, predictions), key=lambda x: x[1], reverse=True)
                    
                    for label, prob in chart_data[:5]:  # Display top 5 matches
                        st.write(f"**{label}**")
                        st.progress(float(prob))

                with st.expander("🔍 Cleaned Corpus Sample Viewed by Model"):
                    st.markdown(f'<div class="cleaned-box">{cleaned_text[:1200]}...</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()