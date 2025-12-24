import streamlit as st
import numpy as np
from transformers import pipeline

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI vs Human Text Detector",
    layout="wide"
)

# =========================
# Load Hugging Face model
# =========================
@st.cache_resource
def load_detector():
    """
    Load Hugging Face AI text detector once
    """
    return pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        tokenizer="roberta-base-openai-detector",
        return_all_scores=True
    )

detector = load_detector()

# =========================
# Prediction function
# =========================
def predict_ai_probability(text: str) -> float:
    """
    Return AI probability between 0 and 1
    """
    outputs = detector(text)[0]
    scores = {item["label"]: item["score"] for item in outputs}
    return scores.get("AI", 0.5)

# =========================
# Uncertainty estimation
# =========================
def estimate_confidence(text, ai_prob):
    word_count = len(text.split())
    margin = abs(ai_prob - 0.5)

    indicators = {
        "Short text length (<100 words)": word_count < 100,
        "Prediction near 50%": margin < 0.1,
        "Low prediction margin": margin < 0.15,
    }

    if word_count < 100 or margin < 0.1:
        confidence = "🔴 Low Confidence"
    elif word_count < 200 or margin < 0.2:
        confidence = "🟡 Medium Confidence"
    else:
        confidence = "🟢 High Confidence"

    return confidence, indicators, word_count, margin

# =========================
# UI - Main Page
# =========================
st.title("🧠 AI vs Human Text Detection")
st.caption(
    "This system estimates the likelihood of AI-generated text. "
    "It does not provide definitive authorship judgment."
)

user_text = st.text_area(
    "✏️ Enter text to analyze",
    height=220,
    placeholder="Paste or type your text here..."
)

analyze_btn = st.button("🔍 Analyze Text")

# =========================
# Inference & UI Update
# =========================
if analyze_btn and user_text.strip():

    ai_prob = predict_ai_probability(user_text)
    human_prob = 1 - ai_prob

    confidence, indicators, word_count, margin = estimate_confidence(
        user_text, ai_prob
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("📊 Detection Result")
        st.metric("AI Probability", f"{ai_prob * 100:.2f}%")
        st.metric("Human Probability", f"{human_prob * 100:.2f}%")

        st.divider()

        st.header("🧠 Model Uncertainty")
        st.markdown(f"**Confidence Level:** {confidence}")
        st.markdown(f"- Word count: **{word_count}**")
        st.markdown(f"- Prediction margin: **{margin:.3f}**")

        st.markdown("**Uncertainty Indicators:**")
        for k, v in indicators.items():
            if v:
                st.markdown(f"✅ {k}")
            else:
                st.markdown(f"❌ {k}")

        st.divider()

        st.header("⚠️ System Limitations")
        st.markdown(
            """
            • This system provides probabilistic estimation, not definitive judgment  
            • Short texts may lead to unreliable predictions  
            • Mixed human-AI writing cannot be accurately detected  
            • Results may vary across languages and domains  
            """
        )

    # ---------- Main Panel ----------
    st.subheader("📈 Probability Visualization")
    st.bar_chart({
        "AI": ai_prob,
        "Human": human_prob
    })

    st.info(
        "Interpret results with caution. Prediction confidence depends on "
        "text length and prediction margin."
    )

elif analyze_btn:
    st.warning("Please enter some text before analysis.")
