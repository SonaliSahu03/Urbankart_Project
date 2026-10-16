from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "customer_sentiment_model.pkl"
DATA_PATH = BASE_DIR / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"


@st.cache_resource
def load_sentiment_model(model_path: str):
    """Cache the trained TF-IDF + Logistic Regression pipeline."""
    return joblib.load(model_path)


@st.cache_data
def load_review_data(data_path: str):
    return pd.read_csv(data_path)


st.title("💬 Customer Feedback Analysis")
st.write(
    "Enter a customer review to classify its sentiment. The existing trained "
    "TF-IDF and Logistic Regression pipeline is used; the model is not retrained "
    "when the page opens."
)

if not MODEL_PATH.exists():
    st.error(
        f"Sentiment model not found: {MODEL_PATH.name}. "
        "Keep it in the project root beside app.py."
    )
    st.stop()

try:
    model = load_sentiment_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The sentiment model could not be loaded. Check the scikit-learn version "
        "in requirements.txt and confirm that the model file is valid."
    )
    st.exception(exc)
    st.stop()

review_text = st.text_area(
    "Customer review",
    placeholder="Example: The product quality was excellent and delivery was fast.",
    height=150,
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a customer review first.")
    else:
        try:
            sentiment = str(model.predict([review_text.strip()])[0])
            st.subheader("Predicted sentiment")
            if sentiment.lower() == "positive":
                st.success(sentiment.title())
            elif sentiment.lower() == "negative":
                st.error(sentiment.title())
            else:
                st.info(sentiment.title())

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([review_text.strip()])[0]
                class_names = model.named_steps["classifier"].classes_
                probability_df = pd.DataFrame({
                    "Sentiment": [str(label).title() for label in class_names],
                    "Probability": [float(value) for value in probabilities],
                }).sort_values("Probability", ascending=False)
                st.markdown("#### Class probabilities")
                st.dataframe(
                    probability_df.style.format({"Probability": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.error("Sentiment prediction failed.")
            st.exception(exc)

if DATA_PATH.exists():
    with st.expander("Explore the available review dataset"):
        try:
            df = load_review_data(str(DATA_PATH))
            if {"review_text", "sentiment_label"}.issubset(df.columns):
                left, right = st.columns(2)
                left.metric("Total review records", f"{len(df):,}")
                right.metric("Sentiment classes", df["sentiment_label"].nunique())
                st.markdown("**Sentiment distribution**")
                counts = df["sentiment_label"].astype(str).value_counts().rename_axis(
                    "Sentiment"
                ).reset_index(name="Review count")
                st.bar_chart(counts.set_index("Sentiment"))
                st.dataframe(
                    df[["review_text", "sentiment_label"]].head(10),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "The CSV is present, but it does not contain both review_text "
                    "and sentiment_label columns."
                )
        except Exception as exc:
            st.warning(f"Dataset preview could not be loaded: {exc}")

st.divider()
st.caption("UrbanKart Customer Intelligence Hub | Customer review sentiment analysis")
