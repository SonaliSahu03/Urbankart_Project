from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# One configuration for the entire multipage application.
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
REVENUE_MODEL_PATH = BASE_DIR / "urbankart_best_model.pkl"
REVIEWS_CSV_PATH = BASE_DIR / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _find_predictor(obj):
    """Return a trained estimator/pipeline from a serialized object or bundle."""
    if callable(getattr(obj, "predict", None)):
        return obj

    if isinstance(obj, dict):
        # Prefer conventional keys so metadata is not mistaken for the model.
        preferred_keys = (
            "model", "best_model", "estimator", "pipeline",
            "best_estimator", "regressor", "trained_model",
        )
        for key in preferred_keys:
            if key in obj:
                candidate = _find_predictor(obj[key])
                if candidate is not None:
                    return candidate
        for value in obj.values():
            candidate = _find_predictor(value)
            if candidate is not None:
                return candidate
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            candidate = _find_predictor(value)
            if candidate is not None:
                return candidate
    return None


@st.cache_resource(show_spinner="Loading the revenue prediction model...")
def load_revenue_model(model_path: str, modified_time: float):
    """Load a model or extract the estimator from a saved bundle/dictionary."""
    loaded_object = joblib.load(model_path)
    model = _find_predictor(loaded_object)
    if model is None:
        if isinstance(loaded_object, dict):
            details = f"Saved dictionary keys: {list(loaded_object.keys())}"
        else:
            details = f"Loaded object type: {type(loaded_object).__name__}"
        raise ValueError(
            "No trained estimator with a predict() method was found. " + details
        )
    return model


@st.cache_data(show_spinner=False)
def load_reviews(csv_path: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner="Loading the Hugging Face sentiment model...")
def load_sentiment_model(model_name: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )


def revenue_prediction_page():
    st.title("💰 Revenue Prediction")
    st.write(
        "Enter a customer's profile and behaviour to estimate annual revenue using "
        "the saved UrbanKart regression model."
    )

    if not REVENUE_MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{REVENUE_MODEL_PATH.name}`. Place it in the "
            "same project-root folder as `app.py`."
        )
        st.caption(f"Expected location: {REVENUE_MODEL_PATH}")
        st.stop()

    try:
        model = load_revenue_model(
            str(REVENUE_MODEL_PATH), REVENUE_MODEL_PATH.stat().st_mtime
        )
    except Exception as exc:
        st.error(
            "Could not load the regression model. Check that the file is a valid "
            "joblib model and that scikit-learn versions are compatible."
        )
        st.exception(exc)
        st.stop()

    with st.expander("Model information and disclaimer"):
        st.write(f"Model file: `{REVENUE_MODEL_PATH.name}`")
        st.write(f"Loaded estimator: `{type(model).__name__}`")
        st.caption(
            "Revenue is a model-generated estimate, not a guarantee of future results. "
            "Use it as decision support rather than as the sole basis for decisions."
        )

    with st.form("revenue_prediction_form"):
        st.subheader("Customer profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            income = st.number_input("Income (₹ per year)", min_value=0.0, value=400000.0, step=10000.0)
            tenure = st.number_input("Customer Tenure (months)", min_value=0, value=24, step=1)
        with c2:
            acquisition_channel = st.selectbox(
                "Acquisition Channel",
                ["Offline Store", "Organic Search", "Affiliate", "Social Media", "Paid Search", "Email", "Referral", "Other"],
            )
            website_visits = st.number_input("Website Visits Monthly", min_value=0.0, value=8.0, step=1.0)
            app_sessions = st.number_input("App Sessions Monthly", min_value=0.0, value=10.0, step=1.0)
            preferred_category = st.selectbox(
                "Preferred Category",
                ["Electronics", "Grocery", "Fashion", "Books", "Accessories", "Sports & Fitness", "Home & Kitchen", "Other"],
            )
            preferred_channel = st.selectbox(
                "Preferred Channel", ["Website", "Physical Store", "Mobile App", "Marketplace"]
            )
        with c3:
            purchase_frequency = st.number_input("Purchase Frequency Monthly", min_value=0.0, value=3.0, step=0.5)
            average_order_value = st.number_input("Average Order Value (₹)", min_value=0.0, value=2500.0, step=100.0)
            orders_last_12m = st.number_input("Orders Last 12 Months", min_value=0, value=20, step=1)
            days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=15, step=1)

        st.subheader("Engagement, loyalty, and service")
        c4, c5, c6 = st.columns(3)
        with c4:
            discount_usage_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.25, 0.01)
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.05, 0.01)
            marketing_response_rate = st.slider("Marketing Response Rate", 0.0, 1.0, 0.20, 0.01)
        with c5:
            customer_satisfaction = st.slider("Customer Satisfaction Score", 1.0, 5.0, 4.0, 0.1)
            support_tickets = st.number_input("Support Tickets in Last 12 Months", min_value=0, value=1, step=1)
            payment_delay_count = st.number_input("Payment Delay Count", min_value=0, value=0, step=1)
        with c6:
            loyalty_points = st.number_input("Loyalty Points", min_value=0, value=250, step=10)
            churn = st.selectbox("Churn", [0, 1], format_func=lambda v: "No (0)" if v == 0 else "Yes (1)")
            purchase_next_30_days = st.selectbox(
                "Purchase Next 30 Days", [0, 1],
                format_func=lambda v: "No (0)" if v == 0 else "Yes (1)",
            )

        submitted = st.form_submit_button("Predict Annual Revenue", type="primary", use_container_width=True)

    if not submitted:
        return

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "City_Tier": city_tier,
        "Income": income,
        "Customer_Tenure_Months": tenure,
        "Acquisition_Channel": acquisition_channel,
        "Website_Visits_Monthly": website_visits,
        "App_Sessions_Monthly": app_sessions,
        "Purchase_Frequency_Monthly": purchase_frequency,
        "Average_Order_Value": average_order_value,
        "Discount_Usage_Rate": discount_usage_rate,
        "Orders_Last_12M": orders_last_12m,
        "Days_Since_Last_Purchase": days_since_last_purchase,
        "Return_Rate": return_rate,
        "Support_Tickets_12M": support_tickets,
        "Customer_Satisfaction_Score": customer_satisfaction,
        "Payment_Delay_Count": payment_delay_count,
        "Loyalty_Points": loyalty_points,
        "Marketing_Response_Rate": marketing_response_rate,
        "Churn": churn,
        "Purchase_Next_30_Days": purchase_next_30_days,
        "Preferred_Category": preferred_category,
        "Preferred_Channel": preferred_channel,
    }])

    # A Pipeline should receive raw columns and perform its own preprocessing.
    # A standalone estimator can only predict if the app supplies its training features.
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        missing = [col for col in expected if col not in input_data.columns]
        if missing:
            st.error(
                "This saved model expects features not supplied by the current form: "
                + ", ".join(missing)
            )
            st.warning(
                "The model and app feature schemas do not match. Use the full saved "
                "training pipeline (including its encoder/preprocessor), or update the "
                "form to match the exact features used during training. Renaming columns "
                "alone cannot replace missing preprocessing."
            )
            return
        input_data = input_data.reindex(columns=expected)
    elif hasattr(model, "n_features_in_") and model.n_features_in_ != input_data.shape[1]:
        st.error(
            f"The saved model expects {model.n_features_in_} input features, but this "
            f"form provides {input_data.shape[1]}. Save and load the complete training "
            "pipeline, including categorical encoding, to ensure compatible predictions."
        )
        return

    try:
        prediction = float(model.predict(input_data)[0])
        st.subheader("Estimated annual revenue")
        st.metric("Predicted Annual Revenue", f"₹{prediction:,.2f}")
        st.caption("This estimate may differ from actual customer revenue.")
        with st.expander("View submitted customer inputs"):
            st.dataframe(input_data, use_container_width=True)
    except Exception as exc:
        st.error(
            "Prediction failed. The model must be trained on the same raw feature names "
            "and compatible preprocessing used by this form."
        )
        st.exception(exc)


def normalize_sentiment(label: str) -> str:
    label_text = str(label).strip().lower()
    if label_text in {"label_0", "0"} or "negative" in label_text:
        return "Negative"
    if label_text in {"label_1", "1"} or "neutral" in label_text:
        return "Neutral"
    if label_text in {"label_2", "2"} or "positive" in label_text:
        return "Positive"
    return str(label).title()


def feedback_analysis_page():
    import matplotlib.pyplot as plt

    st.title("💬 Customer Feedback Analysis")
    st.write(
        "Analyze a customer review or run batch sentiment analysis on the UrbanKart "
        "customer review dataset."
    )

    if not REVIEWS_CSV_PATH.exists():
        st.error(
            f"Dataset not found: `{REVIEWS_CSV_PATH.name}`. Place it in the same "
            "project-root folder as `app.py`."
        )
        st.caption(f"Expected location: {REVIEWS_CSV_PATH}")
        st.stop()

    try:
        reviews_df = load_reviews(str(REVIEWS_CSV_PATH), REVIEWS_CSV_PATH.stat().st_mtime)
    except Exception as exc:
        st.error("Could not read the customer reviews CSV. Confirm that it is a valid CSV file.")
        st.exception(exc)
        st.stop()

    review_candidates = [
        "review_text", "Review_Text", "review", "Review", "customer_review",
        "Customer_Review", "feedback", "Feedback", "text", "Text",
    ]
    review_col = next((col for col in review_candidates if col in reviews_df.columns), None)
    if review_col is None:
        st.error("Could not identify the review-text column in the dataset.")
        st.write("Available columns:", reviews_df.columns.tolist())
        st.stop()

    reviews_df[review_col] = reviews_df[review_col].fillna("").astype(str)
    nonempty_count = int(reviews_df[review_col].str.strip().ne("").sum())
    with st.expander("Dataset overview"):
        a, b, c = st.columns(3)
        a.metric("Total rows", f"{len(reviews_df):,}")
        b.metric("Non-empty reviews", f"{nonempty_count:,}")
        c.metric("Review text column", review_col)
        st.dataframe(reviews_df.head(5), use_container_width=True)

    try:
        sentiment_model = load_sentiment_model(SENTIMENT_MODEL_NAME)
    except Exception as exc:
        st.error(
            "Could not load the Hugging Face model. The first run requires internet "
            "access to download the model, plus compatible `transformers` and `torch` packages."
        )
        st.exception(exc)
        st.stop()

    def analyze_texts(texts, batch_size=16):
        predictions = sentiment_model(
            [str(text).strip() for text in texts],
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )
        return (
            [normalize_sentiment(item["label"]) for item in predictions],
            [float(item["score"]) for item in predictions],
        )

    single_tab, batch_tab = st.tabs(["Analyze one review", "Batch analysis"])
    with single_tab:
        st.subheader("Individual review")
        review_input = st.text_area(
            "Customer review",
            placeholder="Example: I am very happy with my order and delivery was quick.",
            height=140,
            key="single_review_text",
        )
        if st.button("Analyze sentiment", type="primary", key="single_analyze"):
            if not review_input.strip():
                st.warning("Please enter a review before analyzing.")
            else:
                with st.spinner("Analyzing review..."):
                    try:
                        labels, scores = analyze_texts([review_input])
                        left, right = st.columns(2)
                        left.metric("Predicted sentiment", labels[0])
                        right.metric("Confidence score", f"{scores[0]:.1%}")
                        st.progress(min(max(scores[0], 0.0), 1.0))
                        st.caption("Confidence is the model's score for its predicted class, not a guarantee of correctness.")
                    except Exception as exc:
                        st.error("Sentiment analysis failed for this review.")
                        st.exception(exc)

    with batch_tab:
        st.subheader("Batch sentiment analysis")
        st.write(f"Review text column detected: `{review_col}`")
        max_rows = len(reviews_df)
        batch_limit = st.number_input(
            "Rows to analyze (0 means all rows)",
            min_value=0,
            max_value=max_rows,
            value=min(max_rows, 500),
            step=100 if max_rows >= 100 else 1,
        )
        selected_df = reviews_df.copy() if batch_limit == 0 else reviews_df.head(int(batch_limit)).copy()

        if st.button("Run batch sentiment analysis", type="primary", key="batch_analyze"):
            nonempty_mask = selected_df[review_col].str.strip().ne("")
            texts = selected_df.loc[nonempty_mask, review_col].tolist()
            if not texts:
                st.warning("No non-empty review text was found in the selected rows.")
            else:
                progress = st.progress(0.0)
                status = st.empty()
                labels_all, scores_all = [], []
                chunk_size = 16
                try:
                    for start in range(0, len(texts), chunk_size):
                        labels, scores = analyze_texts(texts[start:start + chunk_size], chunk_size)
                        labels_all.extend(labels)
                        scores_all.extend(scores)
                        done = min(start + chunk_size, len(texts))
                        progress.progress(done / len(texts))
                        status.write(f"Analyzed {done:,} of {len(texts):,} reviews")

                    selected_df["predicted_sentiment"] = ""
                    selected_df["sentiment_confidence"] = np.nan
                    selected_df.loc[nonempty_mask, "predicted_sentiment"] = labels_all
                    selected_df.loc[nonempty_mask, "sentiment_confidence"] = scores_all
                    st.session_state["urban_kart_sentiment_results"] = selected_df
                    status.success("Batch sentiment analysis completed.")
                except Exception as exc:
                    st.error("Batch sentiment analysis failed.")
                    st.exception(exc)

        result_df = st.session_state.get("urban_kart_sentiment_results")
        if result_df is not None:
            st.subheader("Sentiment distribution")
            distribution = (
                result_df.loc[result_df["predicted_sentiment"].astype(str).ne(""), "predicted_sentiment"]
                .value_counts()
                .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            )
            chart_col, summary_col = st.columns([2, 1])
            with chart_col:
                fig, ax = plt.subplots()
                distribution.plot(kind="bar", ax=ax)
                ax.set_title("Predicted Customer Sentiment")
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Number of reviews")
                ax.tick_params(axis="x", rotation=0)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with summary_col:
                for label in ["Positive", "Neutral", "Negative"]:
                    st.metric(label, int(distribution[label]))
                st.metric("Reviews analyzed", int(distribution.sum()))

            st.subheader("Analysis results")
            st.dataframe(result_df.head(100), use_container_width=True)
            st.download_button(
                "Download results as CSV",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="UrbanKart_Customer_Sentiment_Results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.caption("Sentiment predictions are estimates and should be reviewed before consequential decisions.")


pages = {
    "Customer Intelligence": [
        st.Page(revenue_prediction_page, title="Revenue Prediction", icon="💰", default=True),
        st.Page(feedback_analysis_page, title="Customer Feedback Analysis", icon="💬"),
    ]
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
