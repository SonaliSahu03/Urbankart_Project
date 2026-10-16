# UrbanKart Customer Intelligence Hub

A single Streamlit app with two pages:
- Revenue Prediction
- Customer Feedback Analysis

## Project files
- `app.py`: app entry point and navigation using `st.navigation` / `st.Page`
- `pages/revenue_prediction.py`: existing regression prediction form
- `2_Customer_Feedback_Analysis.py`: interactive customer sentiment page
- `urbankart_best_regression_model.joblib`: regression model (renamed copy of the uploaded `best_regression_model.pkl`)
- `customer_sentiment_model.pkl`: trained TF-IDF + Logistic Regression sentiment pipeline
- `UrbanKart_HuggingFace_NLP_Customer_Reviews.csv`: optional review dataset for the page's dataset preview
- `UrbanKart_Customer_Analytics_Master.csv`: supplied master data, retained with the project

## Setup (Windows / Anaconda Prompt)

1. Open Anaconda Prompt and change to this folder:
   `cd path\to\urbankart_customer_intelligence_hub`
2. Create and activate an environment with Python 3.11:
   `conda create -n urbancustomer python=3.11 -y`
   `conda activate urbancustomer`
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Run the application:
   `python -m streamlit run app.py`

## Notes
- Keep the model files in the project root; paths are resolved relative to the Python files.
- The models are loaded with `st.cache_resource` so they are not reloaded on every rerun.
- `st.set_page_config` is only called once, in `app.py`.
- The regression model was serialized with scikit-learn 1.9.1. Keep the dependency version aligned with the version used to save the model.
- Local syntax checks are separate from a live Streamlit run. Test both pages after installing the dependencies.
