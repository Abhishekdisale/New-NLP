import streamlit as st
import sklearn.utils.validation
import sklearn.ensemble
import sklearn.compose._column_transformer

# ==============================================================================
# 🔥 CRITICAL CRASH FIXES: MONKEY-PATCHES FOR SKLEARN & IMBLEARN COMPATIBILITY 🔥
# ==============================================================================

# 1. Fixes the "cannot import name '_is_pandas_df'" error
if not hasattr(sklearn.utils.validation, '_is_pandas_df'):
    def _is_pandas_df(X):
        try:
            import pandas as pd
            return isinstance(X, pd.DataFrame)
        except ImportError:
            return False
    sklearn.utils.validation._is_pandas_df = _is_pandas_df

# 2. Fixes the "AdaBoostClassifier.__init__() got an unexpected keyword argument 'algorithm'" error
original_adaboost_init = sklearn.ensemble.AdaBoostClassifier.__init__
def patched_adaboost_init(self, *args, **kwargs):
    if 'algorithm' in kwargs:
        kwargs.pop('algorithm') # Strip out the deprecated parameter safely
    original_adaboost_init(self, *args, **kwargs)
sklearn.ensemble.AdaBoostClassifier.__init__ = patched_adaboost_init

# 3. Fixes the "module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'" error
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ==============================================================================

import joblib
import nltk
import difflib
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download NLTK data required for TextCleaner
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# --- TEXT CLEANER & TRANSFORMER GLOBAL DEFINITIONS ---
stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemma.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [clean_text(text) for text in X]

# Declared here so joblib can unpack the pipeline's downweight function
def scale_down_features(x):
    return x * 0.1


# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    # Ensure these names precisely match your uploaded .pkl filenames
    model = joblib.load("drug_condition_model.pkl") 
    drug_list = joblib.load("drug_list.pkl")
    return model, drug_list

try:
    model, drug_list = load_assets()
    drug_list_lower = [d.lower() for d in drug_list]
except Exception as e:
    st.error(f"⚠️ Error loading model files. Please check your filenames. Details: {e}")
    st.stop()


# --- STREAMLIT UI ---
st.set_page_config(page_title="Drug Review Predictor", page_icon="💊", layout="centered")

st.title("💊 Drug Review Condition Predictor")
st.write("Predict patient conditions (Depression, Diabetes Type 2, or High Blood Pressure) using the drug name and text reviews.")

st.markdown("---")

# ---------- Drug Input Mode ----------
st.subheader("1. Enter Medication Details")
input_method = st.radio(
    "Choose Drug Input Method",
    ["Select from dataset", "Enter manually"]
)

if input_method == "Select from dataset":
    drug = st.selectbox("Select Drug Name", drug_list)
else:
    drug = st.text_input("Enter Drug Name", placeholder="e.g., Metformin, Sertraline...")

# ---------- Review Input ----------
st.subheader("2. Enter Patient Review")
review = st.text_area("Patient Review/Symptoms", placeholder="Describe the symptoms or medication experience here...", height=150)

st.markdown("---")

# ---------- Prediction Logic ----------
if st.button("🔮 Predict Condition", type="primary"):

    if drug.strip() == "":
        st.warning("⚠️ Please provide a drug name.")

    elif review.strip() == "":
        st.warning("⚠️ Please enter the review text.")

    elif len(review.split()) < 3:
        st.warning("⚠️ Review is too short. Please provide at least 3 words for an accurate assessment.")

    else:
        if input_method == "Enter manually":
            if drug.lower() not in drug_list_lower:
                matches = difflib.get_close_matches(drug, drug_list, n=1, cutoff=0.6)
                st.warning("Note: This drug name was not in the training dataset.")
                if matches:
                    st.info(f"Did you mean: **{matches[0]}**?")
        
        with st.spinner("Analyzing text tokens and computing prediction..."):
            
            # Structure input into a 2D DataFrame for ColumnTransformer compatibility
            input_df = pd.DataFrame({
                'review': [review],
                'drugName': [drug]
            })
            
            try:
                prediction = model.predict(input_df)[0]
                
                st.balloons()
                st.success(f"🎯 **Predicted Condition:** {prediction}")
                
                if "Depression" in prediction:
                    st.info("Mental Health Related Assessment")
                elif "Diabetes" in prediction:
                    st.info("Metabolic System Related Assessment")
                elif "Blood Pressure" in prediction:
                    st.info("Cardiovascular System Related Assessment")
                    
            except Exception as prediction_error:
                st.error(f"An unexpected error occurred during pipeline prediction: {prediction_error}")