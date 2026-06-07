import streamlit as st
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

# --- TEXT CLEANER DEFINITION ---
# Kept here locally to ensure scikit-learn/joblib can unpickle the pipeline seamlessly
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
    
def scale_down_features(x):
    return x * 0.1


# --- LOAD MODEL & ASSETS ---
# NOTE: Make sure these filenames match exactly what you uploaded to your repository!
@st.cache_resource
def load_assets():
    # Change "drug_condition_model.pkl" if your saved file has a different name
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
        # If manual drug input, perform a similarity check to help the user
        if input_method == "Enter manually":
            if drug.lower() not in drug_list_lower:
                matches = difflib.get_close_matches(drug, drug_list, n=1, cutoff=0.6)
                st.warning("Note: This drug name was not in the training dataset.")
                if matches:
                    st.info(f"Did you mean: **{matches[0]}**?")
        
        # Create a Status Spinner while processing
        with st.spinner("Analyzing text tokens and computing prediction..."):
            
            # 1. Structure input into a 2D DataFrame with matching column names for ColumnTransformer
            input_df = pd.DataFrame({
                'review': [review],
                'drugName': [drug]
            })
            
            # 2. Run the pipeline prediction safely
            try:
                prediction = model.predict(input_df)[0]
                
                # 3. Display Result
                st.balloons()
                st.success(f"🎯 **Predicted Condition:** {prediction}")
                
                # Contextual alert styling based on result
                if "Depression" in prediction:
                    st.info("Mental Health Related Assessment")
                elif "Diabetes" in prediction:
                    st.info("Metabolic System Related Assessment")
                elif "Blood Pressure" in prediction:
                    st.info("Cardiovascular System Related Assessment")
                    
            except Exception as prediction_error:
                st.error(f"An unexpected error occurred during pipeline prediction: {prediction_error}")
                st.info("Tip: Ensure 'imbalanced-learn' is added to requirements.txt and your saved model pipeline matches this input format.")