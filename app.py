import re
import streamlit as st
from qg_pipeline import Pipeline

## Load NLTK
import nltk
nltk.download('punkt')

def preprocess_text(text):
    text = re.sub('\[[0-9]+\]', '', text)
    text = re.sub('[\s]{2,}', ' ', text)
    text = text.strip()
    return text

# Add a model selector to the sidebar
q_model = st.sidebar.selectbox(
    'Select Question Generation Model',
    ('ck46/t5-base-hotpot-qa-qg', 'ck46/t5-small-hotpot-qa-qg')
)

a_model = st.sidebar.selectbox(
    'Select Answer Extraction Model',
    ('ck46/t5-base-hotpot-qa-qg', 'ck46/t5-small-hotpot-qa-qg')
)

st.header('Question-Answer Generation')
st.write(f'Model: {q_model}')

txt = st.text_area('Text for context')

pipeline = Pipeline(
    q_model=q_model,
    q_tokenizer=q_model,
    a_model=a_model,
    a_tokenizer=a_model
)

if len(txt) >= 1:
    autocards = pipeline(preprocess_text(txt))
else:
    autocards = []

st.header('Generated question and answers')
st.write(autocards)
