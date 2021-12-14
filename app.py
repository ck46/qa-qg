import streamlit as st

# Add a model selector to the sidebar
model = st.sidebar.selectbox(
    'Select Model',
    ('t5-base-squad-qa-qg', 't5-small-squad-qa-qg', 't5-base-hotpot-qa-qg', 't5-small-hotpot-qa-qg')
)

st.header('Question-Answer Generation')
st.write(f'Model in use: {model}')

txt = st.text_area('Text for context')


if len(txt) >= 1:
    autocards = []
else:
    autocards = []

st.header('Generated question and answers')
st.write(autocards)


