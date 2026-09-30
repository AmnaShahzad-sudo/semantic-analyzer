import os
import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN is missing. Add it to .streamlit/secrets.toml or set the HF_TOKEN environment variable, then restart Streamlit.")
    st.stop()

# Create Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN
)

# Streamlit page
st.title("Semantic Analyzer")

st.write("Compare two sentences and find their semantic similarity.")

# Input fields
sentence1 = st.text_input("Enter first sentence")
sentence2 = st.text_input("Enter second sentence")

# Analyze button
if st.button("Analyze"):

    if sentence1 and sentence2:

        # Get similarity from Hugging Face
        result = client.sentence_similarity(
            sentence1,
            [sentence2]
        )

        # Get score
        score = result[0]

        # Convert to percentage
        percentage = score * 100

        st.subheader("Result")

        st.metric(
            "Semantic Similarity",
            f"{percentage:.2f}%"
        )

    else:
        st.warning("Please enter both sentences.")