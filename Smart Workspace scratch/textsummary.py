from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.0-pro-latest")

def text_summarizer():
    st.header("Text Summarizer")
    
    # Create a text input field for the text to summarize
    text_to_summarize = st.text_area("Input Text", key="qa_input")
    
    # Create a numeric input field for the number of words
    num_words = st.number_input("Number of Words", min_value=10, max_value=500, step=10, value=100)
    
    # Create a dropdown for tone selection
    tone = st.selectbox("Select Tone", options=["formal", "informal", "fluent"])
    
    # Create a text input field for additional user prompts
    additional_prompt = st.text_input("Additional Prompt", "Enter any additional instructions or specific needs...")
    
    # Create a button to submit the request
    submit_qa = st.button("Summarize Text")

    # When the submit button is clicked
    if submit_qa:
        if text_to_summarize:
            try:
                # Craft a prompt instructing the model on summarizing text
                prompt = f"Summarize this article in {num_words} words in a {tone} tone: {text_to_summarize}"
                
                # Add additional prompt if provided
                if additional_prompt:
                    prompt += f"\nAdditional Prompt: {additional_prompt}"

                # Generate the text using the prompt and model
                response = model.generate_content(prompt)
                summary = response.text  # Correct attribute to access the generated content
                st.subheader("Summary")
                st.write(summary)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter the text to summarize before submitting.")

# Run the text summarizer function in the Streamlit app
text_summarizer()
