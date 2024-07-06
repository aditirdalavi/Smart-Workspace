from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import google.generativeai as genai
import os

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the generative model
model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(question):
    try:
        response = model.generate_content(question)
        # Extract the text content from the response
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app configuration
st.set_page_config(page_title="Question & Answer")

st.header("Question & Answer LLM Application")

# Create a text input field
user_input = st.text_input("Input:", key="input")

# Create a button to submit the question
submit = st.button("Ask a Question")

# When the submit button is clicked
if submit:
    if user_input:
        response = get_gemini_response(user_input)
        st.subheader("The response is")
        st.write(response)
    else:
        st.warning("Please enter a question before submitting.")
