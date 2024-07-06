from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the generative model
model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_response(input_text, image):
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.candidates[0].content.parts[0].text

# Streamlit app configuration
st.set_page_config(page_title="Image to Text Generator")
st.header("Image to Text Generator")

# Create a text input field
user_input = st.text_input("Input Prompt:", key="input")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Create")

# When the submit button is clicked
if submit:
    response = get_gemini_response(user_input, image)
    st.subheader("Generated Text Response")
    st.write(response)
