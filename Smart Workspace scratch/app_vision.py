from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the generative models
qa_model = genai.GenerativeModel("gemini-pro")
image_to_text_model = genai.GenerativeModel("gemini-pro-vision")

# Streamlit app configuration
st.set_page_config(page_title="AI Application")

# Sidebar selection for functionalities
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Question & Answer', 'Image to Text')
)

# Function to handle Question & Answer page
def render_qa_page():
    st.header("Question & Answer LLM Application")
    
    # Create a text input field
    user_input = st.text_input("Input:", key="qa_input")
    
    # Create a button to submit the question
    submit_qa = st.button("Ask a Question")
    
    # When the submit button is clicked
    if submit_qa:
        if user_input:
            try:
                response = qa_model.generate_content(user_input)
                st.subheader("The response is")
                st.write(response.candidates[0].content.parts[0].text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question before submitting.")

# Function to handle Image to Text page
def render_image_to_text_page():
    st.header("Image to Text Generator")
    
    # Create a text input field for prompt
    user_input = st.text_input("Input Prompt:", key="image_input")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Create a button to generate text from image
    submit_image = st.button("Generate Text")
    
    # When the submit button is clicked
    if submit_image:
        if image is not None:
            try:
                response = image_to_text_model.generate_content([user_input, image])
                st.subheader("Generated Text Response")
                st.write(response.candidates[0].content.parts[0].text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image before generating text.")

# Render pages based on sidebar selection
if option == 'Question & Answer':
    render_qa_page()
elif option == 'Image to Text':
    render_image_to_text_page()
