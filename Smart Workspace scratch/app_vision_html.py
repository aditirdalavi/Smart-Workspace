from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set Streamlit page configuration
st.set_page_config(page_title="AI Application")

# Custom CSS for styling
custom_css = """
<style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        background-color: #f0f0f0;
        margin: 0;
        padding: 20px;
    }
    .container {
        max-width: 800px;
        margin: auto;
        background: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .option {
        margin-bottom: 10px;
    }
    .input-field {
        margin-bottom: 20px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin-top: 10px;
        cursor: pointer;
        border: none;
        border-radius: 5px;
    }
    .button:hover {
        background-color: #45a049;
    }
    .warning {
        color: red;
    }
    .title {
        text-align: center;
    }
</style>
"""

# Render custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Main Streamlit app logic
def main():
    st.title("Welcome to AI Application")

    # Option selection
    option = st.radio(
        "Select an option:",
        ('Question & Answer', 'Image to Text')
    )

    # Handle Question & Answer functionality
    if option == 'Question & Answer':
        render_qa_page()

    # Handle Image to Text functionality
    elif option == 'Image to Text':
        render_image_to_text_page()

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
                # Assuming qa_model is initialized properly
                response = genai.GenerativeModel("gemini-pro").generate_content(user_input)
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
                # Assuming image_to_text_model is initialized properly
                response = genai.GenerativeModel("gemini-pro-vision").generate_content([user_input, image])
                st.subheader("Generated Text Response")
                st.write(response.candidates[0].content.parts[0].text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image before generating text.")

# Run the main application
if __name__ == "__main__":
    main()
