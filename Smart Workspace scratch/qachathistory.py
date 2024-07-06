from dotenv import load_dotenv
import os
import streamlit as st
from google.generativeai import GenerativeModel, configure

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the generative model
model = GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        # Wait for the response to complete
        response.resolve()
        
        # Extract the text content from the response
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app configuration
st.set_page_config(page_title="Question & Answer")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create a text input field
user_input = st.text_input("Input:")

# Create a button to submit the question
submit = st.button("Ask a Question")

# When the submit button is clicked
if submit and user_input:
    response = get_gemini_response(user_input)
    
    # Store user input and bot response in session state
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Bot", response))
    
    # Display the bot's response
    st.subheader("Response:")
    st.write(response)
    
    # Display the chat history
    st.subheader("Chat History:")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

else:
    st.warning("Please enter a question before submitting.")
