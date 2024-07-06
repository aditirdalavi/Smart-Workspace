from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure the generative AI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the generative models
qa_model = genai.GenerativeModel("gemini-pro")
image_to_text_model = genai.GenerativeModel("gemini-pro-vision")

# Streamlit app configuration
st.set_page_config(page_title="AI Application")

# Menu selection for functionalities
option = st.radio(
    'Choose an option:',
    ('Question & Answer', 'Image to Text', 'Chat with PDFs')
)

# Function to handle Question & Answer page
def render_qa_page():
    st.header("Question & Answer")    
    # Create a text input field
    user_input = st.text_input("Enter your question:")
    
    # Create a button to submit the question
    if st.button("Ask a Question"):
        if user_input:
            try:
                response = qa_model.generate_content(user_input)
                st.subheader("Response")
                st.write(response.candidates[0].content.parts[0].text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question before submitting.")

# Function to handle Image to Text page
def render_image_to_text_page():
    st.header("Image to Text Generator")
    
    # Create a text input field for prompt
    user_input = st.text_input("Enter a prompt:")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
        # Create a button to generate text from image
        if st.button("Generate Text"):
            try:
                response = image_to_text_model.generate_content([user_input, image])
                st.subheader("Generated Text Response")
                st.write(response.candidates[0].content.parts[0].text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image before generating text.")

# Function to handle Chat with PDFs page
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def setup_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context". Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = setup_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Response:", response["output_text"])

def render_chat_with_pdfs_page():
    st.header("Interactive PDF Chat")
   
    pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("Processing Complete")
        else:
            st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a question based on the uploaded PDFs:")
    
    if user_question:
        process_user_question(user_question)

# Render pages based on menu selection
if option == 'Question & Answer':
    render_qa_page()
elif option == 'Image to Text':
    render_image_to_text_page()
elif option == 'Chat with PDFs':
    render_chat_with_pdfs_page()
