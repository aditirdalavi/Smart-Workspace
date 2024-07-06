from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
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

# Sidebar selection for functionalities
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Question & Answer', 'Image to Text', 'Chat with PDFs')
)

# Function to handle Question & Answer page
def render_qa_page():
    st.header("Question & Answer")
    
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
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def setup_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = setup_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},  # Correct key names
        return_only_outputs=True
    )
    st.write("Response:", response["output_text"])

def render_chat_with_pdfs_page():
    st.header("Interactive PDF Chat")

    user_question = st.text_input("Ask a Question from the Uploaded PDFs")

    if user_question:
        process_user_question(user_question)

    with st.sidebar:
        #st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Your PDF Files and Click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("Processing Complete")

# Render pages based on sidebar selection
if option == 'Question & Answer':
    render_qa_page()
elif option == 'Image to Text':
    render_image_to_text_page()
elif option == 'Chat with PDFs':
    render_chat_with_pdfs_page()
