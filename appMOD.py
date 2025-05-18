# APP.PY


# frontend creation
import streamlit as st
# allows pdfs to be read
from PyPDF2 import PdfReader
# allows loading of environment vars from .env
from dotenv import load_dotenv
# allows text to be split to create chunks
from langchain.text_splitter import CharacterTextSplitter
# allows embeddings to be created using OpenAI model
from langchain_openai import OpenAIEmbeddings
# allows FAISS to be used to store vector embeddings
from langchain.vectorstores import FAISS

import os


def main():
    # load api keys
    load_dotenv(override=True)
    # set page configuration
    st.set_page_config(page_title="Feedback Bot", page_icon=":books:")
    # create an html header
    st.header("Feedback")
    # create a text input box
    st.text_input("Ask a question about your documents: ")

    # allow user to upload file
    with st.sidebar:
        st.subheader("Your documents")
        file = st.file_uploader("Upload PDF")
        if file is not None:
            text = processPDF(file)
            chunks = textToChunks(text)
            vectors = chunksToEmbeddings(chunks)
            query = "What is the candidates name?"
            similaritySearch(vectors, query)



# Takes a given pdf file and returns all text contained in it
def processPDF(file):
    reader = PdfReader(file)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text()
    return all_text

# Takes a string of text and converts it into a list of chunks
def textToChunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(text)

# Takes a list of chunks of text and returns a corresponding list of vector embeddings
def chunksToEmbeddings(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db

# Takes embeddings and a query and returns ranked results of most relevant vectors
def similaritySearch(db, query):
    query_vector = embeddings.embed_query(query)



if __name__ == '__main__':
    main()