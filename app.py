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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# allows FAISS to be used to store vector embeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os

def main():
    # load api keys
    load_dotenv(override=True)
    # set page configuration
    st.set_page_config(page_title="Feedback Bot", page_icon=":books:")
    # create an html header
    st.header("Feedback")
    # create a text input box
    st.text_input("Ask a question: ")

    # allow user to upload file
    with st.sidebar:
        st.subheader("Your documents")
        file = st.file_uploader("Upload PDF")
        if file is not None:
            query = "What is the candidates name and what processes are they familiar with?"
            query = "What color is a cheetah?"

            # Takes a given pdf file and returns all text contained in it
            reader = PdfReader(file)
            all_text = ""
            for page in reader.pages:
                all_text += page.extract_text()

            # Takes a string of text and converts it into a list of chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(all_text)

            # Takes a list of chunks of text and returns a corresponding list of vector embeddings
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_texts(chunks, embeddings)

            # Takes embeddings and a query and returns ranked results of most relevant vectors
            relevant_vectors = db.similarity_search(query)

            # Set up llm model
            prompt_template = ChatPromptTemplate.from_template("""
                You are a knowledgeable assistant. You can only answer questions using the context provided below.
                If the answer is not in the context, say "I don't know".

                Context:
                {context}

                Question:
                {question}

                Answer:
            """)
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            chain = create_stuff_documents_chain(llm, prompt_template)

            answer = chain.invoke({
                "context":relevant_vectors,
                "question":query
            })

            print("OUTPUT:")
            print(answer)
            print("\n")




if __name__ == '__main__':
    main()