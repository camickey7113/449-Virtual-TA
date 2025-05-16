import streamlit as st
from PyPDF2 import PdfReader

def main():
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
            processPDF(file)

    
def processPDF(file):
    reader = PdfReader(file)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text()
    st.write(all_text)



if __name__ == '__main__':
    main()