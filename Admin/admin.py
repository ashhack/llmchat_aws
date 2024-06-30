import os
import sys
import uuid

import boto3
import streamlit as st

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## import FAISS
from langchain_community.vectorstores import FAISS

## s3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_NAME = 'llmchataws'


##
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_uid():
    """
    Generate unique UUID for tracking
    """
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    """
    Split the pages/text into chunksSS
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    """
    Creating vector store 
    """
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path = "./"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## Upload to S3
    s3_client.upload_file(file_name + ".faiss", Bucket=BUCKET_NAME, Key="faiss.faiss")
    s3_client.upload_file(file_name + ".pkl", Bucket=BUCKET_NAME, Key="faiss.pkl")

    return True

def main():
    st.write("This is the admin site for the chat with PDF")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_uid()
        st.write(f"Request Id: {request_id}")
        save_file_name = f"{request_id}.pdf"

        with open(save_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(save_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text From the PDF
        splitted_docs = split_text(pages, 2000, 200)
        st.write(f"Splitted Doc length: {len(splitted_docs)}")
        st.write("================================")
        st.write(splitted_docs[0])
        st.write("================================")

        st.write("Creating Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("PDF Processed Successfully")
        else:
            st.write("Please check logs!!")


if __name__=="__main__":
    main()