import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

main_dir = os.listdir('Your path')


""" list of PDFs path """


def get_paths(list_items):
    all_files = []
    for file in list_items:
        path = 'Your path'+file
        all_files.append(path)

    return all_files


""" embedding in base of PDFs content """


def make_embeddings(list_paths):
    full_text = ''
    # extract all content of PDFs
    for pdf in list_paths:
        if pdf.endswith('.pdf'):
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                full_text += page.extract_text()

            # segmentation of the content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )
            chunk = text_splitter.split_text(full_text)
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            knowledge_data_base = FAISS.from_texts(chunk, embeddings)

    return knowledge_data_base


# try the model with any question about PDFs content

if main_dir:
    list_files = get_paths(main_dir)
    knowledge_base = make_embeddings(list_files)
    user_question = input("Enter a question about your PDFs content: ")
    if user_question:
        slices = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type='stuff')
        answer = chain.run(input_documents=slices,
                           question=user_question)
        print(answer)
