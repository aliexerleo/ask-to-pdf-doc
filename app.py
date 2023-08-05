from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

pdf_path = "./CV-Aliexer-Mayor.pdf"


def make_embeddings(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    # chunks proccess
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunk = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    knowledge_base = FAISS.from_texts(chunk, embeddings)

    return knowledge_base


if pdf_path:
    knowledge_base = make_embeddings(pdf_path)
    user_question = input("Enter a query: ")
    if user_question:
        slices = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type='stuff')
        answer = chain.run(input_documents=slices,
                           question=user_question)
        print(answer)
