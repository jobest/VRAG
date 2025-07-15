import os
import tempfile
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source_type": "pdf",
            "file_name": file.name,
            "timestamp": datetime.now().isoformat()
        })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    os.remove(path)
    return chunks