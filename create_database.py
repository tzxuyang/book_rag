from hashlib import sha256
from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
 
def load_documents(path: str):
    loader = DirectoryLoader(path, glob="*.md")
    documents = loader.load()
    return documents

def build_embeddings(model: str = EMBEDDING_MDL):
    return OllamaEmbeddings(model=model)

def load_vectorstore(embeddings=None, db_path: str = DB_PATH):
    chroma_kwargs = {
        "persist_directory": db_path,
    }
    if embeddings is not None:
        chroma_kwargs["embedding_function"] = embeddings

    return Chroma(**chroma_kwargs)

def build_vectorstore(embeddings, db_path: str = DB_PATH):
    return load_vectorstore(embeddings, db_path=db_path)

def build_chunk_id(chunk):
    source = chunk.metadata.get("source", "")
    start_index = chunk.metadata.get("start_index", -1)
    raw_id = f"{source}:{start_index}:{chunk.page_content}"
    return sha256(raw_id.encode("utf-8")).hexdigest()

def split_text(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    return chunks

def create_db(data_path=DATA_PATH, embed_model=EMBEDDING_MDL, db_path=DB_PATH, reset_db=True):
    docs = load_documents(data_path)
    chunks = split_text(docs)
    chunk_ids = [build_chunk_id(chunk) for chunk in chunks]

    db_directory = Path(db_path)
    if reset_db and db_directory.exists():
        shutil.rmtree(db_directory)

    embeddings = build_embeddings(embed_model)
    vectorstore = load_vectorstore(embeddings, db_path=db_path)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    return vectorstore




if __name__=="__main__":
    vectorstore = create_db()

    print("--------retrieve--------------")
    results = vectorstore.similarity_search(
        "阿长,长妈妈",
        k=10,
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
