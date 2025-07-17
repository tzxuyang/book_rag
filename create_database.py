from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"
 
def load_documents(path: str):
    loader = DirectoryLoader(path, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    text_list = []
    meta_list = []
    uuid_list = []
    for i in range(len(chunks)):
        text_list.append(chunks[i].page_content)
        meta_list.append(chunks[i].metadata)
        uuid_list.append(str(uuid4()))

    return chunks, text_list, meta_list, uuid_list

def create_db(data_path=DATA_PATH, embed_model=EMBEDDING_MDL, db_path=DB_PATH):
    docs = load_documents(data_path)
    chunks, texts, metas, uuids = split_text(docs)
    embeddings = OllamaEmbeddings(model = embed_model)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    vectorstore.add_documents(documents=chunks, ids=uuids)

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
