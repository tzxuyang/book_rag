from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"

def similarity_search(search_string, num_chunks, embed_model=EMBEDDING_MDL, db_path = DB_PATH):
    embeddings = OllamaEmbeddings(model = embed_model)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    db.get() 
    results = db.similarity_search(search_string, k=num_chunks)

    query_content = ""

    for res in results:
        # print(f"* {res.page_content} [{res.metadata}]")
        query_content = query_content + res.page_content

    return query_content

def invoke_question(search_string, context_str, chat_model, temperature=0):
    chain_chat_model = OllamaLLM(model=chat_model, temperature = temperature)
    prompt = ChatPromptTemplate(
        [
            ("system", "请回答关于作品中的问题,书中有这些介绍：{context}"),
            ("human", "{question}")
        ]
    )

    chain = prompt | chain_chat_model | StrOutputParser()

    prompt_value = chain.invoke(
        {
            "context": context_str,
            "question": search_string
        }
    )
    print(prompt_value)

if __name__=="__main__":
    query_content = similarity_search(search_string="长妈妈是怎样的人", num_chunks=15)
    invoke_question("长妈妈是怎样的人", query_content, "llama3:latest")

    