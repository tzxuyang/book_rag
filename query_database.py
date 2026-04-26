import create_database

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"

def build_chat_model(chat_model, temperature=0):
    return OllamaLLM(model=chat_model, temperature=temperature)

def similarity_search(search_string, num_chunks, db):
    results = db.similarity_search(search_string, k=num_chunks)
    return "\n\n".join(res.page_content for res in results)

def invoke_question(search_string, context_str, chat_model):
    prompt = ChatPromptTemplate(
        [
            ("system", "请回答关于作品中的问题,书中有这些介绍：{context}"),
            ("human", "{question}")
        ]
    )

    chain = prompt | chat_model | StrOutputParser()

    prompt_value = chain.invoke(
        {
            "context": context_str,
            "question": search_string
        }
    )
    print(prompt_value)

if __name__=="__main__":
    embeddings = create_database.build_embeddings(EMBEDDING_MDL)
    db = create_database.build_vectorstore(embeddings, db_path=DB_PATH)
    query_content = similarity_search(search_string="长妈妈是怎样的人", num_chunks=15, db=db)
    chat_model = build_chat_model("llama3:latest")
    invoke_question("长妈妈是怎样的人", query_content, chat_model)

    
