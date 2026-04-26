import argparse
import create_database
import query_database

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"
CHAT_MDL = "llama3:latest"
DEFAULT_NUM_CHUNKS = 15

def argment_parser():
    parser = argparse.ArgumentParser(
        description="""Q&A bot on a book collection""",
    )
    parser.add_argument(
        "--mode",
        default="retrieve",
        choices=['retrieve', 'embedding']
    )
    parser.add_argument(
        "--question",
        help="question you want to ask",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="number of retrieved chunks to send to the LLM",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBEDDING_MDL,
        help="embedding model name",
    )
    parser.add_argument(
        "--chat-model",
        default=CHAT_MDL,
        help="chat model name",
    )

    return parser.parse_args()

if __name__=="__main__":
    args = argment_parser()

    print(args)
    if args.mode == "retrieve" and not args.question:
        raise SystemExit("--question is required when --mode retrieve")

    if args.mode == "embedding":
        create_database.create_db(embed_model=args.embed_model, db_path=DB_PATH)
    elif args.mode == "retrieve":
        embeddings = create_database.build_embeddings(args.embed_model)
        db = create_database.build_vectorstore(embeddings, db_path=DB_PATH)
        chat_model = query_database.build_chat_model(args.chat_model)
        query_content = query_database.similarity_search(
            search_string=args.question,
            num_chunks=args.num_chunks,
            db=db,
        )
        print("------------------------answer------------------------------")
        query_database.invoke_question(args.question, query_content, chat_model)
    else:
        print("wrong mode")



    
