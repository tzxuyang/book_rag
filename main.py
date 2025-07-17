import argparse
import create_database
import query_database

DATA_PATH = "data/books"
DB_PATH = "./chroma_langchain_db"
EMBEDDING_MDL = "shaw/dmeta-embedding-zh:latest"

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

    return parser.parse_args()

if __name__=="__main__":
    args = argment_parser()

    print(args)
    # determine test mode
    if args.mode == "embedding":
        create_database.create_db()
    elif args.mode == "retrieve":
        query_content = query_database.similarity_search(search_string=args.question, num_chunks=15)
        # print(query_content)
        print("------------------------answer------------------------------")
        query_database.invoke_question(args.question, query_content, "llama3:latest")
    else:
        print("wrong mode")



    