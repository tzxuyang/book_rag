import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MAIN_PATH = REPO_ROOT / "main.py"


class MainCliTests(unittest.TestCase):
    def run_main_with_stubs(self, argv):
        calls = {}

        fake_create_database = types.ModuleType("create_database")

        def build_embeddings(*args, **kwargs):
            calls["build_embeddings"] = {"args": args, "kwargs": kwargs}
            raise AssertionError("retrieve mode should not build embeddings")

        def load_vectorstore(*args, **kwargs):
            calls["load_vectorstore"] = {"args": args, "kwargs": kwargs}
            return "fake-db"

        def create_db(*args, **kwargs):
            calls["create_db"] = {"args": args, "kwargs": kwargs}
            return "fake-created-db"

        fake_create_database.build_embeddings = build_embeddings
        fake_create_database.load_vectorstore = load_vectorstore
        fake_create_database.create_db = create_db

        fake_query_database = types.ModuleType("query_database")

        def build_chat_model(chat_model):
            calls["build_chat_model"] = chat_model
            return "fake-chat-model"

        def similarity_search(search_string, num_chunks, db):
            calls["similarity_search"] = {
                "search_string": search_string,
                "num_chunks": num_chunks,
                "db": db,
            }
            return "fake-context"

        def invoke_question(search_string, context_str, chat_model):
            calls["invoke_question"] = {
                "search_string": search_string,
                "context_str": context_str,
                "chat_model": chat_model,
            }

        fake_query_database.build_chat_model = build_chat_model
        fake_query_database.similarity_search = similarity_search
        fake_query_database.invoke_question = invoke_question

        original_argv = sys.argv[:]
        original_modules = {
            "create_database": sys.modules.get("create_database"),
            "query_database": sys.modules.get("query_database"),
        }

        try:
            sys.modules["create_database"] = fake_create_database
            sys.modules["query_database"] = fake_query_database
            sys.argv = [str(MAIN_PATH), *argv]

            with redirect_stdout(io.StringIO()):
                runpy.run_path(str(MAIN_PATH), run_name="__main__")
        finally:
            sys.argv = original_argv
            for module_name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = module

        return calls

    def test_retrieve_mode_loads_existing_vectorstore_without_building_embeddings(self):
        calls = self.run_main_with_stubs(
            ["--mode", "retrieve", "--question", "长妈妈是怎样的人"]
        )

        self.assertNotIn("build_embeddings", calls)
        self.assertEqual(
            calls["load_vectorstore"],
            {"args": (), "kwargs": {"db_path": "./chroma_langchain_db"}},
        )
        self.assertEqual(calls["build_chat_model"], "llama3:latest")
        self.assertEqual(
            calls["similarity_search"],
            {
                "search_string": "长妈妈是怎样的人",
                "num_chunks": 8,
                "db": "fake-db",
            },
        )
        self.assertEqual(
            calls["invoke_question"],
            {
                "search_string": "长妈妈是怎样的人",
                "context_str": "fake-context",
                "chat_model": "fake-chat-model",
            },
        )


if __name__ == "__main__":
    unittest.main()
