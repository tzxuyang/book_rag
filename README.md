## Create database
```
python main.py --mode embedding
```

Embedding mode now rebuilds the local Chroma database before inserting fresh chunks, so repeated runs do not accumulate duplicates.

Retrieve mode loads the existing local Chroma database directly from `./chroma_langchain_db`.

## Q & A
```
python main.py --mode retrieve --question "长妈妈是怎样的人"
```

You can also tune retrieval and model selection:
```
python main.py --mode retrieve --question "长妈妈是怎样的人" --num-chunks 8 --chat-model llama3:latest
```
