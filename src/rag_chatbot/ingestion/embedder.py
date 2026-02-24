import os
import ollama
import logging
from ollama._types import ResponseError
from dotenv import load_dotenv
from rag_chatbot.ingestion import chunker

import time


def get_embeddings(cv, model="text-embedding-3-small"):
    load_dotenv()
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        model = os.getenv("EMBEDDING_MODEL")
    except TypeError as ex:
        logging.warning(ex)

    try:
        chunks = chunker.get_chunks(cv, chunk_size, chunk_overlap)
    except UnboundLocalError as ex:
        logging.warning(ex)

    embeddings = {}
    try:
        start = time.time()
        embedding = ollama.embed(model=model, input=chunks)
        embeddings.update({
            "chunk": chunks,
            "embedding": embedding
        })
        end = time.time()
        elapsed = end - start
        logging.info(f"embedding cv {cv["file_name"]} with model {model} took: {elapsed:.4f}s")
    except ResponseError as ex:
        logging.warning("Encountered issue getting embedding model:", ex)
    except Exception as ex:
        logging.warning(ex)

    return embeddings

