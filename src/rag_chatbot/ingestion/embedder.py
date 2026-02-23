import os
import ollama
import logging
from ollama._types import ResponseError
from dotenv import load_dotenv
from rag_chatbot.ingestion import chunker, pptx_loader

import time


def get_embeddings(cvs, model="text-embedding-3-small"):
    load_dotenv()
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        model = os.getenv("EMBEDDING_MODEL")
    except TypeError as ex:
        logging.warning(ex)

    try:
        for cv in cvs:
            cv["chunks"] = chunker.get_chunks(cv, chunk_size, chunk_overlap)
    except UnboundLocalError as ex:
        logging.warning(ex)

    try:
        # ###
        # model = "nomic-embed-text"
        # start = time.time()
        # embeddings = ollama.embed(model=model, input=chunks[0][0])
        # end = time.time()
        # elapsed = end - start
        # print(f"embedding with model {model} took: {elapsed:.4f}s")
        # ###

        ### -> best performance to quality ratio? context len 2k
        # model = "embeddinggemma"
        for cv in cvs:
            start = time.time()
            embeddings = ollama.embed(model=model, input=cv["chunks"])
            end = time.time()
            elapsed = end - start
            logging.info(f"embedding cv {cv["file_name"]} with model {model} took: {elapsed:.4f}s")
            cv["embeddings"] = embeddings
        ###

        # ###
        # model = "qwen3-embedding"
        # start = time.time()
        # embeddings = ollama.embed(model=model, input=chunks[0][0])
        # end = time.time()
        # elapsed = end - start
        # print(f"embedding with model {model} took: {elapsed:.4f}s")
        # ###

        # ###
        # model = "all-minilm"
        # start = time.time()
        # embeddings = ollama.embed(model=model, input=chunks[0][0])
        # end = time.time()
        # elapsed = end - start
        # print(f"embedding with model {model} took: {elapsed:.4f}s")
        # ###
    except ResponseError as ex:
        logging.warning("Encountered issue getting embedding model:", ex)
    except Exception as ex:
        logging.warning(ex)


cvs = pptx_loader.load_cvs("data/cvs/")
get_embeddings(cvs)