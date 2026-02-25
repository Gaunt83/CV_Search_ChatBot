import os
import ollama
import logging
from ollama._types import ResponseError
from dotenv import load_dotenv
from rag_chatbot.ingestion import chunker

load_dotenv()
logger = logging.getLogger(__name__)
chunk_size = int(os.getenv("CHUNK_SIZE"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
model = os.getenv("EMBEDDING_MODEL")


def get_document_embeddings(text, model="text-embedding-3-small"):
    try:
        chunks = chunker.get_chunks(text, chunk_size, chunk_overlap)
    except UnboundLocalError as ex:
        logger.warning(ex)

    embeddings = {}
    try:
        embedding = ollama.embed(model=model, input=chunks)
        embeddings.update({
            "chunk": chunks,
            "embedding": embedding["embeddings"]
        })
    except ResponseError as ex:
        logger.warning("Encountered issue getting embedding model:", ex)
    except Exception as ex:
        logger.warning(ex)

    return embeddings


def get_query_embedding(text, model="text-embedding-3-small"):
    try:
        chunks = chunker.get_chunks(text, chunk_size, chunk_overlap)
    except UnboundLocalError as ex:
        logger.warning(ex)

    try:
        embedding = ollama.embed(model=model, input=chunks)
    except ResponseError as ex:
        logger.warning("Encountered issue getting embedding model:", ex)
    except Exception as ex:
        logger.warning(ex)

    return embedding["embeddings"]