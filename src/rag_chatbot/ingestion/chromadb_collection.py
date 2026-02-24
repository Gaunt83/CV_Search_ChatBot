import chromadb
import os
from dotenv import load_dotenv
from rag_chatbot.ingestion import embedder, pptx_loader
from rag_chatbot.exception.custom_exceptions import ArgumentLengthException
import time
import hashlib
import logging


load_dotenv()
collection_location = os.getenv("CHROMA_DIR")
model_name = os.getenv("EMBEDDING_MODEL")
chroma_client = chromadb.PersistentClient(path=collection_location)
collection_name = f"cv_embeddings_{model_name}"
logger = logging.getLogger(__name__)

def initialise_collection():
    collection = chroma_client.get_or_create_collection(collection_name)
    return collection


def update_embeddings():
    cvs = pptx_loader.load_cvs("data/cvs/")
    collection = initialise_collection()

    for cv in cvs:
        cv_embedding = embedder.get_embeddings(cv)
        chunks = cv_embedding["chunk"]
        embeddings = cv_embedding["embedding"]["embeddings"]
        file_name = cv["file_name"]
        candidate_name = cv["candidate_name"]
        if len(chunks) != len(embeddings):
            raise ArgumentLengthException("Chunk and Embedding length does not match...")
        
        ids = []
        metadatas = []
        for i in range(0, len(chunks)):
            id = f"{chunks[i]} {file_name}"

            # ensure id stay consistent if nothing has changed
            # ensures no duplicate entries
            ids.append(hashlib.md5(id.encode()).hexdigest())
            time_now = time.time()
            metadatas.append(
                {
                    "candidate_name": candidate_name,
                    "source_file": file_name,
                    "chunk_index":  i,
                    "ingestion_timestamp": str(time.gmtime(time_now)),
                    "embedding_model": model_name
                }
            )

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )


def reset_database():
    try:
        chroma_client.delete_collection(collection_name)
        _ = initialise_collection()
    except Exception as ex:
        logger.warning(ex)
    return
