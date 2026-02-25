from rag_chatbot.ingestion.embedder import get_query_embedding
import time
import logging

logger = logging.getLogger(__name__)


def validate_query(query: str) -> bool:
    if query is None:
        raise ValueError("Query of type None passed as argument")
    
    if len(query) == 0:
        raise ValueError("Qurry must contain text")
    
    if query.isspace():
        raise ValueError("Qurry cannot contain white space only")

    return True


def get_query_vectors(query) -> list:
    try:
        validate_query(query)
    except ValueError as ex:
        logger.info(f"Error while querying: {ex}")
        return []

    start_time = time.time()
    query_embedding = get_query_embedding(query)
    end_time = time.time()

    logger.info(f"Query embeddings generated in {(end_time - start_time):.3f}s")

    return query_embedding