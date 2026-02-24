import sys
import logging
import time
from rag_chatbot.ingestion.chromadb_collection import update_embeddings


def main():
    args = sys.argv
    log_level = logging.INFO
    if len(args) > 1 and (args[1] == "--verbose" or args[1] == "-v"):
        log_level = logging.DEBUG

    logging.basicConfig(filename="log_file.log",
                    format='%(asctime)s [%(module)s] %(levelname)s: %(message)s',
                    level=log_level
                    )
    logger = logging.getLogger(__name__)

    try:
        start_time = time.time()
        update_embeddings()
        end_time = time.time()
    except Exception:
        logger.warning("An error occured during updating db...")
    finally:
        logger.info(f"Succesdfully updated embeddings in {(end_time - start_time):.3f} seconds")
    return


if __name__ == "__main__":
    main()