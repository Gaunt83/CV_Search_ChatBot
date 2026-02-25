import tiktoken
import logging
from dotenv import load_dotenv
from operator import concat
from functools import reduce
from rag_chatbot.exception.custom_exceptions import ArgumentLengthException

logger = logging.getLogger(__name__)

def get_chunks(text, chunk_size=500, chunk_overlap=100):
    load_dotenv()

    if chunk_overlap > chunk_size:
        raise ArgumentLengthException("Chunk size should be greater than chunk overlap!")
    
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode_batch(text)
    flattened_tokens = list(reduce(concat, tokens))

    chunks = []
    l_limit = 0
    while l_limit < len(flattened_tokens):
        r_limit = l_limit + chunk_size
        chunks.append(flattened_tokens[l_limit: r_limit])
        l_limit = r_limit - chunk_overlap
        
    chunk_num = len(chunks)
    logger.info(f"number of chunks created is: {chunk_num}")

    chunked_string = encoding.decode_batch(chunks)

    return chunked_string

