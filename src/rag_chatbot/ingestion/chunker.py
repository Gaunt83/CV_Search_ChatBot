import os
import tiktoken
import logging
from dotenv import load_dotenv
from operator import concat
from functools import reduce


def get_chunks(cv):
    load_dotenv()

    try:
        chunk_size = int(os.getenv("CHUNK_SIZE"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
    except Exception as ex:
        logging.warning(ex)

    #TODO: implement propper exception.
    if chunk_overlap > chunk_size:
        raise Exception()
    
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode_batch(cv["text"])
    flattened_tokens = list(reduce(concat, tokens))

    chunks = []
    l_limit = 0
    while l_limit < len(flattened_tokens):
        r_limit = l_limit + chunk_size
        chunks.append(flattened_tokens[l_limit: r_limit])
        l_limit = r_limit - chunk_overlap
        
    chunk_num = len(chunks)
    logging.info(f"number of chunks for cv {cv["file_name"]} is: {chunk_num}")

    return chunks

