from rag_chatbot.ingestion import pptx_loader as loader
from pptx.exc import PackageNotFoundError
import pytest


def test_load_cvs_loads_files():
    output = loader.load_cvs("tests/data/")[0]
    
    assert output["candidate_name"] == "Jangi RUSSEY"
    assert output["file_name"] == "Jangi-Russey - GenAiDev.pptx"
    assert output["text"].index("Rapid Application Developer.") != -1


def test_load_cvs_does_not_throw_error_on_bad_dir():
    try:
        loader.load_cvs("bad_dir")
    except FileNotFoundError:
        pytest.fail("Unexpected 'FileNotFoundError' error thrown")


def test_load_cvs_does_not_throw_error_broken_file():
    try:
        loader.load_cvs("tests/data/faulty")
    except PackageNotFoundError:
        pytest.fail("Unexpected 'PackageNotFoundError' error thrown")

