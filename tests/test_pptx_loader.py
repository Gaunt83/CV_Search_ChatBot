from rag_chatbot.ingestion import pptx_loader as loader


def test_load_pptx_loads_files():
    output = loader.load_cvs("tests/data/")[0]
    
    assert output["candidate_name"] == "Jangi RUSSEY"
    assert output["file_name"] == "Jangi-Russey - GenAiDev.pptx"
    assert output["text"].index("Rapid Application Developer.") != -1
