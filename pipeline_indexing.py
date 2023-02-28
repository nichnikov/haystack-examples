import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(use_bm25=True)
retriever = BM25Retriever(document_store=document_store)

# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
reader = FARMReader(model_name_or_path="distilbert-base-cased-distilled-squad", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

doc_dir = "data/build_your_first_question_answering_system"

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

prediction = pipe.run(
    query="Who is the father of Arya Stark?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)

print(prediction)