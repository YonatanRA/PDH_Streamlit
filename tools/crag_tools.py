
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank
import pickle
import os

from tools.tools import logger


PATH = os.path.dirname(os.path.abspath(__file__))


def ensemble_retriever(collection_name: str) -> EnsembleRetriever:
    
    """
    Retrieving from ChromaDB and BM25 collection.
    
    Params:
    collection_name: str, collection to be called

    Return:
    EnsembleRetriever, ChromaDB+BM25+ReRanker retrieval
    """
    
    embeddings = OpenAIEmbeddings()
    
    # load chromaDB
    retriver_chroma = Chroma(persist_directory=PATH + '/../data/chroma_db',
                             collection_name=collection_name, 
                             embedding_function=embeddings)
    

    logger.info('AQUII 1')

    retriver_chroma = retriver_chroma.as_retriever(search_type='mmr', search_kwargs={'k':20, 
                                                                                     'lambda_mult': 0.5})
    
    logger.info('AQUII 2')

    # load BM25
    with open(PATH + f'/../data/{collection_name}_bm25', 'rb') as bm25_file:
        bm25_retriever = pickle.load(bm25_file)
    
    logger.info('AQUII 3')
    
    try:
        bm25_retriever.k = 10
            
        ensemble_retriever = EnsembleRetriever(retrievers=[retriver_chroma, bm25_retriever],
                                            weights=[0.5, 0.5])

        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

        reranker = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2')

        pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter, reranker])

        compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                            base_retriever=ensemble_retriever)
        
        logger.info('AQUII 4')

    except Exception as e:
        logger.info(e)

    return compression_pipeline

