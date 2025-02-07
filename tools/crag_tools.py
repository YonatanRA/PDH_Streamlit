
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
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
    

    logger.info('Chroma retriever init...')

    retriver_chroma = retriver_chroma.as_retriever(search_type='mmr', search_kwargs={'k':20, 
                                                                                     'lambda_mult': 0.5})
    
    logger.info('Chroma retriever defined.')

    # load BM25
    with open(PATH + f'/../data/{collection_name}_bm25', 'rb') as bm25_file:
        bm25_retriever = pickle.load(bm25_file)
    
    logger.info('BM25 retriever loaded.')
    
    try:
        bm25_retriever.k = 10
            
        ensemble_retriever = EnsembleRetriever(retrievers=[retriver_chroma, bm25_retriever],
                                            weights=[0.5, 0.5])

        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

        reranker = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2')

        pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter, reranker])

        compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                            base_retriever=ensemble_retriever)
        
        logger.info('Reranker loaded')

    except Exception as e:
        logger.info(e)

    logger.info('EnsembleRetriever loaded')

    return compression_pipeline

