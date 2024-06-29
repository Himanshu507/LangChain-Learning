from typing import Optional, List, Dict, Any
from chromadb.api.types import Document
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import Callbacks
from langchain_core.retrievers import BaseRetriever
from utils import Utils


class RedundantFilterRetriever(BaseRetriever):
    """
    RedundantFilterRetriever is a retriever that uses a redundant filter to filter out the documents that are not relevant to the query.
    """

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ):
        # Calculate embeddings for the query string and take that embeddings and feed them into the maximum similarity filter.
        embeddings = Utils.get_embeddings("mps")
        emb = embeddings.embed_query(query)
        db = Chroma(embedding_function=embeddings, persist_directory="db")
        return db.max_marginal_relevance_search_by_vector(embedding=emb, lambda_mult=0.8)

    def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return []
