import os
from typing import Any, List, Optional

import chromadb
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from insurance_system.src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZES,
    HIERARCHICAL_STORAGE_DIR,
    SIMILARITY_TOP_K,
)


def create_hierarchical_index(
    documents: List[Document], persist_dir: str = HIERARCHICAL_STORAGE_DIR
) -> VectorStoreIndex:
    """Creates a hierarchical index using the HierarchicalNodeParser and ChromaDB."""
    # Define the chunk sizes for the hierarchy
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=CHUNK_SIZES, chunk_overlap=CHUNK_OVERLAP
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # Create storage context
    # 1. Docstore (for hierarchy mapping)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # 2. Vector Store (ChromaDB)
    # Ensure persistent client
    chroma_client = chromadb.PersistentClient(path=os.path.join(persist_dir, "chroma"))
    chroma_collection = chroma_client.get_or_create_collection("hierarchical_claims")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )

    # Index the LEAF nodes, but keep reference to parents via docstore
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    # Persist storage context (docstore mostly, vectors are already in chroma)
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    index.storage_context.persist(persist_dir=persist_dir)

    return index


def load_hierarchical_retriever(
    persist_dir: str = HIERARCHICAL_STORAGE_DIR,
) -> AutoMergingRetriever:
    """Loads the hierarchical index and returns an AutoMergingRetriever."""
    # Initialize Chroma again for loading
    chroma_client = chromadb.PersistentClient(path=os.path.join(persist_dir, "chroma"))
    chroma_collection = chroma_client.get_or_create_collection("hierarchical_claims")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load from storage (re-link vector store)
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=vector_store
    )

    index = load_index_from_storage(storage_context)

    # The AutoMergingRetriever will retrieve leaf nodes and merge them into parent nodes
    # if enough siblings are retrieved.
    retriever = AutoMergingRetriever(
        index.as_retriever(similarity_top_k=SIMILARITY_TOP_K),  # type: ignore
        storage_context=storage_context,
        verbose=True,
    )

    return retriever


def get_hierarchical_query_engine(
    retriever: AutoMergingRetriever, llm: Optional[Any] = None
) -> RetrieverQueryEngine:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.query_engine import RetrieverQueryEngine

    from insurance_system.src.config import RERANKER_MODEL, RERANKER_TOP_N

    # Initialize Reranker
    reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=RERANKER_TOP_N)

    return RetrieverQueryEngine.from_args(
        retriever, llm=llm, node_postprocessors=[reranker]
    )
