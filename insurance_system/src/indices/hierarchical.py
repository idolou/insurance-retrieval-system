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

from insurance_system.src.utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZES,
    HIERARCHICAL_STORAGE_DIR,
    SIMILARITY_TOP_K,
)


class HierarchicalIndexError(Exception):
    """Base exception for hierarchical index errors."""

    pass


def create_hierarchical_index(
    documents: List[Document], persist_dir: str = HIERARCHICAL_STORAGE_DIR
) -> VectorStoreIndex:
    """Creates a hierarchical index using the HierarchicalNodeParser and ChromaDB."""
    try:
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
        chroma_path = os.path.join(persist_dir, "chroma")
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            chroma_collection = chroma_client.get_or_create_collection(
                "hierarchical_claims"
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        except Exception as e:
            raise HierarchicalIndexError(f"ChromaDB initialization failed: {e}") from e

        storage_context = StorageContext.from_defaults(
            docstore=docstore, vector_store=vector_store
        )

        # Index the LEAF nodes, but keep reference to parents via docstore
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

        # Persist storage context (docstore mostly, vectors are already in chroma)
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)

        try:
            index.storage_context.persist(persist_dir=persist_dir)
        except Exception as e:
            raise HierarchicalIndexError(f"Index persistence failed: {e}") from e

        return index
    except ValueError:
        raise
    except Exception as e:
        raise HierarchicalIndexError(f"Index creation failed: {e}") from e


def load_hierarchical_retriever(
    persist_dir: str = HIERARCHICAL_STORAGE_DIR,
) -> AutoMergingRetriever:
    """Loads the hierarchical index and returns an AutoMergingRetriever."""
    if not os.path.exists(persist_dir):
        error_msg = f"Index storage directory not found: {persist_dir}"
        raise FileNotFoundError(error_msg)

    try:
        # Initialize Chroma again for loading
        chroma_path = os.path.join(persist_dir, "chroma")
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            chroma_collection = chroma_client.get_or_create_collection(
                "hierarchical_claims"
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        except Exception as e:
            raise HierarchicalIndexError(f"ChromaDB connection failed: {e}") from e

        # Load from storage (re-link vector store)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir, vector_store=vector_store
        )

        try:
            index = load_index_from_storage(storage_context)
        except Exception as e:
            raise HierarchicalIndexError(f"Index loading failed: {e}") from e

        from insurance_system.src.utils.config import SIMILARITY_TOP_K, VERBOSE

        # The AutoMergingRetriever will retrieve leaf nodes and merge them into parent nodes
        # if enough siblings are retrieved.
        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=SIMILARITY_TOP_K),
            storage_context=storage_context,
            verbose=VERBOSE,
        )

        return retriever
    except FileNotFoundError:
        raise
    except Exception as e:
        raise HierarchicalIndexError(f"Retriever loading failed: {e}") from e


def get_hierarchical_query_engine(
    retriever: AutoMergingRetriever, llm: Optional[Any] = None
) -> RetrieverQueryEngine:
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.core.query_engine import RetrieverQueryEngine

        from insurance_system.src.utils.config import (
            RERANKER_MODEL,
            RERANKER_TOP_N,
            USE_RERANKER,
        )

        # Conditionally initialize Reranker
        node_postprocessors = []
        if USE_RERANKER:
            reranker = SentenceTransformerRerank(
                model=RERANKER_MODEL, top_n=RERANKER_TOP_N
            )
            node_postprocessors.append(reranker)

        return RetrieverQueryEngine.from_args(
            retriever, llm=llm, node_postprocessors=node_postprocessors
        )
    except Exception as e:
        raise HierarchicalIndexError(f"Query engine creation failed: {e}") from e
