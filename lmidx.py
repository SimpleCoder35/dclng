"""Hybrid BM25 + vector retrieval example using LlamaIndex."""

from __future__ import annotations

import os
import logging
import textwrap
from typing import List, Optional, Tuple

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO)


def build_corpus() -> List[Document]:
    """Create a tiny in-memory corpus that covers a few distinct topics."""
    corpus = [
        (
            "Regenerative Agriculture Field Notes",
            "agriculture",
            "Regenerative farms rely on cover crops like crimson clover and hairy vetch to fix nitrogen, "
            "improve soil structure, and provide forage for grazing hens. A weekly log tracks soil organic "
            "matter, rainfall capture, and biodiversity counts for beneficial insects.",
        ),
        (
            "Community Microgrid Pilot Overview",
            "energy",
            "A coastal town deployed a solar-plus-storage microgrid with demand response. Battery-backed "
            "streetlights island automatically during storms, while school rooftops offset peak afternoon "
            "load. Survey data shows a 22 percent drop in diesel generator use after installation.",
        ),
        (
            "Introductory Astronomy Workshop",
            "space",
            "Local science clubs hosted a stargazing workshop covering how to spot Jupiter's moons, why "
            "the Orion nebula glows, and how radio telescopes detect pulsars. Participants logged their "
            "observations and learned safe solar projection techniques for viewing sunspots.",
        ),
        (
            "Urban Wetland Restoration Memo",
            "ecology",
            "City planners mapped an abandoned rail corridor that now hosts cattails, red-winged blackbirds, "
            "and amphibians. Proposed actions include removing invasive phragmites, daylighting a buried "
            "stream, and adding boardwalk access with signage about stormwater filtration.",
        ),
    ]

    return [
        Document(
            text=entry[2],
            metadata={"title": entry[0], "topic": entry[1]},
        )
        for entry in corpus
    ]


def build_hybrid_components() -> Tuple[QueryFusionRetriever, Optional[RetrieverQueryEngine]]:
    """Construct a hybrid retriever and optional query engine."""
    documents = build_corpus()

    # Configure LlamaIndex to use a small sentence-transformers embedding model.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.chunk_size = 512

    vector_index = VectorStoreIndex.from_documents(documents)

    # The storage context tracks the ingested nodes; reuse them for BM25 scoring.
    nodes = list(vector_index.storage_context.docstore.docs.values())

    vector_retriever = vector_index.as_retriever(similarity_top_k=4)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=4,
    )

    # Provide a lightweight mock LLM so no external API key is required.
    fusion_llm = MockLLM(max_tokens=32)
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        llm=fusion_llm,
        similarity_top_k=4,
        num_queries=1,
    )

    # Only build a query engine if an OpenAI API key is present for synthesis.
    api_key_available = bool(os.getenv("OPENAI_API_KEY"))
    query_engine: Optional[RetrieverQueryEngine] = None
    if api_key_available:
        response_synthesizer = get_response_synthesizer(response_mode="compact")
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
        )

    return hybrid_retriever, query_engine


def run_demo() -> None:
    """Execute a few sample questions to showcase hybrid retrieval."""
    hybrid_retriever, query_engine = build_hybrid_components()

    sample_questions = [
        "Which initiative reduced reliance on diesel generators?",
        "How do farmers in the notes improve nitrogen levels in soil?",
        "What did workshop participants learn about observing the sun?",
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"Question: {question}\n")

        retrieved_nodes = hybrid_retriever.retrieve(question)

        if query_engine is not None:
            response = query_engine.query(question)
            print("LLM Answer:\n")
            print(f"{response}\n")
        else:
            if retrieved_nodes:
                best_passage = textwrap.shorten(
                    retrieved_nodes[0].node.get_content(metadata_mode="none"),
                    width=200,
                    placeholder="...",
                )
                print("Heuristic Answer (top fused passage):\n")
                print(f"{best_passage}\n")
            else:
                print("No passages were retrieved.\n")

        print("Top Sources:")
        if not retrieved_nodes:
            print("- none\n")
            continue

        for source in retrieved_nodes:
            title = source.node.metadata.get("title", "Untitled note")
            score = f"{source.score:.3f}" if source.score is not None else "n/a"
            snippet = textwrap.shorten(
                source.node.get_content(metadata_mode="none"),
                width=120,
                placeholder="...",
            )
            print(f"- {title} (score={score}) | {snippet}")
        print()


if __name__ == "__main__":
    run_demo()
