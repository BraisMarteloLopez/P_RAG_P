"""
Test DT-7 #7: Sin reranker configurado -> reranked_status=None,
rerank_failures=None en config_snapshot.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for mod in [
    "boto3", "botocore", "botocore.exceptions",
    "langchain_nvidia_ai_endpoints", "langchain_core",
    "langchain_core.messages", "langchain_core.documents",
    "langchain_core.embeddings", "langchain_chroma", "chromadb",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from shared.retrieval.core import RetrievalResult, RetrievalStrategy, RetrievalConfig
from shared.config_base import InfraConfig, RerankerConfig


class MockRetriever:
    def retrieve(self, query_text, top_k=None):
        k = top_k or 20
        return RetrievalResult(
            doc_ids=[f"doc_{i}" for i in range(k)],
            contents=[f"content_{i}" for i in range(k)],
            scores=[1.0 - i * 0.01 for i in range(k)],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        return self.retrieve(query_text, top_k)


def test_no_reranker():
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=20),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)
    evaluator._retriever = MockRetriever()
    # _reranker queda None (no configurado)

    detail, reranked_status = evaluator._execute_retrieval(
        "test query", ["doc_0", "doc_1"]
    )

    assert reranked_status is None, (
        f"Esperado None sin reranker, obtenido {reranked_status}"
    )
    assert evaluator._rerank_failures == 0
    assert len(detail.generation_doc_ids) == 0, (
        "Sin reranker, generation_doc_ids debe estar vacio"
    )
    print("PASS: sin reranker -> reranked_status=None, generation_doc_ids vacio")


def test_no_reranker_config_snapshot():
    """Verifica que rerank_failures es None en config_snapshot cuando reranker disabled."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.types import (
        LoadedDataset, NormalizedQuery, NormalizedDocument,
        QueryEvaluationResult, EvaluationStatus, DatasetType, MetricType,
    )

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=20),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)

    # Dataset minimo para _build_run
    dataset = LoadedDataset(name="test", corpus={"doc_0": NormalizedDocument("doc_0", "c")})

    run = evaluator._build_run(
        run_id="test_run",
        dataset=dataset,
        query_results=[],
        elapsed_seconds=1.0,
        indexed_corpus_size=1,
    )

    assert run.config_snapshot["rerank_failures"] is None, (
        f"Esperado None, obtenido {run.config_snapshot['rerank_failures']}"
    )
    print("PASS: config_snapshot rerank_failures=None cuando reranker disabled")


if __name__ == "__main__":
    test_no_reranker()
    test_no_reranker_config_snapshot()
