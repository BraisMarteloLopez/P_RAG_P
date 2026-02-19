"""
Tests para DT-5: Trazabilidad de candidatos pre-rerank.

Verifica que:
  - Con reranker: pre_rerank_candidate_ids contiene los IDs del pool completo (PRE_FUSION_K)
  - generation_doc_ids es subconjunto de pre_rerank_candidate_ids
  - Sin reranker: pre_rerank_candidate_ids queda vacio
  - to_dict() incluye el campo solo cuando no vacio
  - to_dict() no incluye el campo cuando no hay reranker

Sin dependencias externas (mocks de retriever y reranker).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for mod in [
    "boto3", "botocore", "botocore.exceptions",
    "langchain_nvidia_ai_endpoints", "langchain_core",
    "langchain_core.messages", "langchain_core.documents",
    "langchain_core.embeddings", "langchain_chroma", "chromadb",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from shared.types import QueryRetrievalDetail, QueryEvaluationResult, DatasetType, MetricType, EvaluationStatus
from shared.retrieval.core import RetrievalResult


# =================================================================
# Helpers
# =================================================================

def _make_evaluator(reranker_enabled=True):
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.retrieval.core import RetrievalConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            retrieval_k=5,
            pre_fusion_k=20,
        ),
        reranker=RerankerConfig(
            enabled=reranker_enabled,
            top_n=3,
        ),
        generation_enabled=True,
    )
    evaluator = MTEBEvaluator(config)
    return evaluator


def _mock_retriever(doc_ids: List[str]):
    """Mock retriever que retorna docs con IDs dados."""
    mock = MagicMock()
    result = RetrievalResult(
        doc_ids=doc_ids,
        contents=[f"content_{did}" for did in doc_ids],
        scores=[1.0 - i * 0.01 for i in range(len(doc_ids))],
        retrieval_time_ms=10.0,
    )
    mock.retrieve.return_value = result
    mock.retrieve_by_vector.return_value = result
    return mock


def _mock_reranker(top_ids: List[str], reranked_ok=True):
    """Mock reranker que retorna top_ids como resultado post-rerank."""
    mock = MagicMock()
    reranked_result = RetrievalResult(
        doc_ids=top_ids,
        contents=[f"content_{did}" for did in top_ids],
        scores=[0.99 - i * 0.1 for i in range(len(top_ids))],
        retrieval_time_ms=15.0,
        metadata={"reranked": reranked_ok},
    )
    mock.rerank.return_value = reranked_result
    return mock


# =================================================================
# Tests
# =================================================================

def test_pre_rerank_ids_populated_with_reranker():
    """Con reranker activo, pre_rerank_candidate_ids contiene los PRE_FUSION_K IDs."""
    evaluator = _make_evaluator(reranker_enabled=True)

    # Simular 20 candidatos (PRE_FUSION_K=20)
    all_candidate_ids = [f"doc_{i:03d}" for i in range(20)]
    reranked_top = ["doc_015", "doc_003", "doc_007"]  # reranker promueve docs

    evaluator._retriever = _mock_retriever(all_candidate_ids)
    evaluator._reranker = _mock_reranker(reranked_top)

    detail, reranked_ok = evaluator._execute_retrieval(
        query_text="test query",
        expected_doc_ids=["doc_003"],
    )

    # pre_rerank_candidate_ids debe contener los 20 candidatos
    assert detail.pre_rerank_candidate_ids == all_candidate_ids, (
        f"Esperado {len(all_candidate_ids)} IDs, "
        f"obtenido {len(detail.pre_rerank_candidate_ids)}"
    )

    # generation_doc_ids debe ser el resultado del reranker
    assert detail.generation_doc_ids == reranked_top

    # generation_doc_ids debe ser subconjunto de pre_rerank_candidate_ids
    pre_set = set(detail.pre_rerank_candidate_ids)
    for gid in detail.generation_doc_ids:
        assert gid in pre_set, (
            f"generation_doc_id '{gid}' no esta en pre_rerank_candidate_ids"
        )

    print("PASS: pre_rerank_candidate_ids poblado con PRE_FUSION_K IDs")


def test_pre_rerank_ids_empty_without_reranker():
    """Sin reranker, pre_rerank_candidate_ids queda vacio."""
    evaluator = _make_evaluator(reranker_enabled=False)

    candidate_ids = [f"doc_{i:03d}" for i in range(5)]
    evaluator._retriever = _mock_retriever(candidate_ids)
    evaluator._reranker = None

    detail, reranked_ok = evaluator._execute_retrieval(
        query_text="test query",
        expected_doc_ids=["doc_001"],
    )

    assert detail.pre_rerank_candidate_ids == [], (
        f"Sin reranker, esperado [], obtenido {detail.pre_rerank_candidate_ids}"
    )
    assert reranked_ok is None
    print("PASS: sin reranker -> pre_rerank_candidate_ids vacio")


def test_to_dict_includes_pre_rerank_ids():
    """to_dict() incluye pre_rerank_candidate_ids cuando no vacio."""
    candidate_ids = [f"doc_{i:03d}" for i in range(10)]
    reranked_ids = ["doc_005", "doc_002"]

    detail = QueryRetrievalDetail(
        retrieved_doc_ids=candidate_ids[:5],
        retrieved_contents=["c"] * 5,
        retrieval_scores=[1.0] * 5,
        expected_doc_ids=["doc_002"],
        generation_doc_ids=reranked_ids,
        generation_contents=["c"] * 2,
        pre_rerank_candidate_ids=candidate_ids,
    )

    qer = QueryEvaluationResult(
        query_id="q1",
        query_text="test",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        retrieval=detail,
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=0.5,
    )

    d = qer.to_dict()
    assert "pre_rerank_candidate_ids" in d, "Falta pre_rerank_candidate_ids en to_dict()"
    assert d["pre_rerank_candidate_ids"] == candidate_ids
    assert d["generation_doc_ids"] == reranked_ids
    print("PASS: to_dict() incluye pre_rerank_candidate_ids")


def test_to_dict_excludes_pre_rerank_ids_when_empty():
    """to_dict() NO incluye pre_rerank_candidate_ids cuando esta vacio (sin reranker)."""
    detail = QueryRetrievalDetail(
        retrieved_doc_ids=["doc_001"],
        retrieved_contents=["content"],
        retrieval_scores=[1.0],
        expected_doc_ids=["doc_001"],
    )

    qer = QueryEvaluationResult(
        query_id="q1",
        query_text="test",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        retrieval=detail,
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=1.0,
    )

    d = qer.to_dict()
    assert "pre_rerank_candidate_ids" not in d, (
        "pre_rerank_candidate_ids no deberia estar en to_dict() cuando vacio"
    )
    assert "generation_doc_ids" not in d, (
        "generation_doc_ids no deberia estar en to_dict() cuando vacio"
    )
    print("PASS: to_dict() excluye pre_rerank_candidate_ids cuando vacio")


def test_reranker_promotes_low_ranked_doc():
    """
    Caso real: reranker promueve doc de posicion 15 al top 3.
    pre_rerank_candidate_ids permite verificar que la promocion es trazable.
    """
    evaluator = _make_evaluator(reranker_enabled=True)

    all_ids = [f"doc_{i:03d}" for i in range(20)]
    # Reranker promueve doc_015 (posicion 16) al primer lugar
    reranked_top = ["doc_015", "doc_000", "doc_001"]

    evaluator._retriever = _mock_retriever(all_ids)
    evaluator._reranker = _mock_reranker(reranked_top)

    detail, _ = evaluator._execute_retrieval(
        query_text="test",
        expected_doc_ids=["doc_015"],
    )

    # doc_015 NO esta en retrieved_doc_ids (top 5 pre-rerank)
    assert "doc_015" not in detail.retrieved_doc_ids, (
        "doc_015 no deberia estar en top 5 pre-rerank"
    )
    # doc_015 SI esta en generation_doc_ids (post-rerank)
    assert "doc_015" in detail.generation_doc_ids
    # doc_015 SI esta en pre_rerank_candidate_ids (pool completo)
    assert "doc_015" in detail.pre_rerank_candidate_ids

    # Posicion original trazable
    original_pos = detail.pre_rerank_candidate_ids.index("doc_015")
    assert original_pos == 15, f"Posicion original esperada 15, obtenida {original_pos}"

    print("PASS: doc promovido por reranker es trazable via pre_rerank_candidate_ids")


def test_to_dict_includes_metadata_when_present():
    """to_dict() incluye metadata (reranked status) cuando no vacio."""
    detail = QueryRetrievalDetail(
        retrieved_doc_ids=["doc_001"],
        retrieved_contents=["content"],
        retrieval_scores=[1.0],
        expected_doc_ids=["doc_001"],
    )

    qer = QueryEvaluationResult(
        query_id="q1",
        query_text="test",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        retrieval=detail,
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=1.0,
        metadata={"reranked": True},
    )

    d = qer.to_dict()
    assert "metadata" in d, "metadata deberia estar en to_dict() cuando no vacio"
    assert d["metadata"]["reranked"] == True
    print("PASS: to_dict() incluye metadata con reranked status")


def test_to_dict_excludes_metadata_when_empty():
    """to_dict() NO incluye metadata cuando esta vacio."""
    detail = QueryRetrievalDetail(
        retrieved_doc_ids=["doc_001"],
        retrieved_contents=["content"],
        retrieval_scores=[1.0],
        expected_doc_ids=["doc_001"],
    )

    qer = QueryEvaluationResult(
        query_id="q1",
        query_text="test",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        retrieval=detail,
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=1.0,
    )

    d = qer.to_dict()
    assert "metadata" not in d, "metadata vacio no deberia estar en to_dict()"
    print("PASS: to_dict() excluye metadata cuando vacio")


if __name__ == "__main__":
    test_pre_rerank_ids_populated_with_reranker()
    test_pre_rerank_ids_empty_without_reranker()
    test_to_dict_includes_pre_rerank_ids()
    test_to_dict_excludes_pre_rerank_ids_when_empty()
    test_reranker_promotes_low_ranked_doc()
    test_to_dict_includes_metadata_when_present()
    test_to_dict_excludes_metadata_when_empty()
