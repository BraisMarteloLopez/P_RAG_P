# RAG_P v3.2

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA actualmente) con infraestructura NVIDIA NIM.

## Arquitectura

```
RAG_P/
├── shared/                      # Libreria compartida
│   ├── types.py                 # NormalizedQuery, LoadedDataset, EvaluationRun, QueryRetrievalDetail, Protocols
│   ├── metrics.py               # F1, ExactMatch, Accuracy, SemanticSimilarity, Faithfulness (LLM-judge)
│   ├── llm.py                   # AsyncLLMService (semaforo, reintentos), load_embedding_model
│   ├── config_base.py           # Helpers _env_*, InfraConfig, RerankerConfig
│   ├── report.py                # RunExporter: JSON + CSV summary + CSV detail
│   ├── vector_store.py          # ChromaVectorStore (similarity_search + search_by_vector)
│   ├── py.typed                 # PEP 561 marker
│   └── retrieval/
│       ├── __init__.py              # Factory get_retriever()
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
│       ├── hybrid_retriever.py      # BM25 + Vector + RRF (componente interno de CONTEXTUAL_HYBRID)
│       ├── contextual_retriever.py  # Enriquecimiento LLM (Anthropic pattern) + inner hybrid
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank wrapper)
│       └── tantivy_index.py         # Indice BM25 via Tantivy (Rust, fallback a rank-bm25)
│
├── sandbox_mteb/                # Evaluacion sobre datasets MTEB/BeIR
│   ├── config.py                # MTEBConfig: .env -> dataclass validada
│   ├── loader.py                # MinIO/Parquet -> LoadedDataset (schema v2.0)
│   ├── evaluator.py             # Pipeline 3 fases: pre-embed batch + retrieval local + gen async
│   ├── run.py                   # Entry point (--dry-run, -v)
│   └── env.example              # Plantilla de configuracion
│
├── tests/                       # Tests unitarios (86 tests, mocks, sin dependencia NIM)
│   ├── test_dt5_pre_rerank_traceability.py
│   ├── test_dt6_01_faithfulness_sync.py
│   ├── test_dt6_02_faithfulness_async.py
│   ├── test_dt6_03_context_utilization.py
│   ├── test_dt7_05_06_rerank_status.py
│   ├── test_dt7_07_no_reranker.py
│   ├── test_dt7_08_csv_reranked.py
│   ├── test_dt8_09_10_11_reranker_sort.py
│   ├── test_dt9_extract_score_fallback.py
│   ├── test_dtm5_12_13_secondary_metric_errors.py
│   ├── test_format_context.py
│   └── test_metrics_reference_based.py
│
├── mypy.ini                     # Configuracion mypy incremental
├── requirements.txt
└── README.md
```

Nota: los directorios `download_datasets/` (ETL HuggingFace->Parquet->MinIO) y `data/` (cache local, resultados) no estan incluidos en este repositorio. El ETL se ejecuto una unica vez para poblar MinIO.

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo (NIM) | Cosine similarity (ChromaDB) | Opcional |
| `CONTEXTUAL_HYBRID` | Enriquecimiento LLM (Anthropic pattern) + embedding | BM25 (Tantivy) + Vector + RRF | Opcional |

`CONTEXTUAL_HYBRID` opera en dos fases: durante la indexacion, cada documento se enriquece con contexto generado por LLM (el modelo recibe el documento y genera una descripcion contextual breve). El texto enriquecido se indexa tanto en el indice vectorial (ChromaDB) como en el indice BM25 (Tantivy). Durante el retrieval, se fusionan resultados BM25 y vectoriales via Reciprocal Rank Fusion (RRF), opcionalmente se aplica reranking con cross-encoder, y se devuelve el contenido original (no enriquecido) para generacion.

`CONTEXTUAL_HYBRID` requiere LLM disponible incluso con `GENERATION_ENABLED=false` (el LLM se usa para enriquecimiento durante indexacion).

## Pipeline de evaluacion

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [enrich(LLM) si CONTEXTUAL_HYBRID]
     -> index(ChromaDB + Tantivy)
     -> pre-embed queries (batch REST NIM)
     -> retrieve(local ChromaDB + BM25 + RRF, sync)
     -> [rerank(cross-encoder) si habilitado]
     -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

Optimizacion v3.2: las queries se pre-embeben en batch via REST al NIM de embeddings antes del loop de retrieval. La fase de retrieval usa vectores pre-computados con busqueda local ChromaDB, eliminando el roundtrip REST por query. Si el pre-embed falla, el sistema hace fallback al comportamiento anterior (embedding por query).

## Metricas

### Retrieval

Hit@K, MRR, Recall@K (K=1,3,5,10,20), NDCG@K. Se computan sobre los top `RETRIEVAL_K` documentos del retriever (pre-rerank).

### Generacion

Metrica primaria adaptativa segun `answer_type` del query (campo del dataset):

- `answer_type == "label"` (yes/no): ACCURACY (match normalizado del label)
- cualquier otro (extractiva): F1 (token-overlap normalizado)

EM (Exact Match) se computa siempre como secundaria. Faithfulness (LLM-judge) se computa como secundaria si esta configurada en el dataset.

### Separacion retrieval vs generacion

Con reranker activo, las metricas de retrieval y el contexto de generacion usan conjuntos distintos de documentos: las metricas de retrieval se calculan sobre los top `RETRIEVAL_K` docs del retriever (pre-rerank), mientras que la generacion recibe los top `RERANKER_TOP_N` docs post-rerank. Sin reranker, ambos usan los mismos `RETRIEVAL_K` docs.

## Dataset: HotpotQA

Fuente: HotpotQA fullwiki (validation split) via HuggingFace.

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 documentos (Wikipedia, deduplicados por titulo) |
| Qrels | 14810 (2.0 por query, solo supporting_facts) |
| Tipos de query | bridge (~80%), comparison (~20%) |
| Almacenamiento | MinIO (Parquet), `s3://lakehouse/datasets/evaluation/hotpotqa/` |
| Schema | v2.0: query_id, text, answer, answer_type, question_type, level |

### Limitacion del corpus

El corpus se compone de 10 pasajes por query (2 gold + 8 distractores originales del dataset), deduplicados globalmente por titulo. Esto significa que cada query tiene sus gold docs garantizados en el corpus. Los resultados de retrieval **no son comparables con benchmarks publicados** (que usan el corpus completo de Wikipedia, ~5.2M articulos). Los valores absolutos son optimistas; solo las comparaciones relativas entre estrategias sobre este mismo corpus son validas.

El corpus se baraja con seed fijo antes de aplicar `EVAL_MAX_CORPUS`, evitando alineamiento artificial entre posiciones de queries y documentos en el Parquet.

## Requisitos

- Python 3.10+
- NVIDIA NIM endpoints: embedding + LLM, opcionalmente reranker
- MinIO con datasets MTEB en formato Parquet
- GPU recomendada: H100 (para NIM)

## Setup

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

## Uso

```bash
python -m sandbox_mteb.run                  # Run con config del .env
python -m sandbox_mteb.run --dry-run        # Solo validar config, no ejecutar
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Logging verbose (DEBUG)
```

Toda la parametrizacion del run (queries, corpus, estrategia, DEV_MODE, etc.) se controla via `.env`. Ver `sandbox_mteb/env.example`.

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables criticas:

```bash
# --- Embedding (NIM) ---
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://172.30.79.98:8000/v1
# "asymmetric" para NIM con input_type (query/passage). "symmetric" para gRPC/Triton.
# El NIM actual REQUIERE asymmetric; symmetric produce HTTP 400.
EMBEDDING_MODEL_TYPE=asymmetric
# Batch size para indexacion y pre-embed. Reducir si NIM da errores de buffer.
# Valor funcional con NIM actual: 5.
EMBEDDING_BATCH_SIZE=5

# --- LLM (NIM) ---
LLM_BASE_URL=http://172.30.79.99:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
# ATENCION: NIM_MAX_CONCURRENT_REQUESTS y NIM_REQUEST_TIMEOUT (no NIM_MAX_CONCURRENT ni NIM_TIMEOUT).
NIM_MAX_CONCURRENT_REQUESTS=32
NIM_REQUEST_TIMEOUT=120
NIM_MAX_RETRIES=3

# --- Retrieval ---
RETRIEVAL_STRATEGY=SIMPLE_VECTOR          # SIMPLE_VECTOR | CONTEXTUAL_HYBRID
RETRIEVAL_K=20                            # Docs para metricas retrieval (y generacion sin reranker)
RETRIEVAL_PRE_FUSION_K=150                # Candidatos pre-RRF / pool para reranker
RETRIEVAL_RRF_K=60                        # Parametro k de RRF (solo CONTEXTUAL_HYBRID)
RETRIEVAL_BM25_WEIGHT=0.5                 # Peso BM25 en RRF (solo CONTEXTUAL_HYBRID)
RETRIEVAL_VECTOR_WEIGHT=0.5               # Peso vector en RRF (solo CONTEXTUAL_HYBRID)
RETRIEVAL_BM25_LANGUAGE=en                # Stemmer Tantivy (en, es, de, fr, it, pt, ...)

# --- Contextual Retrieval (solo CONTEXTUAL_HYBRID) ---
RETRIEVAL_CONTEXT_MAX_TOKENS=150          # Max tokens del contexto generado por LLM
RETRIEVAL_CONTEXT_BATCH_SIZE=10           # Docs enriquecidos en paralelo

# --- Reranker (cross-encoder, opcional) ---
RERANKER_ENABLED=false
RERANKER_BASE_URL=http://172.30.79.98:9000/v1
RERANKER_MODEL_NAME=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_N=5                          # Docs post-rerank para generacion

# --- Dataset ---
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=0                        # 0 = todas (7405)
EVAL_MAX_CORPUS=0                         # 0 = todo (66576)
GENERATION_ENABLED=true
GENERATION_MAX_CONTEXT_CHARS=0            # 0 = auto-derivar del modelo via /v1/models
CORPUS_SHUFFLE_SEED=42                    # -1 = desactivar shuffle (NO recomendado)

# --- Modo desarrollo ---
DEV_MODE=false                            # true = subset con gold docs garantizados
DEV_QUERIES=200                           # Queries para modo dev
DEV_CORPUS_SIZE=4000                      # Corpus para modo dev (gold + distractores)

# --- MinIO ---
MINIO_ENDPOINT=http://172.30.79.110:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minio123
MINIO_BUCKET_NAME=lakehouse
S3_DATASETS_PREFIX=datasets/evaluation
```

### Interaccion entre RETRIEVAL_K y RERANKER_TOP_N

| Config | Metricas retrieval | Contexto generacion |
|---|---|---|
| Reranker OFF | top RETRIEVAL_K | top RETRIEVAL_K |
| Reranker ON | top RETRIEVAL_K (pre-rerank) | top RERANKER_TOP_N (post-rerank) |

Con reranker ON, PRE_FUSION_K candidatos se recuperan del retriever, las metricas se computan sobre los top RETRIEVAL_K de esos candidatos, y el reranker selecciona los top RERANKER_TOP_N para generacion.

### Modo desarrollo (DEV_MODE)

`DEV_MODE=true` activa un subset inteligente: selecciona `DEV_QUERIES` queries aleatorias (con seed) y garantiza que sus gold docs estan presentes en el corpus, rellenando con distractores aleatorios hasta `DEV_CORPUS_SIZE`. Ignora `EVAL_MAX_QUERIES` y `EVAL_MAX_CORPUS`.

Las metricas absolutas son optimistas (ratio gold/distractores ~10% vs ~0.003% en corpus completo). Solo validas para comparacion relativa entre estrategias sobre el mismo subset.

## Salida

Tres archivos por run en `data/results/`:

- `<run_id>.json`: detalle completo (config, metricas por query, doc_ids recuperados)
- `<run_id>_summary.csv`: 1 fila con metricas agregadas
- `<run_id>_detail.csv`: N filas, una por query (metricas individuales, respuestas generadas)


## Deuda tecnica

| ID | Descripcion | Impacto |
|---|---|---|
| DT-1 | Duplicacion de contenido en memoria: `QueryRetrievalDetail` almacena `retrieved_contents` + `generation_contents` por query. Con 7K queries y 20 docs cada uno, ~1.5GB redundante. | Memoria. No urgente con 50GB RAM. |
| DT-2 | Enriquecimiento contextual degradado: `LLMContextGenerator` usa siempre Mode B (solo chunk + titulo) porque documentos corpus no tienen `parent_content`. El patron Anthropic completo (Mode A: documento padre + chunk) no aplica a pasajes individuales de Wikipedia. | Calidad retrieval. Requiere rediseno. |
| DT-3 | Logging estructurado ausente: todo el logging usa `logging.info/warning` con f-strings. Para analisis post-hoc de multiples runs, un JSONL estructurado (`run_start`, `query_result`, `embedding_batch`, `llm_request`, `run_complete`) facilitaria comparacion entre runs via pandas/jq. El detail CSV cubre parcialmente esta necesidad. | Observabilidad. No bloqueante. |
| DT-4 | **RESUELTO.** Tipado estricto ausente. Fix: (a) `mypy.ini` con `disallow_untyped_defs` para modulos core (`shared/types.py`, `shared/metrics.py`, `shared/llm.py`, `shared/retrieval/*`, `shared/vector_store.py`). (b) `EmbeddingModelProtocol` centralizado en `types.py` (antes duplicado en `metrics.py`); `LLMJudgeProtocol` extendido con `max_tokens`. (c) Reemplazado `embedding_model: Any` por `EmbeddingModelProtocol` en: `retrieval/__init__.py`, `core.py`, `hybrid_retriever.py`, `contextual_retriever.py`, `vector_store.py`, `evaluator.py`. (d) Reemplazado `llm_service: Any` por `LLMJudgeProtocol` en `contextual_retriever.py` y `retrieval/__init__.py`. (e) `load_embedding_model()` retorna `EmbeddingModelProtocol`. (f) `py.typed` marker creado. mypy 0 errores en modulos modificados. | ~~Mantenibilidad. No bloqueante.~~ Resuelto. |
| DT-5 | **RESUELTO.** Trazabilidad reranker ausente: el JSON no almacenaba los PRE_FUSION_K candidate IDs que el reranker recibio. Fix: nuevo campo `pre_rerank_candidate_ids` en `QueryRetrievalDetail` (`shared/types.py`) que almacena solo los IDs (sin contenidos, ~3KB/query). Poblado en `evaluator.py _execute_retrieval()` cuando reranker activo. Serializado en `to_dict()` condicionalmente (solo si no vacio). Permite: verificar post-hoc que `generation_doc_ids` provienen del pool de candidatos, analizar que posiciones originales fueron promovidas, y diagnosticar cuando un doc gold no estaba en el pool. | ~~Diagnostico. No bloqueante.~~ Resuelto. |
| DT-6 | **RESUELTO. Bug: truncacion asimetrica de contexto en faithfulness.** En `shared/metrics.py`, el LLM-judge de faithfulness recibia `context[:4000]` (4000 chars hardcodeados). Sin embargo, el LLM de generacion recibia el contexto completo (truncado por `_format_context()` al limite del modelo). Fix: eliminado el truncamiento hardcodeado en las 4 funciones afectadas (faithfulness sync/async, context_utilization sync/async). El contexto que llega al judge ahora es el mismo que recibio el LLM de generacion. | ~~Correccion metrica. Alta prioridad.~~ Resuelto. |
| DT-7 | **RESUELTO. Fallback silencioso en reranker.** En `shared/retrieval/reranker.py`, si el cross-encoder falla, el fallback retorna los primeros `top_n` candidatos sin rerank, marcado como `"reranked": False` en metadata. Fix: `evaluator.py` ahora (a) detecta el flag y emite `logger.warning` identificando la query afectada, (b) acumula un contador `_rerank_failures` que se incluye en `config_snapshot` del JSON de salida, (c) propaga el estado `reranked` al `metadata` de `QueryEvaluationResult`. `report.py` expone la columna `reranked` en el detail CSV. Fix adicional post-validacion en produccion (run `090645`): `to_dict()` no serializaba `metadata` al JSON -- corregido, ahora incluye `metadata` condicionalmente cuando no vacio. Verificado en CSV (25/25 `reranked=True`) y en runtime (`rerank_failures=0`). | ~~Correccion diagnostico. Media prioridad.~~ Resuelto. |
| DT-8 | **RESUELTO. Reranker no garantiza orden descendente por score.** En `shared/retrieval/reranker.py`, `compress_documents()` no garantiza formalmente orden descendente por `relevance_score`. Fix: `sorted()` explicito por `relevance_score` descendente antes del slice `[:top_n]`. | ~~Robustez. Baja prioridad.~~ Resuelto. |
| DT-9 | **RESUELTO.** Regex permisivo en fallback de score LLM-judge: en `shared/metrics.py` `_extract_score_fallback()`, el patron `(?:score[:\s]*)?([01]\.?\d*)` capturaba parciales de numeros mayores (ej: "10.5" -> "10" -> 1.0 falso positivo). Fix: reescrita la funcion con 3 patrones explicitos en orden de especificidad: (1) fracciones N/M, (2) decimales 0.X/1.0 con word boundaries, (3) enteros 1-10 solo con prefijo "score:". Elimina la normalizacion ciega `value/10` que convertia cualquier parcial en score valido. | ~~Correccion metrica. Media prioridad.~~ Resuelto. |

### Deudas tecnicas menores

| ID | Descripcion | Impacto |
|---|---|---|
| DTm-1 | Duplicacion de parsing en `loader.py`: `load_dataset()` y `_load_from_cache()` repiten la misma logica de conversion DataFrame -> NormalizedQuery/NormalizedDocument (~50 lineas duplicadas). Extraer a un metodo privado `_dataframes_to_dataset()`. | Mantenibilidad. Trivial. |
| DTm-2 | Fragilidad de API interna en `vector_store.py`: `similarity_search_by_vector_with_score()` accede a `self._store._collection` (API interna de LangChain Chroma, no publica). Puede romperse sin aviso en actualizaciones de `langchain-chroma`. Alternativa: usar el cliente nativo `chromadb` directamente (bypass LangChain). | Estabilidad. Riesgo en upgrades de dependencias. |
| DTm-3 | `_run_async()` incompatible con event loops activos: usa `asyncio.run()` directamente, lo cual falla en contextos con event loop preexistente (Jupyter, frameworks async). Problema adicional: `AsyncLLMService._semaphore` se crea en `__init__` (fuera de cualquier event loop), pero `asyncio.run()` crea un loop nuevo por invocacion. En Python 3.10+ esto funciona porque `Semaphore` ya no se vincula a un loop en construccion, pero si `invoke()` sync se llamara desde multiples threads, cada `asyncio.run()` crearia su propio loop y el semaforo no cumpliria su funcion de limitar concurrencia. El pipeline CLI no se ve afectado (single-threaded). | Portabilidad. No bloqueante para uso actual. |
| DTm-4 | **PARCIALMENTE RESUELTO.** Tests ausentes en repositorio. Se han implementado 86 tests unitarios en `tests/` cubriendo: DT-5 (trazabilidad reranker), DT-6 (faithfulness context), DT-7 (rerank status), DT-8 (reranker sort), DT-9 (score fallback), DTm-5 (secondary metric errors), metricas reference-based (F1, EM, Accuracy, normalizer), y format_context. Pendiente: cobertura del pipeline core (subset selection, RRF, MinIO loader, Tantivy edge cases) y CI/CD. Ver seccion "Tests unitarios > Cobertura pendiente". | Calidad. |
| DTm-5 | **RESUELTO.** Error handling silenciado en metricas secundarias: en `evaluator.py` `_calculate_metrics_async()`, las excepciones de metricas secundarias se capturaban con `pass`. Fix: el `except` ahora (a) emite `logger.warning` con el tipo de metrica, mensaje de error, y fragmento de la query afectada, (b) crea un `MetricResult` con `value=0.0` y `error` descriptivo que se propaga al dict de secundarias. Esto hace que la key exista en el JSON/CSV (en lugar de desaparecer) y que el fallo sea rastreable via logs. | ~~Diagnostico. No bloqueante.~~ Resuelto. |
| DTm-6 | Sin retry en pre-embed batch: en `evaluator.py` `_batch_embed_queries()`, si un batch de embedding falla, se abandona todo el pre-embed (retorna lista vacia). El fallback per-query funciona, pero un retry por batch antes de abandonar seria trivial de implementar. | Rendimiento. No bloqueante (fallback funcional). |
| DTm-7 | `_run_async` exportado como API publica pese a nombre privado: en `shared/llm.py`, `_run_async` tiene prefijo `_` indicando caracter privado, pero esta en `__all__` y se importa explicitamente en `evaluator.py` y `contextual_retriever.py`. Es parte de la API de facto del modulo. Renombrar a `run_async_sync` (o similar sin prefijo) o encapsular como metodo estatico en una clase helper. | Claridad API. Trivial. |
| DTm-8 | Nombre confuso `failure_rate_at_k`: en `evaluator.py` `_build_run()`, `failure_rate = {k: 1.0 - v for k, v in avg_recall.items()}`. Esto calcula `1 - avg_recall@K`, no el porcentaje de queries con 0 hits (que seria la interpretacion natural de "failure rate"). El nombre induce a error en analisis post-hoc. Renombrar a `complement_recall_at_k` o documentar la semantica real en el JSON/CSV. | Claridad. Trivial. |
| DTm-9 | `get_full_text()` concatena titulo + contenido para indexacion sin documentar sesgo: en `evaluator.py` linea 454, `doc.get_full_text()` produce `"{titulo}\n\n{contenido}"` para indexar en ChromaDB. Si el titulo es largo o muy generico (ej: "Wikipedia"), el embedding se sesga hacia el titulo. Esta es una decision de diseno legitima, pero no esta documentada como tal ni es configurable. | Documentacion. No bloqueante. |
| DTm-10 | Riesgo de colision en `collection_name`: en `evaluator.py` linea 441, `f"eval_{dataset_name}_{uuid.uuid4().hex[:8]}"` usa 8 hex chars (32 bits). Para uso CLI secuencial no hay problema. Si se ejecutan multiples runs en paralelo sobre el mismo directorio ChromaDB, la probabilidad de colision es baja pero no cero (~1 en 4 mil millones). Considerar usar el `run_id` completo como collection_name. | Robustez. No bloqueante. |
| DTm-11 | `LLMMetrics._lock` compartido en copias: en `shared/llm.py`, `_lock: asyncio.Lock = field(default_factory=asyncio.Lock)` crea un Lock nuevo por instancia, lo cual es correcto. Sin embargo, si `LLMMetrics` se copia via `copy.copy()`, la copia shallow comparte la referencia al Lock original. No hay copias en el flujo actual, pero el dataclass no protege contra este uso. | Fragilidad. No bloqueante. |
| DTm-12 | Sesgo sistematico del LLM-judge en faithfulness para respuestas yes/no. Observado en run `mteb_hotpotqa_20260218_145705`: queries de tipo yes/no reciben faithfulness 0.1-0.2 incluso cuando F1=1.0 y la respuesta es correcta (ej: "Are Giuseppe Verdi and Ambroise Thomas both Opera composers?" -> generated="yes", F1=1.0, faithfulness=0.2). El prompt del judge penaliza respuestas cortas porque no hay suficiente texto para evaluar "fundamentacion en el contexto". Inspeccionar prompts de faithfulness en `shared/metrics.py` y considerar: (a) normalizacion de score cuando respuesta generada es binaria, o (b) branch especifico en el prompt del judge para respuestas de una sola palabra. Afecta `avg_generation_score` y diagnostico post-hoc. | Sesgo metrica. No bloqueante. |
| DTm-13 | No-determinismo HNSW entre runs. Observado comparando runs `145705` (rerank OFF) vs `090645` (rerank ON): 6/25 queries tienen `retrieved_doc_ids` (top-20 pre-rerank) diferentes pese a usar mismo embedding, mismo corpus, mismo seed=42, y `hnsw_num_threads=1`. Causa: ChromaDB 0.5-0.6 no soporta `hnsw:random_seed`, y los niveles del grafo HNSW se asignan con PRNG no controlado. Diferente `collection_name` (uuid) implica diferente instancia interna. Impacto: Recall@K varia +/-0.02 entre runs. Mitigacion posible: fijar random_seed si ChromaDB lo soporta en version futura, o usar persistent client con coleccion nombrada fija. | Reproducibilidad. No bloqueante. |

### Tests unitarios

Tests en `tests/`. Ejecutables sin infraestructura NIM (usan mocks). Sin dependencias externas.

```bash
cd RAG_P
python3 tests/<nombre_test>.py
```

| Archivo | Cubre | Resultado |
|---|---|---|
| `test_dt6_01_faithfulness_sync.py` | DT-6: `faithfulness()` sync no re-trunca contexto >4000 chars | PASS |
| `test_dt6_02_faithfulness_async.py` | DT-6: `faithfulness_async()` no re-trunca contexto >4000 chars | PASS |
| `test_dt6_03_context_utilization.py` | DT-6: `context_utilization` sync/async no re-trunca + caso borde contexto vacio retorna 0.0 sin invocar judge | PASS |
| `test_dt7_05_06_rerank_status.py` | DT-7: rerank exitoso retorna `reranked_ok=True` con contador=0; rerank fallido retorna `reranked_ok=False` con contador incrementando y warning en log | PASS |
| `test_dt7_07_no_reranker.py` | DT-7: sin reranker, `reranked_status=None`, `rerank_failures=None` en config_snapshot | PASS |
| `test_dt7_08_csv_reranked.py` | DT-7: columna `reranked` en detail CSV con valores True/False/vacio segun metadata | PASS |
| `test_dt8_09_10_11_reranker_sort.py` | DT-8: sort explicito por relevance_score descendente, scores identicos no fallan, doc sin score queda al final con 0.0 | PASS |
| `test_dtm5_12_13_secondary_metric_errors.py` | DTm-5: metrica secundaria fallida produce MetricResult con error y value=0.0 (no desaparece); fallo emite warning con tipo de metrica y query; caso borde todas las secundarias fallan | PASS |
| `test_metrics_reference_based.py` | TextNormalizer (normalize, accents, dashes, tokenize, articles) + ReferenceBasedMetrics: f1_score (identico, cero, parcial, duplicados, dashes, acentos, empty), exact_match (identico, case, puntuacion, acentos, dashes, empty, sin normalize), accuracy (match, mismatch, case, valid_labels, extra text, empty) — 33 casos | PASS |
| `test_format_context.py` | `_format_context()`: placeholder vacio, headers [Doc N] numerados, separador \\n\\n, truncacion por max_chars, primer doc excede limite, boundary exacto, docs con string vacio — 9 casos | PASS |
| `test_dt9_extract_score_fallback.py` | DT-9: decimales 0-1 validos, falsos positivos eliminados ("10.5", "100", "20"), escala 1-10 con/sin prefijo, fracciones (8/10, 1/2, 0/0), default 0.5, respuestas realistas LLM — 21 casos | PASS |
| `test_dt5_pre_rerank_traceability.py` | DT-5/DT-7: pre_rerank_candidate_ids poblado con PRE_FUSION_K IDs, vacio sin reranker, incluido/excluido en to_dict() condicionalmente, doc promovido por reranker trazable a posicion original, metadata con reranked status serializado/excluido en to_dict() -- 7 casos | PASS |

#### Cobertura pendiente

Areas sin tests que representan riesgo de regresion. Ordenadas por impacto:

| Prioridad | Area | Modulo | Que testear |
|---|---|---|---|
| Media | Subset selection | `evaluator.py` | `_select_subset_dev()`: gold docs presentes en corpus, distractores rellenan hasta dev_corpus_size, seed determinista, gold_ids > dev_corpus_size lanza error. |
| Media | RRF | `hybrid_retriever.py` | `reciprocal_rank_fusion()`: rankings vacios, pesos desiguales, doc presente en un solo ranking, top_n menor que candidatos. |
| Baja | MinIOLoader | `sandbox_mteb/loader.py` | `_load_from_cache()` vs `load_dataset()` producen LoadedDataset equivalente (validar DTm-1). Mock de boto3 client. |
| Baja | TantivyIndex | `shared/retrieval/tantivy_index.py` | `search()`: query con caracteres especiales (apostrofes, comillas, `?`), query vacia, indice vacio. |

#### Mejoras estructurales pendientes en tests

| Mejora | Estado | Descripcion |
|---|---|---|
| Migracion a pytest nativo | Pendiente | Los tests actuales usan `if __name__ == "__main__"` con `assert` manual. Migrar a funciones `test_*` con assertions nativas de pytest para reporting, parametrize, y ejecucion batch (`pytest tests/`). El `requirements.txt` ya incluye pytest. |
| `conftest.py` | Pendiente | Centralizar setup de `sys.path` y mocking de modulos de infra (`boto3`, `langchain_*`, `chromadb`) en `tests/conftest.py`. Eliminar los bloques `sys.path.insert` y `sys.modules[mod] = MagicMock()` repetidos en cada archivo de test. |
| `pytest.ini` o `pyproject.toml [tool.pytest]` | Pendiente | Configurar `testpaths = ["tests"]`, `pythonpath = ["."]` para eliminar manipulacion manual de `sys.path`. |
| CI/CD | Pendiente | Ejecutar `pytest tests/` en pipeline de integracion. Sin dependencias de infra (mocks), ejecutable en cualquier entorno con Python 3.10+. |