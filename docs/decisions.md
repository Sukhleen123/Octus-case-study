# Design Decisions

## Bootstrap Definition

"Bootstrap" means the app's startup initializer: a single function `ensure_ready(settings)` that takes the system from _raw files only_ → _processed artifacts + indexes + runtime objects ready_, in an idempotent way (safe to call repeatedly without rebuilding if inputs haven't changed).

---

## Data Format: Parquet vs CSV

| | Parquet | CSV |
|--|---------|-----|
| **Type safety** | Binary-typed columns | String-typed, type ambiguous |
| **Size** | Compressed columnar (typically 5–10× smaller) | Uncompressed, larger |
| **Query speed** | Very fast for column-selective reads | Slower, row-oriented |
| **Human readability** | Binary (needs viewer) | Plain text, openable in Excel |
| **Recommended for** | Production analytics, DuckDB integration | Debugging, inspection |

**Default: `table_format=parquet`**. Set `TABLE_FORMAT=csv` in `.env` to use CSV throughout.

Note: `company_ids` is stored as a JSON-encoded string (`json.dumps([...])`) in both formats because pyarrow does not handle `list[int]` columns consistently across all parquet write/read scenarios.

**References:**
- [Apache Parquet official](https://parquet.apache.org/)
- [DuckDB Parquet overview](https://duckdb.org/docs/stable/data/parquet/overview.html)
- [DuckDB: CSV vs Parquet](https://duckdb.org/2024/12/05/csv-files-dethroning-parquet-or-not.html)

---

## DuckDB for SimFin Tabular Data

DuckDB is an embedded analytical SQL engine — it runs in-process, requires no server, and can efficiently query Parquet files and DataFrames via SQL. It is well-suited for financial statement data where users want to run aggregations, joins, and time-series queries.

**Why not SQLite?** SQLite is row-oriented and slower for analytical workloads. DuckDB's columnar engine makes it significantly faster for aggregation queries over wide financial statement tables.

**References:**
- [DuckDB Parquet overview](https://duckdb.org/docs/stable/data/parquet/overview.html)

---

## Vectorstore: Pinecone vs FAISS

| | Pinecone | FAISS |
|--|---------|-------|
| **Metadata filtering** | Native server-side filter at query time | External store required; Python-side filtering |
| **Setup** | Requires API key + cloud index | Local, no API key needed |
| **Scalability** | Managed, scales to billions | Limited by RAM |
| **Latency** | Network round-trip | Local (fast) |
| **Default** | If `PINECONE_API_KEY` set | Fallback (auto-detect) |

**FAISS metadata filtering approach:**
FAISS is a pure vector similarity library — it has no metadata filtering capability ([FAISS GitHub README](https://github.com/facebookresearch/faiss)). Our `FAISSStore` over-fetches (`k × 10`) candidates, then applies Python-side filtering using a SQLite `MetadataStore`.

**Auto-select logic:** `vectorstore_provider=auto` → uses Pinecone if `PINECONE_API_KEY` is set; otherwise FAISS.

**References:**
- [Pinecone: filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata)
- [Pinecone: indexing overview](https://docs.pinecone.io/guides/index-data/indexing-overview)
- [Pinecone: data modeling](https://docs.pinecone.io/guides/index-data/data-modeling)
- [FAISS GitHub README](https://github.com/facebookresearch/faiss)

---

## Chunking Strategy

| Chunker | Approach | Strengths | Weaknesses |
|---------|---------|-----------|-----------|
| **Heading** | Split on HTML `<h1>-<h6>` or Markdown `##` headers | Preserves document structure; improves citation readability; sections are semantically coherent | Sections can be very large; irregular heading use in some documents |
| **Token** | Sliding window (tiktoken, `chunk_size` tokens, `chunk_overlap` overlap) | Bounded, predictable size; robust when structure is inconsistent; easy to tune | No structural awareness; may split mid-sentence or mid-thought |

**Decision:** Always build both chunkers (`chunkers_to_build: ["heading", "token"]`) for A/B comparison. The active retriever uses the first chunker in the list (configurable). Both tables are stored and can be switched via settings.

---

## Retrieval Strategy

| Retriever | Strategy | Tradeoffs |
|-----------|---------|-----------|
| **Dense** (`retriever_id=dense`) | Embed query → vector search → metadata filter | Fast, controllable, deterministic results |
| **Dense+MMR** (`retriever_id=dense_mmr`) | Dense search → MMR re-ranking | Reduces redundancy (avoids returning 5 chunks from same paragraph); slightly slower due to re-embedding |

**MMR:** Maximal Marginal Relevance selects results that are relevant to the query while maximally different from already-selected results. `mmr_lambda=1.0` → pure relevance (same as dense); `mmr_lambda=0.0` → pure diversity.

---

## SimFin: Batch vs Realtime

| | Batch (`simfin_mode=batch`) | Realtime (`simfin_mode=realtime`) |
|--|---------------------------|----------------------------------|
| **Freshness** | Depends on simfin package update cadence (typically daily) | Fresh on each request |
| **Latency** | Negligible (local DuckDB query) | Network round-trip + rate limiting |
| **Complexity** | Simple: download once, query locally | Higher: cache management, rate limiting |
| **Recommended for** | Production chat UX, historical analysis | "Latest" queries, monitoring use cases |

**Realtime caching:** All v3 API responses are cached in SQLite keyed by `sha256(method + url + sorted_params)` with configurable TTL (default: 3600s). This is mandatory to respect SimFin rate limits.

**References:**
- [SimFin Python API docs](https://simfin.readthedocs.io/)
- [SimFin GitHub repo](https://github.com/SimFin/simfin)
- [SimFin load variants](https://github.com/SimFin/simfin/blob/master/simfin/load.py)
- [SimFin API v3: getting started (Authorization header)](https://simfin.readme.io/reference/getting-started-1)
- [SimFin API v3: rate limits](https://simfin.readme.io/reference/rate-limits)
- [SimFin API v3: list companies](https://simfin.readme.io/reference/list-1)
- [SimFin API v3: statements](https://simfin.readme.io/reference/statements-1)
- [SimFin major update (base URL)](https://www.simfin.com/en/blog/major-simfin-update/)

---

## Streamlit Caching

`@st.cache_resource` is used to cache `ensure_ready(settings)` across Streamlit reruns. This is the correct Streamlit primitive for singleton objects (database connections, models, index handles) that should be shared across all users and reruns in the same process.

`@st.cache_data` would be wrong here because it pickles and copies results, which is not appropriate for non-serializable objects like FAISS indexes or database connections.

`st.cache_resource` + `filelock` together guarantee:
1. Only one build runs at a time (even if multiple browser tabs trigger simultaneous reruns)
2. The built context is reused across all subsequent reruns (fast path)

**References:**
- [Streamlit caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [st.cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
- [st.cache_data](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)

---

## Agent Architecture (4-5 Agents)

| Agent | Role | Inputs | Outputs |
|-------|------|--------|---------|
| **Router** | Classify query as octus/simfin/both | Query text | Route decision |
| **DocAgent** | Retrieve Octus document chunks | Query + filters | Excerpts + OctusCitations |
| **SimFinAgent** | Retrieve financial metrics | Query + tickers | DataFrames + SimFinCitations |
| **SynthesisAgent** | Merge results + produce final answer | Doc + SimFin results | SynthesisResult |
| **Orchestrator** | Coordinate all agents | Query | SynthesisResult |

**Why separate agents?**
- **Separation of concerns**: each agent has a single, testable responsibility
- **Easier debugging**: trace events pinpoint exactly which agent produced which result
- **Clearer UI traceability**: each agent_start/agent_end event pair maps to a visible UI step
- **Independent upgrades**: replace the router with an LLM without changing other agents

---

## Configuration / Environment Variables

| Variable | Required | Default | Purpose |
|----------|---------|---------|---------|
| `SIMFIN_API_KEY` | For SimFin | `""` | SimFin authentication |
| `SIMFIN_BASE_URL` | No | `https://backend.simfin.com` | SimFin API base URL |
| `PINECONE_API_KEY` | For Pinecone | `""` | Pinecone authentication |
| `PINECONE_INDEX_NAME` | No | `octus-financial` | Pinecone index |
| `PINECONE_NAMESPACE` | No | `default` | Pinecone namespace |
| `EMBEDDING_PROVIDER` | No | `mock` | `mock` / `openai` |
| `VECTORSTORE_PROVIDER` | No | `auto` | `auto` / `pinecone` / `faiss` |
| `TABLE_FORMAT` | No | `parquet` | `parquet` / `csv` |
| `SIMFIN_MODE` | No | `batch` | `batch` / `realtime` |
| `SIMFIN_PERIODICITY` | No | `annual` | `annual` / `quarterly` / `both` |
| `MAPPING_MODE` | No | `auto_matched` | `auto_matched` / `confirmed` / `both` |

All variables map to `Settings` fields in `src/app/settings.py`. See `configs/app.yaml` for full defaults.

---

## FastAPI (Optional)

If included, use `lifespan` events for startup so `ensure_ready()` runs once at server start:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = ensure_ready(Settings())
    yield

app = FastAPI(lifespan=lifespan)
```

Reference: [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/)
