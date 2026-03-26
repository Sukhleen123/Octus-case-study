# Design Decisions

## Bootstrap Definition

"Bootstrap" means the app's startup initializer: a single function `load_context(settings)` that takes the system from _raw files only_ → _processed artifacts + runtime objects ready_, returning an `AppContext(retriever, graph)`. Safe to call repeatedly; Streamlit caches the result via `@st.cache_resource`.

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

**6 tables:** `income_annual`, `income_quarterly`, `balance_annual`, `balance_quarterly`, `cashflow_annual`, `cashflow_quarterly`. Tables are fully replaced on each ingestion run (DROP + CREATE).

**Annual vs quarterly split logic:**
- Income statements and cashflow: annual rows have `Fiscal Period == "FY"`, quarterly = all other periods
- Balance sheet: annual rows have `Fiscal Period == "Q4"` (year-end snapshot), quarterly = Q1/Q2/Q3

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
FAISS is a pure vector similarity library — it has no metadata filtering capability. `FAISSStore` over-fetches (`k × 10`) candidates, then applies Python-side filtering using a SQLite `MetadataStore`.

**References:**
- [Pinecone: filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata)
- [Pinecone: indexing overview](https://docs.pinecone.io/guides/index-data/indexing-overview)
- [FAISS GitHub README](https://github.com/facebookresearch/faiss)

---

## Multi-Namespace Pinecone Design

The system uses **two Pinecone namespaces** within a single index: `transcripts` and `sec_filings`.

**Why two namespaces instead of one?**
- Transcripts and SEC filings have fundamentally different content and retrieval patterns. Many queries are source-specific: "what did the CEO say on the earnings call" targets transcripts; "risk factors in the 10-K" targets SEC filings.
- Two namespaces let `MultiStoreRetriever` route to the correct store, avoiding noise from mixing 10-K risk factor boilerplate with earnings call Q&A.
- SEC-keyword queries (risk factors, MD&A, balance sheet, item 7) default to SEC-first ordering; conversational queries default to transcript-first.
- Namespaces share one Pinecone index — no extra cost, but with logical separation and independent `clear()` operations.

**MetadataStore (SQLite) for chunk text:**

Pinecone enforces a hard 40 KB metadata limit per vector. Financial document chunks — especially SEC filing sections containing financial tables — regularly exceed this.

Solution: chunk `text` is stored locally in SQLite (`MetadataStore`) keyed by `chunk_id`, with a separate SQLite file per namespace. Only structured metadata fields (company_name, doc_source, document_type, document_date, chunk_id) are stored in Pinecone. On query, Pinecone returns IDs + scores; the MetadataStore batch-fetches texts by chunk_id (O(1) indexed lookup), which are rejoined before returning to the retriever.

---

## Embedding Model

**Model: `Alibaba-NLP/gte-ModernBERT-base`** (default for `sentence_transformers` provider)

| Property | Value |
|----------|-------|
| Dimension | 768 |
| Max sequence length | 8,192 tokens |
| MTEB score | 64.38 |
| Task prefixes | Not required |
| `trust_remote_code` | Not required |

**Why not `intfloat/e5-base-v2`?**

The previous default, `e5-base-v2`, has a 512-token max sequence length. `SECSectionChunker` produces chunks up to 2,048 tokens (hard cap on large ITEM 7 sections). At 512 tokens, the embedding model silently truncates ~75% of those chunks — the input is truncated before the dense forward pass, so the resulting vector represents only the first ~400 words of a section that may be 1,600+ words. This makes long SEC sections unreliable to retrieve: their vectors only capture the opening paragraph.

**Why not use two separate models?**

Transcript and SEC chunks are stored in the same Pinecone index (different namespaces but identical dimension). Pinecone indexes enforce a single dimension. Using two models would require two separate indexes, two Pinecone API keys or index configurations, and query-time routing to select the right model per namespace. The cleaner fix: one model with sufficient context for both document types.

**Why gte-ModernBERT-base?**

- **768-dim** — same as e5-base-v2; no Pinecone index rebuild required
- **8,192-token context** — comfortably handles the 2,048-token SECSectionChunker cap with 4× headroom
- **No task prefixes** — E5 models require prepending `"query: "` / `"passage: "` to embed correctly; gte-ModernBERT does not. The existing `_needs_prefix()` in `SentenceTransformerEmbedder` returns False for non-E5 model names, so no code change needed there
- **MTEB 64.38** vs e5-base-v2 ~62 — modestly stronger on retrieval benchmarks

---

## Chunking Strategy

| Chunker | Approach | Used for | Strengths | Weaknesses |
|---------|---------|----------|-----------|------------|
| **RecursiveChunker** | Paragraph merge + sentence-boundary split (tiktoken) | Transcripts | Handles free-flowing prose; respects sentence boundaries; consistent token size | No structural awareness; may split mid-topic |
| **SECSectionChunker** | TOC-anchor DOM split (primary) → PART/ITEM regex fallback; HTML table preservation | SEC filings (10-K, 10-Q) | Preserves legal/regulatory structure; uses actual TOC section titles; tables readable | Wdesk/generic-ID filings fall back to regex; some ITEM sections are very long |

**Why two separate chunkers for two document types?**

Transcripts are free-flowing prose in Q&A format with no reliable structural headers. `RecursiveChunker` — which merges small paragraphs and splits on sentence boundaries — handles this well. 512-token chunks are long enough to contain a full analyst question + executive answer.

SEC filings have a strict regulatory structure defined by `PART` and `ITEM` headings (mandated by the SEC). The `SECSectionChunker` splits on these boundaries so retrieval returns semantically coherent sections (e.g., all of "Item 1A — Risk Factors" as a unit, not a mid-sentence break).

`SECSectionChunker` uses two parsing paths depending on the filing format:

1. **TOC-anchor (primary)**: Most 10-Ks and 10-Qs embed a machine-readable table of contents as `<a href="#item_1_business">` links pointing to `<p id="item_1_business">` anchor elements. The chunker detects these by looking for ≥3 `href="#..."` links whose text matches a PART/ITEM pattern. It then walks the DOM, flushing a new section each time it hits a matching anchor element. Section titles come directly from the TOC text — so the actual document-specific names ("Part I — Financial Information (Unaudited): Item 1.") are preserved rather than guessed from a static dict.

2. **Regex-text (fallback)**: Wdesk-generated filings (e.g., Ford Credit subsidiary filings) use opaque UUID-style IDs (`id="iae0d8ad5e..."`) with no corresponding TOC href links. The chunker extracts plain text and falls back to splitting on `PART`/`ITEM` boundary regex patterns, with titles resolved via a static item-number-to-title dict.

**Why tiktoken `cl100k_base`?**
Token counting is consistent with OpenAI embedding models (`text-embedding-3-*`). This prevents over-chunking when the embedder tokenizes differently from the chunker.

**Why `target_size=512, overlap=64`?**
512 tokens ≈ 400 words — long enough to contain a full financial discussion or regulatory paragraph, short enough for embedding quality (most embedding models degrade beyond ~512 tokens of input). 64-token overlap prevents losing context at boundaries (the end of one chunk repeats as the start of the next).

**Why preserve HTML tables in SEC filings as markdown?**
Financial tables (balance sheets, income statements embedded in the HTML body) would become garbled text if the `<table>` tags were stripped. Converting to markdown tables preserves row/column relationships, making them both retrievable (the cell values appear in the chunk text) and readable in citations.

**Why sub-split large SEC sections at 2048 tokens?**
Some ITEM 7 (MD&A) sections run 10,000+ tokens. Embedding a 10,000-token chunk produces a low-quality vector (the model averages over too much content). Hard cap at 2048 tokens keeps each chunk focused enough for meaningful similarity search.

---

## Retrieval Strategy

| Retriever | Strategy | Tradeoffs |
|-----------|---------|-----------|
| **Dense** (`retriever_id=dense`) | Embed query → vector search → metadata filter → top-k | Fast, controllable, deterministic results |
| **Dense+MMR** (`retriever_id=dense_mmr`) | Dense over-fetch (k×4) → MMR re-rank → top-k | Reduces redundancy; returns diverse results across companies/periods; slightly slower |

**MMR (Maximal Marginal Relevance):**

For queries spanning multiple companies or time periods, dense retrieval returns multiple chunks from the same document (the most similar one). MMR re-ranks to balance relevance with diversity — ensuring top-k results come from different documents/companies rather than 5 chunks from the same 10-K.

Score: `lambda × relevance_to_query − (1 − lambda) × max_similarity_to_already_selected`

- `mmr_lambda=1.0` → pure relevance (equivalent to dense)
- `mmr_lambda=0.5` → balanced (default)
- `mmr_lambda=0.0` → pure diversity

**Over-fetch factor of 4**: `dense_mmr` retrieves `k × 4` candidates from Pinecone, then re-ranks. Trade-off: 4× the vector search cost, but much better result diversity. Re-ranking requires re-embedding the candidate texts (not query vectors) to compute pairwise similarity.

**Boilerplate de-prioritization:**

SEC filings contain mandatory boilerplate sections (Safe Harbor Statements, Cautionary Language About Forward-Looking Statements) that are legally required but contain no company-specific information. These chunks often score high on vector similarity for queries about "forward-looking statements", crowding out informative content.

`MultiStoreRetriever` detects chunks whose `section_title` matches `_BOILERPLATE_HEADERS` and demotes them to the end of the result list, only returned if fewer than `top_k` clean chunks are available.

---

## Agent Framework: LangGraph

**Why LangGraph over Google ADK / plain LangChain?**

LangGraph models the pipeline as an explicit directed graph with typed state. The `AgentState` Pydantic model is the contract — every node reads from and writes to it, making data flow explicit and debuggable.

| Need | LangGraph approach |
|------|--------------------|
| Parallel doc + simfin retrieval | Native fan-out: router → (doc_agent, simfin_agent) → synthesize |
| Trace accumulation across nodes | `operator.add` reducer on `trace_events` field — no coordination needed |
| Live streaming UI updates | `stream_mode="updates"` yields per-node output as each node completes |
| Typed state contract | Pydantic `AgentState` enforced at every node boundary |

Plain LangChain chains don't support parallel branches or shared typed state across agents cleanly. Google ADK requires more scaffolding for the same pattern.

**Why 4 nodes?**

- **Router**: Performs all inference upfront (routing, company/sector extraction, temporal parsing, table selection). Can be upgraded to an LLM classifier without touching downstream nodes.
- **DocAgent**: Executes the retrieval strategy resolved by the router. Deterministic — no reasoning, only execution.
- **SimFinAgent**: Executes DuckDB queries against the pre-selected tables with pre-computed limits.
- **Synthesize**: Sees only selected chunks and citations. Clean inputs → cleaner outputs.

**Router intelligence:** The router node now performs 6 steps before any agent runs: route classification, company/sector extraction, ticker resolution, doc filter inference, temporal filter extraction, and SimFin table selection. By the time the router completes, all retrieval parameters are fully resolved. Downstream agents are execution engines.

---

## HyDE Query Expansion (Optional)

HyDE (Hypothetical Document Embeddings) is an optional router step (`hyde=False` by default) that only activates when `HYDE_TRIGGERS` matches the query — patterns like "forward-looking statements", "management guidance/outlook/commentary", "what did X say about", or "what does management expect/anticipate".

**The problem it solves:** Certain queries were over-matching the wrong vector store. A query like "what is management's guidance on free cash flow?" is phrased as a question — its embedding sits in a different region of the vector space than the actual financial document text it's trying to retrieve. This caused some of these queries to pull from SEC boilerplate (e.g., Safe Harbor language, which is dense with "forward-looking" phrasing) rather than the substantive transcript excerpts where management actually discusses guidance.

**How it works:** `maybe_expand_query()` prompts Claude to write a 2-3 sentence passage from a financial filing or earnings call that *would be* the ideal answer to the query — using specific numbers, guidance ranges, or management quotes typical of financial documents. That hypothetical passage (not the original question) is then embedded and used as the query vector.

**Why optional?** HyDE adds one LLM call per query and can occasionally over-specify — generating a passage that assumes facts not in evidence. It's most useful for qualitative management commentary queries; for factual lookups ("what was ORCL revenue in Q3?") the original query embeds well enough on its own.

---

## Doc Agent: Deterministic Retrieval + Sparse Retry

The doc agent does **not** use a tool loop. The retrieval strategy is fully determined by the router before the agent runs:

- **Single-company / filtered queries**: one call to `_retrieve_documents()` with company, doc_type, and date range filters already resolved by the router.
- **Multi-company / sector queries** (`all_companies_mode=True` or `sector_mode` with >1 company): `_retrieve_for_each_company()` runs in parallel (ThreadPoolExecutor), fetching up to 4 chunks per company. This ensures equal representation across companies regardless of embedding similarity — critical for sector-wide queries ("common trends across cable companies").
- **Sparse result retry**: If initial retrieval returns <`RETRY_THRESHOLD=3` chunks AND an LLM is configured, `_retry_with_llm()` asks Claude to generate 3 alternative query rephrasings. Each rephrasing is retried. If still sparse, the company filter is dropped to broaden the search. This is a safety net, not the primary path.

**Why not a tool loop?** A tool loop adds latency and non-determinism — the number of LLM calls depends on agent judgment at runtime. The router has enough context to determine upfront whether to do targeted vs parallel retrieval. The LLM is only invoked reactively when retrieval results are provably poor (sparse). MAX_FINAL=15 chunks are forwarded to synthesis after deduplication.

---

## Temporal Filtering in Router

The router parses temporal intent from natural language into structured filters before any retrieval occurs:

**Document date filters** (`extract_doc_temporal_filters()`):
- "last 3 quarters" → `doc_date_from = ~9 months ago` (ISO string), `doc_date_to = today`
- "in 2023" → date range for that fiscal year
- "recent" → last 6 months

**SimFin temporal filters** (`extract_simfin_temporal_filters()`):
- "last 4 quarters" → `simfin_max_quarterly_periods=4`
- "FY2022" → `simfin_date_from="2022"`
- For annual data: `simfin_max_annual_periods` controls how many fiscal years to return

**SimFin table selection** (`select_simfin_tables()`):
Avoids querying all 6 DuckDB tables on every request:
- All-companies queries → `[income_annual, balance_annual]` only (manageable response size)
- Cash flow keywords → cashflow tables
- Quarterly keywords → `income_quarterly`, `balance_quarterly`, `income_annual`
- Default → all 6

---

## Sector Detection

`SECTOR_TRIGGERS` regex detects queries about a sector or industry ("common trends across cable companies", "compare companies in the media industry", "risks seen across the sector").

When triggered:
- `extract_sector_companies()` looks up companies in `company_map` by `sub_industry` field
- Level 1: exact word match on sub_industry
- Level 2: rapidfuzz fuzzy match (threshold ≥75%) for terminology variation
- Sets `sector_mode=True` on AgentState

With `sector_mode=True`, the doc agent uses `_retrieve_for_each_company()` (parallel, not filtered by company name), ensuring all sector companies contribute equally to the retrieved context — critical for cross-company analysis queries.

---

## Citation Numbering System

**Design rationale:**

The synthesis agent receives a pre-numbered citation list in its prompt. Claude is instructed to cite by number: financial metrics must be cited, document excerpts must be cited with the source chunk number, long quotes use markdown blockquotes with [N] at the end.

After synthesis, a regex `\[(\d+)\]` extracts all reference numbers actually used in the answer. Only citations that appear in the answer receive `ref_number > 0` — unused retrieved context is silently dropped.

**Why this matters:**
- The user sees a minimal, accurate citations list — every entry corresponds to a specific claim in the answer.
- No "dump of everything retrieved" — if the doc_agent gathered 15 chunks but only 4 were cited, only 4 citations appear.
- `OctusCitation.cited_text` stores the full chunk text; `company_name` identifies which company the chunk belongs to.

**Citation display (`_render_citations()` in `app/streamlit_app.py`):**
- SimFin citations → grouped markdown tables by `(ticker, statement_type)` — easy to scan across periods
- Octus citations → one `st.expander` per citation; label shows `[N] DOC_SOURCE · DocumentType · date · company`; expanding reveals the full chunk text sent to the LLM

---

## SimFin: Batch vs Realtime

| | Batch (`simfin` Python package) | Realtime (SimFin v3 API) |
|--|-------------------------------|--------------------------|
| **Freshness** | Depends on simfin package update cadence | Fresh on each request |
| **Latency** | Negligible (local DuckDB query) | Network round-trip + rate limiting |
| **Complexity** | Simple: download once, query locally | Higher: cache management, rate limiting |
| **Recommended for** | Production chat UX, historical analysis | "Latest" queries, monitoring use cases |

**Realtime caching:** All v3 API responses are cached in SQLite keyed by `sha256(method + url + sorted_params)` with configurable TTL (default: 3600s). Mandatory to respect SimFin's rate limits (5 req/s free tier).

**References:**
- [SimFin API v3: getting started](https://simfin.readme.io/reference/getting-started-1)
- [SimFin API v3: rate limits](https://simfin.readme.io/reference/rate-limits)
- [SimFin API v3: statements](https://simfin.readme.io/reference/statements-1)

---

## SimFin Mapping: Fuzzy Matching + Manual Overrides

**Why rapidfuzz `WRatio`?**

`WRatio` is a weighted combination of `ratio`, `partial_ratio`, `token_sort_ratio`, and `token_set_ratio`. It handles common company name variations robustly — "Oracle Corp" vs "Oracle Corporation", "Charter Communications Inc." vs "Charter Communications". Tested against the 15 Octus companies and SimFin's ~8,000 US listings.

**Why 90% auto-promote threshold?**

Below 90, matches are ambiguous — "Charter Communications" might match "Charter Financial". At 90+, normalized names are very close and false positive rate is low. A lower threshold would pull in incorrect tickers, corrupting the financial data.

**Manual overrides (`_MANUAL_TICKER_OVERRIDES`):**

Companies like "Optimum Communications, Inc." score ~83% against SimFin's "Optimum" — below the 90% threshold but clearly correct. Rather than lowering the global threshold (which increases false positives across all companies), specific known mismatches are hardcoded in a dict constant in `src/simfin/mapping.py`. Current overrides: `Optimum Communications, Inc. → OPTU`, `Dave and Buster's → PLAY`.

**`sub_industry` field:**

`company_map.parquet` now includes the `sub_industry` column (e.g., "Passenger Airlines", "Leisure Facilities", "Cable & Satellite") sourced from `company_metadata.json`. This field is used by `extract_sector_companies()` in the router to expand sector queries — "airlines sector" → all companies with `sub_industry == "Passenger Airlines"` — without any LLM call or hardcoded sector mapping.

**Mapping modes (`settings.mapping_mode`):**
- `"confirmed"` — only rows with status == "confirmed" (high-confidence matches)
- `"auto_matched"` — only rows auto-promoted at 90%+ but not manually confirmed
- `"both"` — confirmed + auto_matched (default; excludes "needs_review")

---

## Streamlit Caching

`@st.cache_resource` is used to cache `load_context(settings)` across Streamlit reruns. This is the correct Streamlit primitive for singleton objects (database connections, Pinecone index handles, compiled LangGraph) that should be shared across all users and reruns in the same process.

`@st.cache_data` would be wrong here because it pickles and copies results, which is not appropriate for non-serializable objects like database connections or FAISS indexes.

**Live streaming:** The Streamlit app uses `ctx.graph.stream(initial, stream_mode="updates")` which yields per-node output dictionaries as each LangGraph node completes. This enables the UI to show agent progress in real time via `st.status()`.

**References:**
- [Streamlit caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [st.cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)

---

## Configuration / Environment Variables

| Variable | Required | Default | Purpose |
|----------|---------|---------|---------|
| `SIMFIN_API_KEY` | For SimFin | `""` | SimFin v3 API authentication |
| `SIMFIN_BASE_URL` | No | `https://backend.simfin.com` | SimFin API base URL |
| `PINECONE_API_KEY` | For Pinecone | `""` | Pinecone authentication |
| `PINECONE_INDEX_NAME` | No | `octus-financial` | Pinecone index name |
| `PINECONE_TRANSCRIPTS_NAMESPACE` | No | `transcripts` | Namespace for transcript vectors |
| `PINECONE_SEC_NAMESPACE` | No | `sec_filings` | Namespace for SEC filing vectors |
| `EMBEDDING_PROVIDER` | No | `sentence_transformers` | `sentence_transformers` / `openai` / `anthropic` |
| `EMBEDDING_MODEL` | No | `""` | Model name for the embedding provider |
| `EMBEDDING_DIM` | No | `768` | Embedding dimension |
| `TABLE_FORMAT` | No | `parquet` | `parquet` / `csv` |
| `MAPPING_MODE` | No | `both` | `confirmed` / `auto_matched` / `both` |
| `AUTO_PROMOTE_MATCHED` | No | `true` | Promote ≥90% fuzzy matches to confirmed |
| `RETRIEVER_ID` | No | `dense` | `dense` / `dense_mmr` |
| `MMR_LAMBDA` | No | `0.5` | MMR diversity/relevance trade-off |
| `TOP_K` | No | `5` | Number of chunks per retrieval call |
| `LLM_PROVIDER` | No | `none` | `none` / `anthropic` |
| `LLM_MODEL` | No | `claude-sonnet-4-6` | Model for synthesis agent |
| `ANTHROPIC_API_KEY` | For LLM synthesis | `""` | Anthropic API authentication |
| `DOC_AGENT_MODEL` | No | `claude-haiku-4-5-20251001` | Model for doc agent sparse-retry LLM calls |
| `HYDE` | No | `false` | Enable HyDE query expansion in router |
| `REQUIRE_SEC_HTML` | No | `true` | Fail ingestion if SEC HTML file missing |

All variables map to `Settings` fields in `src/app/settings.py` (Pydantic `BaseSettings`, loaded from `.env`).
