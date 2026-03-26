# Architecture

## System Overview

```mermaid
graph TB
    subgraph "Frontends"
        ST[Streamlit App\napp/streamlit_app.py]
    end

    subgraph "App Bootstrap"
        LC[load_context\nsrc/app/loader.py]
        SET[Settings\nsrc/app/settings.py\nPydantic BaseSettings + .env]
    end

    subgraph "Octus Ingestion"
        OI[ingest_octus\nsrc/ingest/octus.py]
        RI[run_ingestion\nsrc/octus/ingest.py]
        H2T[html_to_text.py\nBeautifulSoup → plain text]
        DD[dedupe.py\n(company_ids, doc_type, date)]
        NRM[normalize.py\ndate + company_id parsing]
    end

    subgraph "Chunking"
        RC[RecursiveChunker\nsrc/chunking/recursive_chunker.py\ntiktoken cl100k_base 512 tokens]
        SC[SECSectionChunker\nsrc/chunking/sec_section_chunker.py\nTOC-anchor DOM split → regex fallback]
    end

    subgraph "Embeddings"
        EMB[get_embedder factory\nsentence_transformers / openai / anthropic]
    end

    subgraph "Vectorstore — Pinecone"
        PCT[PineconeStore\nnamespace: transcripts]
        PCS[PineconeStore\nnamespace: sec_filings]
        TS[MetadataStore SQLite\ntranscripts_text_store.sqlite]
        SS[MetadataStore SQLite\nsec_filings_text_store.sqlite]
    end

    subgraph "Retrieval"
        MSR[MultiStoreRetriever\nsrc/retrieval/multi_store.py]
        TR[DenseRetriever / DenseMMRRetriever\nfor transcripts]
        SR[DenseRetriever / DenseMMRRetriever\nfor sec_filings]
    end

    subgraph "SimFin Ingestion"
        SI[ingest_simfin\nsrc/ingest/simfin.py]
        MAP[mapping.py\nrapidfuzz WRatio ≥90 auto-promote\nsimfin pkg → company name list]
        RTC[SimFinV3Client\nsrc/simfin/realtime_client_v3.py\nhttpx + Bearer auth]
        CHC[RealtimeCache\nSQLite TTL=3600s]
        DDB[(DuckDB\nsimfin.duckdb\n6 tables)]
    end

    subgraph "LangGraph Agent Graph"
        RTR[router_node\nkeyword routing + company extraction]
        DA[doc_agent_node\ndeterministic retrieval + sparse retry]
        SA[simfin_agent_node\nDuckDB queries]
        SYN[synthesize_node\nClaude synthesis + citation numbering]
    end

    subgraph "Citations"
        OC[OctusCitation\ndocument_id, chunk_id, cited_text, company_name]
        SC2[SimFinCitation\nticker, fiscal_period, metric_name/value]
        FMT[formatter.py\nSimFin: markdown tables\nOctus: expandable expanders]
    end

    subgraph "Raw Data"
        CM[company_metadata.json\n15 companies]
        TR2[transcripts.json\n105 transcripts]
        SM[sec_filings_metadata.json\n137 filings]
        SH[sec_html/\n137 HTML files]
    end

    subgraph "Processed Artifacts"
        DOC[documents.parquet]
        CMP[company_map.parquet\nOctus→SimFin ticker + sub_industry]
    end

    ST --> SET
    ST --> LC
    LC --> SET
    LC --> EMB
    LC --> PCT
    LC --> PCS
    PCT --> TS
    PCS --> SS
    LC --> TR
    LC --> SR
    TR --> PCT
    SR --> PCS
    LC --> MSR
    MSR --> TR
    MSR --> SR
    LC --> DDB

    OI --> RI
    RI --> CM
    RI --> TR2
    RI --> SM
    RI --> SH
    RI --> H2T
    RI --> DD
    RI --> NRM
    RI --> DOC
    OI --> RC
    OI --> SC
    RC --> EMB
    SC --> EMB
    RC --> PCT
    SC --> PCS

    SI --> MAP
    MAP --> CM
    SI --> RTC
    RTC --> CHC
    SI --> DDB

    LC --> RTR
    RTR --> DA
    RTR --> SA
    DA --> MSR
    SA --> DDB
    DA --> SYN
    SA --> SYN
    SYN --> OC
    SYN --> SC2
    OC --> FMT
    SC2 --> FMT
    FMT --> ST

    MAP --> CMP
```

---

## Ingestion Pipeline

### Octus Documents

```
Raw JSON/HTML
  → src/octus/ingest.py       — parse JSON, normalize dates, extract company IDs
  → src/octus/dedupe.py       — deduplicate SEC filings on (sorted company_ids, doc_type, date)
  → src/octus/html_to_text.py — BeautifulSoup HTML → clean plain text
  → documents.parquet         — columns: doc_source, document_id, document_type, document_date,
                                         octus_company_id, company_name, company_ids (JSON),
                                         raw_path, cleaned_text
```

**Transcript vs SEC Filing handling:**

| | Transcripts | SEC Filings |
|--|-------------|-------------|
| `company_ids` | Single int | List of ints |
| Text source | Body embedded inline in JSON | Separate HTML files in `sec_html/` |
| Deduplication | None | Yes — key: `(sorted company_ids, doc_type, date)` |
| Chunker | `RecursiveChunker` | `SECSectionChunker` |
| Pinecone namespace | `transcripts` | `sec_filings` |

### Chunking Strategies

**RecursiveChunker** (`src/chunking/recursive_chunker.py`)

- Tokenizer: `tiktoken cl100k_base` (consistent with OpenAI embedding models)
- Algorithm: Split on `\n\n` paragraph boundaries → merge small paragraphs until `target_size=512` tokens → sub-split oversized paragraphs on sentence boundaries with `overlap=64` tokens
- Heading detection: Markdown `##` headers OR ALL-CAPS lines
- Used for: **transcripts** (free-flowing prose, Q&A format — no reliable structural headers)

**SECSectionChunker** (`src/chunking/sec_section_chunker.py`)

Two-path parsing strategy:

| Path | Trigger | How |
|------|---------|-----|
| **TOC-anchor (primary)** | Document has ≥3 `<a href="#...">` TOC links pointing to semantic anchors (`id="item_1_business"`, etc.) | Walk DOM; flush new section at each anchor element; titles come directly from TOC text |
| **Regex-text (fallback)** | Wdesk/generic-ID filings (no semantic anchor links) | Extract plain text, split on `PART`/`ITEM` boundary patterns; titles resolved via static dict |

- HTML table preservation: Converts `<table>` tags to markdown in both paths
- Subsplit: Sections exceeding `max_section_tokens=2048` are further split with token windows + overlap
- XBRL removal: Strips XBRL headers and hidden elements before parsing
- Used for: **SEC filings** (10-K, 10-Q) — TOC path preserves company-specific section names; regex path handles sparse/Wdesk filings

**Chunk metadata schema** (both chunkers):

```
chunk_id (UUID), document_id, doc_source, document_type, document_date,
octus_company_id, company_name, company_ids (JSON), section_title,
chunk_index, char_start, char_end, chunker_id, text
```

### SimFin Financial Data

```
Octus company_metadata.json
  → src/simfin/mapping.py      — fuzzy match (rapidfuzz WRatio) Octus names → SimFin tickers
                                  uses simfin Python package (sf.load_companies) for company name list only
                                  score ≥ 90 → auto-promoted to "confirmed"
                                  score < 90 → "needs_review" (requires manual override)
                                  manual overrides: Optimum Communications → OPTU, Dave & Buster's → PLAY
  → company_map.parquet

SimFin v3 API (per-ticker, real-time)
  → src/simfin/realtime_client_v3.py — GET /api/v3/companies/statements/verbose
                                        ?ticker=X&statements=PL,BS,CF
                                        httpx + Bearer auth, SQLite TTL cache (3600s)
  → src/ingest/simfin.py             — normalizes verbose response, splits annual vs. quarterly
  → DuckDB: 6 tables
      income_annual,    income_quarterly
      balance_annual,   balance_quarterly
      cashflow_annual,  cashflow_quarterly
```

**DuckDB annual vs quarterly split logic:**

| Statement | Annual filter | Quarterly filter |
|-----------|--------------|-----------------|
| Income / Cashflow | `Fiscal Period == "FY"` | all other periods |
| Balance Sheet | `Fiscal Period == "Q4"` (year-end snapshot) | Q1, Q2, Q3 |

---

## Agent Architecture (LangGraph)

```
User Query
  → AgentState initialization (query, messages)

  → router_node  (6 inference steps, all run before any agent executes)
      1. route_decision(): keyword scoring — SIMFIN_KEYWORDS vs OCTUS_KEYWORDS
      2. company extraction: substring + rapidfuzz (70% threshold)
         OR sector extraction: SECTOR_TRIGGERS regex → extract_sector_companies()
         looks up companies by sub_industry (exact match, then fuzzy ≥75%)
         sets sector_mode=True
      3. resolve_tickers(): company_map DataFrame lookup → SimFin tickers
      4. infer_doc_filters(): detects doc_type ("10-K" / "10-Q" / "Transcript") and doc_source
      5. extract_doc_temporal_filters(): "last N quarters/years" → doc_date_from, doc_date_to (ISO)
         extract_simfin_temporal_filters(): → simfin_date_from (year), simfin_max_quarterly_periods,
                                              simfin_max_annual_periods
      6. select_simfin_tables(): narrows which of 6 DuckDB tables to query
         maybe_expand_query(): HyDE — Claude rewrites query (only if hyde=True + HYDE_TRIGGERS match)
      - All-companies mode: ALL_COMPANIES_TRIGGERS regex detected
      - Sets on state: route, companies, tickers, doc_filters, doc_query, all_companies_mode,
                       sector_mode, doc_date_from, doc_date_to, simfin_date_from,
                       simfin_max_quarterly_periods, simfin_max_annual_periods, simfin_tables
      → emits: agent_start, agent_end trace events

  → [Conditional routing on `route`]
      "octus"  → doc_agent_node only
      "simfin" → simfin_agent_node only
      "both"   → doc_agent_node AND simfin_agent_node in PARALLEL

  → doc_agent_node  (if route includes "octus")
      Retrieval strategy is fully determined by router state — no tool loop:
      - If all_companies_mode=True OR (sector_mode=True AND >1 company):
          _retrieve_for_each_company() — parallel per-company (ThreadPoolExecutor)
          up to 4 chunks per company; ensures equal representation across companies
      - Otherwise:
          _retrieve_documents() — single targeted retrieval using doc_filters + date range
      Sparse result retry (if LLM available AND <RETRY_THRESHOLD=3 chunks returned):
          _retry_with_llm() — Claude generates 3 alternative query rephrasings
          Re-runs retrieval with each; if still sparse, drops company filter and broadens
      MAX_FINAL=15 chunks forwarded to synthesis
      → emits: agent_start, tool_call_start/end, retrieval_results,
               citations_emitted, agent_end

  → simfin_agent_node  (if route includes "simfin")
      - Queries only state.simfin_tables (pre-selected subset of 6 tables by router)
      - Applies fiscal_year_from filter if state.simfin_date_from is set
      - Limits rows: simfin_max_quarterly_periods (quarterly), simfin_max_annual_periods (annual)
      - Key metrics: Revenue, Net Income, Operating Income, Gross Profit,
        D&A, Total Assets, Total Equity, Net Cash from Operations
      - Builds up to 16 SimFinCitation objects per ticker (4 most recent periods × metrics)
      → emits: agent_start, tool_call_start/end (tool="duckdb"),
               simfin_results, citations_emitted, agent_end

  → synthesize_node
      - Assembles numbered citation list (SimFin citations first, then Octus)
      - Prompts Claude with citation rules:
          financial data → state metric + value + [N]
          document excerpts → quote + [N]
          long quotes → blockquote with [N] at end
      - Regex extracts all [N] refs used in the answer
      - Citations NOT referenced in the answer are dropped (ref_number stays 0)
      → SynthesisResult(final_answer_text, citations[], trace_events[])
```

**AgentState** (Pydantic BaseModel with LangGraph message reducers):

| Field | Set by | Reducer |
|-------|--------|---------|
| `query`, `retrieval_kwargs` | caller | — |
| `route`, `companies`, `tickers`, `doc_filters`, `doc_query` | router | — |
| `all_companies_mode`, `sector_mode` | router | — |
| `doc_date_from`, `doc_date_to` (ISO strings) | router | — |
| `simfin_date_from` (year str), `simfin_max_quarterly_periods`, `simfin_max_annual_periods` | router | — |
| `simfin_tables` (list of table names) | router | — |
| `doc_result` (chunks, citations, events) | doc_agent | — |
| `simfin_result` (DataFrames, citations, events) | simfin_agent | — |
| `synthesis_result` | synthesize | — |
| `trace_events` | all nodes | `operator.add` |
| `messages` | all nodes | `add_messages` |

---

## Retrieval Architecture

```
MultiStoreRetriever  (src/retrieval/multi_store.py)
  ├── transcript_retriever  →  PineconeStore(namespace="transcripts")
  └── sec_retriever         →  PineconeStore(namespace="sec_filings")

Routing logic (_route()):
  doc_source == "transcript"  → transcript_retriever only
  doc_source == "sec_filing"  → sec_retriever only
  (no doc_source filter)      → SEC-first if query contains SECTION_QUERY_KEYWORDS
                                  (risk factor, MD&A, balance sheet, item 7, ...)
                                else transcript-first
                              → results from both stores merged, deduplicated by chunk_id,
                                sorted by _score descending

Post-retrieval:
  - Boilerplate chunks (Safe Harbor, Cautionary Note section_title matches) demoted to end
  - Date range filtering applied Python-side after store returns results
  - Final list: clean_chunks[:top_k] + boilerplate (only if needed to fill top_k)
```

**Retriever variants** (configured via `settings.retriever_id`):

| Retriever | Strategy | When to use |
|-----------|---------|-------------|
| `dense` | Embed query → ANN search → metadata filter → top-k | Fast, deterministic |

---

## Vectorstore Architecture

**PineconeStore + local SQLite MetadataStore**

Pinecone enforces a 40 KB metadata limit per vector. Financial document chunks (especially SEC sections with embedded tables) regularly exceed this. Workaround: chunk text is stored in a local SQLite `MetadataStore` keyed by `chunk_id`; Pinecone only receives structured metadata fields.

```
Upsert flow:
  chunk text  →  MetadataStore (SQLite, indexed by chunk_id)
  metadata (company_name, doc_source, document_type, document_date, chunk_id, ...)
              →  Pinecone (server-side ANN + filter)

Query flow:
  query vector → Pinecone(filter + ANN search) → [(chunk_id, score), ...]
  chunk_ids    → MetadataStore.get_batch_by_chunk_ids() → {chunk_id: {text, metadata}}
  rejoin       → [{...metadata, text, _score}, ...]
```

**Pinecone filter syntax** (auto-converted from simple dict in `_to_pinecone_filter()`):

```python
{"company_name": "Oracle"}          → {"company_name": {"$eq": "Oracle"}}
{"doc_source": ["sec_filing"]}      → {"doc_source": {"$in": ["sec_filing"]}}
```
---

## Citation System

```
doc_agent    → OctusCitation per selected chunk:
               document_id, doc_source, document_type, document_date,
               chunk_id, cited_text (full chunk text), company_name

simfin_agent → SimFinCitation per metric per period:
               ticker, fiscal_year, fiscal_period (Q1/Q2/Q3/Q4/FY),
               statement_type, metric_name, metric_value, metric_unit

synthesize_node:
  1. Assigns [1], [2], ... ref numbers to all citations in prompt
  2. Claude answer uses inline [N] markers
  3. Regex \[(\d+)\] extracts used ref numbers
  4. Citations with ref_number == 0 (not cited) are dropped from output

Streamlit display (app/streamlit_app.py — _render_citations()):
  SimFin  → grouped markdown tables: (ticker × statement_type) per table
            columns: Ref | Period | metric1 | metric2 | ...
  Octus   → st.expander per citation, label: [N] DOC_SOURCE · DocumentType · date · company
            expander body: full cited_text chunk
```

---

## Data Flow Summary

```
Raw JSON + HTML
  → [Octus Ingestion]   → documents.parquet

documents.parquet
  → [RecursiveChunker]  → embed → Pinecone namespace: transcripts
                                   + transcripts_text_store.sqlite
  → [SECSectionChunker] → embed → Pinecone namespace: sec_filings
                                   + sec_filings_text_store.sqlite

company_metadata.json
  → [Fuzzy Mapping]     → company_map.parquet
  → [SimFin v3 API]     → simfin.duckdb (6 tables)

User Query
  → [LangGraph graph]
      router_node → doc_agent_node ─┐
                  → simfin_agent_node ─┤ (parallel)
                                    └─→ synthesize_node
  → SynthesisResult
      ├── final_answer_text  (markdown, inline [N] citations)
      ├── citations[]        (OctusCitation + SimFinCitation, ref_number assigned)
      └── trace_events[]     (agent_start/end, tool_call_start/end, retrieval_results, ...)
  → Streamlit: answer rendered as markdown, citations in expander, trace in expander
```

---

## Storage Architecture

| Store | Purpose | Technology |
|-------|---------|------------|
| `documents.parquet` | Normalized Octus documents | Apache Parquet (pyarrow) |
| `company_map.parquet` | Octus → SimFin ticker mapping | Apache Parquet |
| `simfin.duckdb` | SimFin financial statements (6 tables) | DuckDB (embedded SQL) |
| `income/balance/cashflow_annual/quarterly.parquet` | SimFin raw parquet export | Apache Parquet |
| Pinecone index — `transcripts` namespace | Transcript vector embeddings | Pinecone serverless (cosine) |
| Pinecone index — `sec_filings` namespace | SEC filing vector embeddings | Pinecone serverless (cosine) |
| `transcripts_text_store.sqlite` | Chunk texts for transcripts (40 KB workaround) | SQLite |
| `sec_filings_text_store.sqlite` | Chunk texts for SEC filings (40 KB workaround) | SQLite |
| `simfin_realtime_cache.sqlite` | SimFin v3 API response cache (TTL=3600s) | SQLite |
