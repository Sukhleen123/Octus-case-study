# Architecture

## System Overview

```mermaid
graph TB
    subgraph "Frontends"
        ST[Streamlit App\nstreamlit run apps/streamlit_app.py]
        CL[Chainlit App\nchainlit run apps/chainlit_app.py]
    end

    subgraph "Bootstrap Layer"
        BS[bootstrap.ensure_ready\nidempotent + file-locked]
        FP[fingerprints.py\nSHA-256 of raw files + settings]
        LK[locks.py\nfilelock - prevents parallel builds]
    end

    subgraph "Settings"
        SET[src/app/settings.py\nPydantic BaseSettings\nconfigs/app.yaml + env vars]
    end

    subgraph "AppContext"
        VS[vectorstore\nPinecone or FAISS]
        RET[retriever\ndense or dense_mmr]
        SFC[simfin_client\nv3 realtime or None]
        ORC[orchestrator]
    end

    subgraph "Octus Ingestion"
        ING[ingest.py\nrun_ingestion]
        H2T[html_to_text.py\nBeautifulSoup/lxml]
        DD[dedupe.py\nSEC deduplication]
        NRM[normalize.py\ndate parsing]
    end

    subgraph "Chunking"
        HC[HeadingChunker\nchunks_heading.parquet]
        TC[TokenChunker\nchunks_token.parquet]
    end

    subgraph "Embeddings"
        EMB[get_embedder factory\nmock default / openai]
        ME[MockEmbedder\ndeterministic, no API key]
    end

    subgraph "Vectorstore"
        PC[PineconeStore\nserver-side metadata filter]
        FA[FAISSStore\nIndexFlatIP + SQLite metadata]
        MS[MetadataStore SQLite\nPython-side filter for FAISS]
    end

    subgraph "Agents"
        RTR[Router\nkeyword-based routing]
        DA[DocAgent\nretrieve Octus chunks]
        SA[SimFinAgent\nretrieve financial metrics]
        SYN[SynthesisAgent\nmerge + final answer]
        ORC2[Orchestrator\ncoordinates agents]
    end

    subgraph "SimFin"
        MAP[mapping.py\nrapidfuzz Octus->SimFin]
        BAT[batch_ingest.py\nsimfin Python pkg]
        RTC[realtime_client_v3.py\nhttpx + Bearer auth]
        CHC[RealtimeCache SQLite\nTTL-based caching]
    end

    subgraph "Raw Data"
        CM[company_metadata.json\n15 companies]
        TR[transcripts.json\n105 transcripts]
        SM[sec_filings_metadata.json\n137 filings]
        SH[sec_html/\n137 HTML files]
    end

    subgraph "Processed Artifacts"
        DOC[documents.parquet]
        CHK[chunks_heading/token.parquet]
        CMP[company_map.parquet]
        MAN[bootstrap_manifest.json]
    end

    ST --> SET
    CL --> SET
    ST --> BS
    CL --> BS
    SET --> BS
    BS --> FP
    BS --> LK
    BS --> ING
    BS --> HC
    BS --> TC
    BS --> EMB
    BS --> VS
    BS --> RET
    BS --> ORC

    ING --> CM
    ING --> TR
    ING --> SM
    ING --> SH
    ING --> H2T
    ING --> DD
    ING --> NRM
    ING --> DOC

    HC --> DOC
    TC --> DOC
    HC --> CHK
    TC --> CHK

    EMB --> ME

    VS --> PC
    VS --> FA
    FA --> MS

    ORC --> RTR
    ORC --> DA
    ORC --> SA
    ORC --> SYN
    DA --> RET
    RET --> VS

    SA --> SFC
    SFC --> RTC
    RTC --> CHC

    MAP --> CM
    BAT --> MAP
```

## Data Flow

```
Raw JSON/HTML → [Ingestion] → documents.parquet
                           → ingest_report.parquet

documents.parquet → [Chunking] → chunks_heading.parquet
                              → chunks_token.parquet

chunks_*.parquet → [Embedding] → [Vectorstore: Pinecone or FAISS]

User Query → [Router] → DocAgent → [Retriever] → [Vectorstore]
                     → SimFinAgent → [SimFin API / DuckDB]
                     → SynthesisAgent → SynthesisResult
                                        ├── final_answer_text
                                        ├── citations[]
                                        └── trace_events[]
```

## Storage Architecture

| Store | Purpose | Technology |
|-------|---------|-----------|
| `documents.parquet` | Normalized Octus documents | Apache Parquet (pyarrow) |
| `chunks_*.parquet` | Chunked text for retrieval | Apache Parquet (pyarrow) |
| `simfin.duckdb` | SimFin financial statements | DuckDB (embedded SQL) |
| `company_map.parquet` | Octus→SimFin ticker mapping | Apache Parquet |
| Pinecone index | Vector embeddings (primary) | Pinecone serverless |
| FAISS index | Vector embeddings (fallback) | FAISS IndexFlatIP |
| `metadata.sqlite` | FAISS chunk metadata | SQLite |
| `embedding_cache.sqlite` | Embedding cache | SQLite |
| `simfin_realtime_cache.sqlite` | SimFin API response cache | SQLite |
| `.bootstrap.lock` | Build mutex | filelock |
| `bootstrap_manifest.json` | Fingerprint + build metadata | JSON |
