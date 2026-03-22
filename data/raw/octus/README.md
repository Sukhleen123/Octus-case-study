# Overview

These files contain data for the following companies:

- Charter Communications
- Cable One, Inc.
- Optimum Communications, Inc.
- Commscope Holding Company Inc.
- Viasat
- Ciena Corp
- Dave & Buster's
- Six Flags Entertainment Corporation
- Life Time Fitness
- Delta Air Lines
- American Airlines
- JetBlue
- Lucid Group, Inc.
- Rivian Automotive Inc.
- Ford

## Data Coverage

Documents span **January 2024 – March 11, 2026** and include the following types:

- **SEC Filings:** 10-K, 10-Q
- **Transcripts**

---

# company_metadata.json

A list of company records used to map Octus entities and their associated identifiers.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `octus_company_id` | string | The primary company identifier in the Octus platform |
| `company_name` | string | Legal or commonly used company name |
| `sub_industry` | string | Industry classification for the company |
| `company_ids` | string | Comma-separated list of all related Octus entity IDs associated with this company (subsidiaries, tranches, instruments, etc.) |

---

# transcript.json

A list of transcript documents associated with Octus company entities.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Unique identifier for the transcript document |
| `source_type` | string | Source category of the document (e.g. Transcript) |
| `document_type` | string | Type of document (e.g. Transcript) |
| `document_date` | string | Timestamp of when the document was published or created |
| `company_id` | integer | Octus entity ID linking the document to a specific company record |
| `body` | string | Full HTML content of the document |

---

# sec_filings.json

A list of SEC filing documents associated with Octus company entities.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Unique identifier for the SEC filing document |
| `source_type` | string | Source category of the document (e.g. SEC Filing) |
| `document_type` | string | Type of SEC filing (e.g. 10-Q, 10-K) |
| `document_date` | string | Timestamp of when the document was published or created |
| `company_id` | array | List of Octus entity IDs linking the document to company records |