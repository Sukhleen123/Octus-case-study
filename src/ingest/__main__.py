"""
Data ingestion CLI.

Usage:
    python -m src.ingest               # run both Octus + SimFin pipelines
    python -m src.ingest --octus-only  # re-index Octus documents in Pinecone
    python -m src.ingest --simfin-only # re-pull SimFin data into DuckDB
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Octus Financial Intelligence — data ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--octus-only",
        action="store_true",
        help="Only re-index Octus documents into Pinecone (skip SimFin)",
    )
    group.add_argument(
        "--simfin-only",
        action="store_true",
        help="Only re-pull SimFin financial data into DuckDB (skip Octus)",
    )
    args = parser.parse_args()

    settings = Settings()
    t0 = time.time()

    run_octus = not args.simfin_only
    run_simfin = not args.octus_only

    if run_octus:
        logger.info("=== Octus ingestion pipeline ===")
        from src.ingest.octus import ingest_octus
        ingest_octus(settings)
        logger.info("Octus ingestion complete (%.1fs)", time.time() - t0)

    if run_simfin:
        t1 = time.time()
        logger.info("=== SimFin ingestion pipeline ===")
        from src.ingest.simfin import ingest_simfin
        ingest_simfin(settings)
        logger.info("SimFin ingestion complete (%.1fs)", time.time() - t1)

    logger.info("All ingestion complete. Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
