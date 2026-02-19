import os
import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure project root is on sys.path so imports work when running
# this file as a script (python src/ingestion.py) or as a module
# (python -m src.ingestion)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import TravelDataLoader
from src.search_engine import TravelSearchEngine
from src.config import Config

import mlflow


def ingest_travel_documents():
    """
    Ingests travel documents into Azure AI Search vector store.
    """
    print("\n🚀 Starting Travel Document Ingestion")
    print("=" * 70)

    # Initialize components
    loader = TravelDataLoader()

    try:
        engine = TravelSearchEngine()
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return

    # -----------------------------
    # MLflow (fail-safe)
    # -----------------------------
    mlflow_active = False
    if Config.MLFLOW_TRACKING_URI:
        try:
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            mlflow.start_run(run_name="document_ingestion")
            mlflow_active = True
        except Exception as e:
            print(f"⚠️  MLflow disabled: {e}")

    try:
        # Load documents
        documents = loader.load_all_travel_documents()

        if not documents:
            print("\n⚠️  No documents found in data directory")
            print("\nExpected structure:")
            print("  data/")
            print("    ├── *.pdf   (policies, FAQs, rules)")
            print("    └── *.csv   (routes or tabular data)")
            return

        # Split into chunks
        chunks = loader.split_documents(documents)

        print(f"\n📊 Ingestion Summary:")
        print(f"   Total chunks to index: {len(chunks)}")

        if mlflow_active:
            mlflow.log_param("total_chunks", len(chunks))
            mlflow.log_param("chunk_size", loader.text_splitter._chunk_size)
            mlflow.log_param("chunk_overlap", loader.text_splitter._chunk_overlap)

        # Batch ingestion
        print("\n📥 Indexing documents to Azure AI Search...")
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        ingested_count = 0
        failed_count = 0

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Indexing",
            total=total_batches
        ):
            batch = chunks[i:i + batch_size]

            try:
                engine.vector_store.add_documents(batch)
                ingested_count += len(batch)
                time.sleep(0.5)  # avoid rate limits

            except Exception as e:
                print(f"\n❌ Error indexing batch {i // batch_size + 1}: {e}")
                failed_count += len(batch)

        print(f"\n✅ Ingestion Complete!")
        print(f"   Successfully indexed: {ingested_count} chunks")
        if failed_count > 0:
            print(f"   Failed: {failed_count} chunks")

        if mlflow_active:
            mlflow.log_metric("ingested_count", ingested_count)
            mlflow.log_metric("failed_count", failed_count)

        # Verification
        print("\n🔍 Verifying index...")
        test_query = "What are the baggage rules?"
        results, _ = engine.search_by_text(test_query, k=3)

        if results:
            print("✅ Index verification successful!")
            print(f"   Test query: '{test_query}'")
            print(f"   Retrieved: {len(results)} documents")
        else:
            print("⚠️  Warning: Test query returned no results")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")

    finally:
        if mlflow_active:
            mlflow.end_run()

    print("\n" + "=" * 70)
    print("🎉 Ingestion pipeline completed!\n")


if __name__ == "__main__":
    ingest_travel_documents()
