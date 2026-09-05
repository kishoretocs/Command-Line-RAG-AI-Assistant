import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.observability.db import init_database
from src.ingestion.pipeline import IngestionPipeline

def main():
    print("Initializing SQLite Observability Database...")
    init_database()
    
    print("Starting Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    summary = pipeline.run()
    print("Ingestion successfully finished!")

if __name__ == "__main__":
    main()
