import json
from pathlib import Path
from typing import Dict, List, Any
from src.config import MANIFEST_PATH, BASELINE_DATE

class ManifestLoader:
    def __init__(self, manifest_path: Path = MANIFEST_PATH, baseline_date: str = BASELINE_DATE):
        self.manifest_path = manifest_path
        self.baseline_date = baseline_date

    def load(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads manifest and maps document_id -> metadata dictionary.
        Determines `is_superseded` flag based on baseline date.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data.get("documents", [])
        manifest_map: Dict[str, Dict[str, Any]] = {}

        for doc in docs:
            doc_id = doc["document_id"]
            manifest_map[doc_id] = {
                "document_id": doc_id,
                "file": doc["file"],
                "title": doc["title"],
                "version": doc["version"],
                "effective_date": doc["effective_date"],
                "owner": doc["owner"],
                "classification": doc["classification"],
                "supersedes": doc.get("supersedes"),
                "is_superseded": False,
                "superseded_by": None
            }

        # Determine supersession
        # If doc A supersedes doc B, and doc A is effective on/before baseline_date, doc B is superseded.
        for doc_id, meta in manifest_map.items():
            superseded_target = meta.get("supersedes")
            effective_date = meta.get("effective_date", "9999-99-99")
            
            if superseded_target and effective_date <= self.baseline_date:
                # Target could be "SALES-PL-2025 v1.0" or "HR-POL-002 v3.6"
                target_id = superseded_target.split()[0]
                if target_id in manifest_map:
                    manifest_map[target_id]["is_superseded"] = True
                    manifest_map[target_id]["superseded_by"] = doc_id

        return manifest_map

if __name__ == "__main__":
    loader = ManifestLoader()
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    for did, info in docs.items():
        print(f"{did} (v{info['version']}) -> Superseded: {info['is_superseded']} (by {info['superseded_by']})")
