import json
import time
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_engine import AnswerEngine

def main():
    golden_path = PROJECT_ROOT / "eval" / "golden_set.json"
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    raw_output_path = results_dir / "raw_outputs.json"

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} benchmark questions from {golden_path}")
    engine = AnswerEngine()

    raw_outputs = []

    for item in tqdm(golden_set, desc="Evaluating Questions"):
        q_id = item["id"]
        question = item["question"]

        try:
            res = engine.answer_question(question)
            output_data = {
                "id": q_id,
                "question": question,
                "expected_behavior": item.get("expected_behavior"),
                "observed_behavior": res.behavior,
                "answer": res.answer,
                "citations": [c.model_dump() for c in res.citations],
                "retrieved_section_ids": res.retrieved_section_ids,
                "top_score": res.top_score,
                "hyde_triggered": res.hyde_triggered,
                "latency_ms": res.latency_ms,
                "canary_leaked": res.canary_leaked
            }
        except Exception as e:
            output_data = {
                "id": q_id,
                "question": question,
                "expected_behavior": item.get("expected_behavior"),
                "observed_behavior": "ERROR",
                "answer": f"ERROR: {str(e)}",
                "citations": [],
                "retrieved_section_ids": [],
                "top_score": 0.0,
                "hyde_triggered": False,
                "latency_ms": 0,
                "canary_leaked": False,
                "error": str(e)
            }

        raw_outputs.append(output_data)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved raw evaluation outputs to {raw_output_path}")

if __name__ == "__main__":
    main()
