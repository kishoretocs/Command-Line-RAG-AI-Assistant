import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_engine import AnswerEngine

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ").strip()

    if not query:
        print("Empty question. Exiting.")
        return

    print(f"\nProcessing query: '{query}'...")
    engine = AnswerEngine()
    result = engine.answer_question(query)

    print("\n" + "="*50)
    print(f"Query ID:        {result.query_id}")
    print(f"Behavior:        {result.behavior}")
    print(f"Top Score:       {result.top_score:.4f}")
    print(f"HyDE Fallback:   {result.hyde_triggered}")
    print(f"Latency:         {result.latency_ms} ms")
    print(f"Canary Leaked:   {result.canary_leaked}")
    print(f"Citations:       {[c.document_id for c in result.citations]}")
    print("="*50)
    print(f"\nAnswer:\n{result.answer}\n")

if __name__ == "__main__":
    main()
