import json
import requests

API_URL = "http://127.0.0.1:8000/ask"
OUTPUT_FILE = "evaluation_output.txt"

with open("questions.json", "r") as f:
    questions = json.load(f)

results = []

for q in questions:
    for mode in ["baseline", "hybrid"]:
        payload = {"q": q["q"], "k": q["k"], "mode": mode}
        print(f"Processing: {q['q']} | Mode: {mode} ...")
        res = requests.post(API_URL, json=payload)
        data = res.json()

        entry = {
            "question": q["q"],
            "mode": mode,
            "answer": "[ABSTAINED]" if data.get("abstained", False) else data.get("answer"),
            "top_doc": data["contexts"][0]["doc"] if data["contexts"] else None,
            "contexts": data.get("contexts", []),
            "reranker_used": data.get("reranker_used", None),
            "abstained": data.get("abstained", False)
        }
        results.append(entry)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"Q: {r['question']}  |  Mode: {r['mode']}\n")
        f.write(f"Answer: {r['answer']}\n")
        f.write(f"Top Doc: {r['top_doc']}\n")
        f.write(f"Abstained: {r['abstained']}\n")
        f.write("Contexts:\n")
        for c in r["contexts"]:
            f.write(f"  - Rank: {c['rank']}, Doc: {c['doc']}\n")
            f.write(f"    Score: {c['score']:.4f}\n")
            f.write(f"    URL: {c['url']}\n")
            f.write(f"    Text: {c['text'][:200]}...\n")
        f.write("-" * 60 + "\n\n")

print(f"\n✅ Evaluation complete. Results saved to {OUTPUT_FILE}")
