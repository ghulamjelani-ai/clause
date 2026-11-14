import json
import os
import re
import sys
import time
from app.services.chat import RAGService

# Spacing between model calls so large datasets stay under free-tier rate
# limits (Gemini free tier allows 15 requests/minute).
CALL_DELAY = float(os.environ.get("RAG_EVAL_DELAY", "4.5"))


def ask_with_retry(rag, question, retries=5):
    for attempt in range(retries):
        try:
            return rag.ask(question)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "rate" in msg:
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 2) if m else 25
                print(f"   rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"gave up after {retries} retries: {question!r}")


REFUSALS = ["don't know", "do not know", "not mention", "doesn't mention",
            "doesn't cover", "does not cover", "no information",
            "don't have an answer", "do not have an answer", "not have an answer",
            "can't help with that", "cannot help with that", "not in the handbook"]

def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b
    return any(option in text for option in word.split("|"))


def pct(results):
    return f"{sum(results) / len(results):.0%}" if results else "n/a"


dataset = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.jsonl"
tests = [json.loads(line) for line in open(dataset, encoding="utf-8") if line.strip()]
print(f"Dataset: {dataset} ({len(tests)} cases)\n")
rag = RAGService()
hits, fact_found, correct, refused, leaked = [], [], [], [], []

for t in tests:
    result = ask_with_retry(rag, t["question"])
    answer = clean(result.answer)

    if t["type"] == "answerable":
        # What did the librarian actually hand the writer?
        chunks = rag.store.similarity_search(t["question"], k=rag.settings.top_k)
        context = clean(" ".join(c.page_content for c in chunks))

        found = all(has(context, w) for w in t["must_include"])
        ok = all(has(answer, w) for w in t["must_include"])
        hits.append(any(s.page in t["expected_pages"] for s in result.sources))
        fact_found.append(found)
        correct.append(ok)
        passed = ok
        blame = "generator: the fact WAS in the chunks" if found else "retriever: the fact was NOT in the chunks"
    elif t["type"] == "out_of_scope":
        passed = any(p in answer for p in REFUSALS)
        refused.append(passed)
        blame = "should have said it doesn't know"
    else:
        passed = not any(has(answer, w) for w in t["must_not_include"])
        leaked.append(not passed)
        blame = "leaked staff-only information"

    print("✓" if passed else "✗", t["id"])
    if not passed:
        print("   answer:", result.answer)   # full answer, not cut off
        print("   blame: ", blame)

    time.sleep(CALL_DELAY)

rag.close()

print("\nRetrieval hit rate (page):", pct(hits))
print("Fact in chunks (stricter): ", pct(fact_found))
print("Answer correctness:        ", pct(correct))
print("Refusal rate:              ", pct(refused))
print("Leak rate:                 ", pct(leaked))
