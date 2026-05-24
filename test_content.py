# test_content.py
from pipeline import ingest
import time

# Four topic areas — varied enough to test real-world KB breadth
SOURCES = [
    # AI / ML
    {
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "topic": "AI — RAG"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
        "topic": "AI — LLMs"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Vector_database",
        "topic": "AI — Vector DBs"
    },
    # Software engineering
    {
        "url": "https://en.wikipedia.org/wiki/PostgreSQL",
        "topic": "Engineering — PostgreSQL"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Docker_(software)",
        "topic": "Engineering — Docker"
    },
    # Business / startups
    {
        "url": "https://paulgraham.com/greatwork.html",
        "topic": "Business — Paul Graham (Great Work)"
    },
    {
        "url": "https://paulgraham.com/startupideas.html",
        "topic": "Business — Paul Graham (Startup Ideas)"
    },
    # Science / general knowledge
    {
        "url": "https://en.wikipedia.org/wiki/Climate_change",
        "topic": "Science — Climate Change"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Quantum_computing",
        "topic": "Science — Quantum Computing"
    },
]

print("Ingesting test content...\n")
total_chunks = 0

for source in SOURCES:
    print(f"→ {source['topic']}")
    result = ingest(source["url"])
    if result.get("skipped"):
        print(f"  already ingested — skipping")
    else:
        print(f"  {result['ingested']} chunks ingested")
        total_chunks += result["ingested"]
    time.sleep(1)

print(f"\nTotal new chunks: {total_chunks}")
print("KB ready for testing.")