from pipeline import ingest, query

# ingest
print("Ingesting...")
result = ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
print(f"Ingested {result['ingested']} chunks\n")

# query
print("Querying...")
result = query("How does RAG reduce hallucinations?")

print(f"Answer:\n{result['answer']}\n")
print(f"Sources: {result['sources']}")