# test this in a python shell or quick script
from ingestion.web import ingest_url
from ingestion.local import ingest_file

# test web
chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
print(f"Got {len(chunks)} chunks")
print(chunks[0])

# test local — create a test file first
with open("test.txt", "w") as f:
    f.write("This is a test document. " * 100)

chunks = ingest_file("test.txt")
print(f"Got {len(chunks)} chunks from local file")