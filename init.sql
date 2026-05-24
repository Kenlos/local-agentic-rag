-- init.sql
-- This runs automatically when the PostgreSQL container starts.
-- Creates the vector extension, documents table, and HNSW index.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS query_cache (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    route       TEXT NOT NULL,
    sources     JSONB DEFAULT '[]',
    quality     FLOAT DEFAULT 0.0,
    created_at  TIMESTAMP DEFAULT NOW(),
    hits        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_memory (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    tool        TEXT NOT NULL,
    input       TEXT NOT NULL,
    useful      BOOLEAN NOT NULL,
    note        TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: search depth during index build (higher = better quality, slower build)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search indexes for cache and memory lookups
CREATE INDEX IF NOT EXISTS query_cache_question_idx
    ON query_cache
    USING gin(to_tsvector('english', question));

CREATE INDEX IF NOT EXISTS agent_memory_question_idx
    ON agent_memory
    USING gin(to_tsvector('english', question));