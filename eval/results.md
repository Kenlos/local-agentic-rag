# RAGAs Evaluation Results

## Setup
- Questions: 3
- Enhanced pipeline: query expansion + hybrid retrieval (BM25 + dense) + RRF fusion + cross-encoder reranking
- Naive baseline: dense vector search only

## Results

| Metric            | Naive  | Enhanced | Delta  |
|-------------------|--------|----------|--------|
| answer_relevancy  | 0.6349 | 0.9701   | +0.34  |
| context_recall    | 0.0000 | 0.5000   | +0.50  |

## Notes
- faithfulness and context_precision timed out due to local hardware constraints
- Metrics scored successfully show meaningful improvement over naive baseline