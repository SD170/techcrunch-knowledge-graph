# TechCrunch Knowledge Graph

LLM-extracted knowledge graph from TechCrunch articles (companies, rounds, investors, themes, evidence on every edge). Built as the graph layer for a future GraphRAG pipeline.

## Setup

```bash
cp .env-example .env
# Edit .env: set OPENAI_API_KEY and optionally NEO4J_* if loading to Neo4j.

pip install -r requirements.txt
```

## Instructions

1. **Input:** Put raw TechCrunch article JSON in `data/input/techcruch_raw.json`.  
   Expected shape: list of objects with at least `"title"` and `"content"` (or `"text"`).  
   See `data-example/input/techcruch_raw.json` for an example.

2. **Extract KG:**
   ```bash
   python extract_techcrunch_kg.py
   ```
   Writes `data/output/techcrunch_kg.jsonl` (one JSON object per line: entities + relationships with evidence).

3. **Load into Neo4j (optional):**
   ```bash
   # Start Neo4j (e.g. Docker: docker run -p 7474:7474 -p 7687:7687 neo4j)
   python load_kg_to_neo4j.py
   ```
   Uses `KG_JSONL_PATH` (default `data/output/techcrunch_kg.jsonl`) and env vars `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

Example outputs and visuals are in `data-example/`.
