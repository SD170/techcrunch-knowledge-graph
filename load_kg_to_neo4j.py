"""
Load techcrunch_kg.jsonl into Neo4j in 2 phases:
  Phase A: MERGE nodes by stable keys (Article by id; Company/Person/Investor/Theme/Location/Event by name; FundingRound by round_id).
  Phase B: MERGE edges; store evidence + article_id for extracted edges, article_id only for structural.

Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (default: neo4j://localhost:7687, neo4j, <empty>).
Input: KG_JSONL_PATH (default: data/output/techcrunch_kg.jsonl).
Requires: pip install neo4j
"""

import os
import re
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

STRUCTURAL_EDGE_TYPES = {"PUBLISHED_BY", "ANNOUNCED_IN", "MENTIONS", "ABOUT"}
INPUT_FILE = os.environ.get("KG_JSONL_PATH", "data/output/techcrunch_kg.jsonl")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "unknown"


def _stable_id_entity(e: dict, label: str) -> str:
    """Stable id for non-FundingRound: type:slug(name)."""
    name = (e.get("name") or "").strip()
    return f"{label}:{_slug(name)}"


def _round_id_from_entity(e: dict, company_name: str) -> str:
    """Deterministic round_id = company + stage + amount + date (whatever available)."""
    props = e.get("properties") or {}
    name = (e.get("name") or "").strip()
    parts = [_slug(company_name or "unknown")]
    parts.append(_slug(props.get("stage") or name or "unknown"))
    parts.append(_slug(str(props.get("amount", "")) or "unknown"))
    parts.append(_slug(props.get("date") or "unknown"))
    return "round:" + "_".join(parts)


def _collect_nodes_and_edges(jsonl_path: str):
    """Read JSONL; return (nodes_by_stable_id, edges_with_stable_ids)."""
    nodes = {}  # stable_id -> {label, name, id, evidence, properties, ...}
    edges = []  # {source_id, target_id, type, evidence, article_id, is_structural}
    # id_map: article-scoped id -> stable_id (built as we go)
    id_to_stable = {}
    # Resolve company for FundingRound from RAISED edges (source=company, target=round)
    round_to_company = {}  # round_entity_id -> company_name

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            article = row.get("article") or {}
            article_id = article.get("id")
            publisher = row.get("publisher") or {}
            entities = row.get("entities") or []
            relationships = row.get("relationships") or []

            # Article node
            if article_id:
                nodes[article_id] = {
                    "stable_id": article_id,
                    "label": "Article",
                    "name": article.get("name") or article_id,
                    "properties": article.get("properties") or {},
                }
                id_to_stable[article_id] = article_id

            # Publisher (Organization)
            pub_id = publisher.get("id")
            if pub_id:
                nodes[pub_id] = {
                    "stable_id": pub_id,
                    "label": "Organization",
                    "name": publisher.get("name") or "TechCrunch",
                    "properties": {},
                }
                id_to_stable[pub_id] = pub_id

            # Entity id -> entity dict for this row
            entity_by_id = {e.get("id"): e for e in entities if e.get("id")}

            # First pass: RAISED edges to get company for each FundingRound
            for r in relationships:
                if r.get("type") == "RAISED":
                    src = r.get("source_id")
                    tgt = r.get("target_id")
                    if src and tgt:
                        company_e = entity_by_id.get(src)
                        if company_e and entity_by_id.get(tgt, {}).get("type") == "FundingRound":
                            company_name = (company_e.get("name") or "").strip()
                            round_to_company[tgt] = company_name

            # Entities -> stable ids and nodes
            for e in entities:
                eid = e.get("id")
                if not eid:
                    continue
                label = e.get("type") or "Entity"
                name = (e.get("name") or "").strip()
                if not name:
                    continue
                if label == "FundingRound":
                    company_name = round_to_company.get(eid, "")
                    stable_id = _round_id_from_entity(e, company_name)
                else:
                    stable_id = _stable_id_entity(e, label)
                id_to_stable[eid] = stable_id
                if stable_id not in nodes:
                    nodes[stable_id] = {
                        "stable_id": stable_id,
                        "label": label,
                        "name": name,
                        "evidence": e.get("evidence"),
                        "properties": e.get("properties") or {},
                    }

            # Edges (with article_id)
            for r in relationships:
                src = r.get("source_id")
                tgt = r.get("target_id")
                rtype = r.get("type")
                if not src or not tgt or not rtype:
                    continue
                src_stable = id_to_stable.get(src)
                tgt_stable = id_to_stable.get(tgt)
                if not src_stable or not tgt_stable:
                    continue
                is_structural = rtype in STRUCTURAL_EDGE_TYPES
                evidence = None if is_structural else (r.get("evidence") or "").strip()
                edges.append({
                    "source_id": src_stable,
                    "target_id": tgt_stable,
                    "type": rtype,
                    "evidence": evidence or None,
                    "article_id": article_id,
                    "is_structural": is_structural,
                })

    return nodes, edges


def run_phase_a(driver, nodes: dict):
    """MERGE all nodes by stable key."""
    with driver.session() as session:
        for stable_id, n in nodes.items():
            label = n["label"]
            name = n.get("name") or stable_id
            props = n.get("properties") or {}
            if label == "Article":
                session.run(
                    "MERGE (a:Article {id: $id}) SET a += $props",
                    id=stable_id,
                    props={"title": props.get("title"), "date": props.get("date"), "url": props.get("url"), "name": name},
                )
            elif label == "Organization":
                session.run(
                    "MERGE (o:Organization {id: $id}) SET o.name = $name",
                    id=stable_id,
                    name=name,
                )
            elif label == "FundingRound":
                session.run(
                    "MERGE (f:FundingRound {id: $id}) SET f.name = $name, f += $props",
                    id=stable_id,
                    name=name,
                    props={k: v for k, v in props.items() if v is not None},
                )
            else:
                # Company, Person, Investor, Theme, Location, Event
                session.run(
                    f"MERGE (n:{label} {{name: $name}}) SET n.id = $id, n.evidence = $evidence, n += $props",
                    id=stable_id,
                    name=name,
                    evidence=n.get("evidence"),
                    props={k: v for k, v in props.items() if v is not None},
                )
    print(f"Phase A: merged {len(nodes)} nodes.")


def run_phase_b(driver, edges: list):
    """Create relationships with evidence and article_id."""
    with driver.session() as session:
        for e in edges:
            # Cypher rel type: alphanumeric + underscore only
            rtype = (e["type"] or "").replace(" ", "_").upper()
            rel_type = "".join(c if (c.isalnum() or c == "_") else "_" for c in rtype).strip("_") or "RELATED"
            if not rel_type:
                continue
            article_id = e.get("article_id")
            if e.get("is_structural"):
                session.run(
                    f"""
                    MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.article_id = $article_id
                    """,
                    source_id=e["source_id"],
                    target_id=e["target_id"],
                    article_id=article_id,
                )
            else:
                session.run(
                    f"""
                    MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.evidence = $evidence, r.article_id = $article_id
                    """,
                    source_id=e["source_id"],
                    target_id=e["target_id"],
                    evidence=e.get("evidence"),
                    article_id=article_id,
                )
    print(f"Phase B: created {len(edges)} edges.")


def main():
    jsonl_path = INPUT_FILE
    if not os.path.isfile(jsonl_path):
        print(f"Missing {jsonl_path}")
        return
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        nodes, edges = _collect_nodes_and_edges(jsonl_path)
        run_phase_a(driver, nodes)
        run_phase_b(driver, edges)
    finally:
        driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
