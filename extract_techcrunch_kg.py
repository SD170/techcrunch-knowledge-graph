"""
Extract a KG from TechCrunch raw article JSON → data/output/techcrunch_kg.jsonl.
Reads data/input/techcruch_raw.json.
Schema:
  Nodes: Company, FundingRound, Investor, Theme, Article, Organization (publisher), Event, Person
  Edges (unified in "relationships"): RAISED, LED_ROUND, PARTICIPATED_IN_ROUND, OPERATES_IN,
         USES_TECH, TRANSCRIBES_WITH, LEADS/CEO_OF, PREVIOUSLY_WORKED_AT, HOSTS, SPONSORS,
         ANNOUNCED_IN, MENTIONS, ABOUT, PUBLISHED_BY (Article→Organization).
  Validity: no entity/relationship with null name/type/evidence; theme/relation evidence rules.
  Structural edges (PUBLISHED_BY, ANNOUNCED_IN, MENTIONS, ABOUT) use fixed evidence strings.
  Publisher entity is not duplicated in entities; HOSTS only for Organization->Event.
"""

import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

INPUT_FILE = "data/input/techcruch_raw.json"
OUTPUT_FILE = "data/output/techcrunch_kg.jsonl"
PUBLISHER_ID = "organization:techcrunch"
PUBLISHER_NODE = {"id": PUBLISHER_ID, "type": "Organization", "name": "TechCrunch"}

EXTRACTION_PROMPT = """
Extract entities and relationships from this article for a knowledge graph.

HARD VALIDITY RULES (you MUST follow):
- Do NOT output any entity or relationship with missing name/type/evidence. Output nothing instead.
- Every entity must have non-empty "name" and "evidence" (exact short quote). No placeholders.
- Every relationship must have non-empty "type" and "evidence" (exact short quote). Never output null type or null evidence.
- Do NOT output a relationship if either source or target would be invalid or missing.

THEME RULE:
- Only output Theme if the theme label appears in the evidence quote (verbatim or close match). E.g. evidence "AI news app" supports Theme "AI", not "AI agents".
- Only use real categories (AI, fintech, devtools, robotics, healthcare, etc.). Do NOT use marketing copy ("startup trajectory") or generic labels ("startup", "VC") as Theme.

RELATION EVIDENCE RULE:
- For HOSTS, SPONSORS, PARTICIPATED_IN, PARTICIPATES_IN, EXHIBITS_AT, evidence must include the predicate phrase (e.g. "Sponsors include Discord", "Discord will be at", "partner: Cloudflare"), not just the company name. Otherwise do not create the relation—MENTIONS will be added in code.

PUBLISHER RULE:
- Do not create a separate TechCrunch (or the article publisher) entity in entities. The publisher is provided separately. Use only the publisher id for PUBLISHED_BY. Do not create Organization -[:HOSTS]-> Company (e.g. "told TechCrunch" is reporting, not hosting).

NODE TYPES:
- Company: only when the article clearly discusses a company. Not the publisher (TechCrunch).
- FundingRound: ONLY when the article explicitly mentions a round (amount/stage/date or "raised X").
- Investor: only when named in a funding context; "role": "lead" only if text says "led by".
- Theme: only when the article explicitly uses that category label; theme name must appear in evidence (see THEME RULE).
- Organization: for the publication or event host (e.g. TechCrunch). Do NOT create "TechCrunch HOSTS Company"—that is reporting, not hosting. Use Organization only for publisher or for who hosts an Event.
- Event: for named events (e.g. "TechCrunch Disrupt 2026"). Companies like Discord/Cloudflare/Trello are Company, not Theme.
- Person: for named people. Use LEADS or CEO_OF for current leadership (e.g. "OpenAI CEO Sam Altman" → Sam Altman LEADS OpenAI). Use PREVIOUSLY_WORKED_AT only for past employment (e.g. "formerly at Twitter").

RELATION TYPES:
- Company -[:RAISED]-> FundingRound
- Investor -[:LED_ROUND]-> FundingRound   only if evidence contains "led by"
- Investor -[:PARTICIPATED_IN_ROUND]-> FundingRound
- Company -[:OPERATES_IN]-> Theme   (theme must appear in evidence)
- Company -[:USES_TECH]-> Company   (e.g. uses X for Y)
- Company -[:TRANSCRIBES_WITH]-> Company   (only if explicitly said)
- Person -[:LEADS]-> Company   or -[:CEO_OF]-> Company   (current leadership)
- Person -[:PREVIOUSLY_WORKED_AT]-> Company   (past employment only)
- Organization -[:HOSTS]-> Event   (only if evidence says X hosts the event)
- Company -[:SPONSORS]-> Event   only if evidence says sponsor/sponsoring
- Do not create Article, ANNOUNCED_IN, or MENTIONS (added in code).

Output strict JSON only:
{{
  "entities": [
    {{ "id": "e1", "type": "...", "name": "non-empty", "evidence": "exact short quote", "properties": {{}} }}
  ],
  "relationships": [
    {{ "source_id": "e1", "target_id": "e2", "type": "RELATION_TYPE", "evidence": "exact short quote" }}
  ]
}}

Article to analyze:
---
Title: {title}
Date: {date}
---
{body}
"""


def get_data_from_json_file(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_llm_instance():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = dict(api_key=api_key, model=model, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
    return ChatOpenAI(**kwargs)


def _parse_extraction(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        for part in s.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                s = part
                break
            if '"entities"' in part or '"relationships"' in part:
                s = part
                break
    if s.strip() and not s.lstrip().startswith("{"):
        s = "{" + s.lstrip()
    s = re.sub(r",\s*}\s*$", "}", s)
    s = re.sub(r",\s*]\s*$", "]", s)
    return json.loads(s)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "unknown"


def _is_valid_entity(e: dict) -> bool:
    """Entity must have non-empty name and non-empty evidence. No placeholders."""
    name = e.get("name")
    evidence = e.get("evidence")
    if name is None or evidence is None:
        return False
    if not isinstance(name, str) or not name.strip():
        return False
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    return True


def _filter_entities(entities: list) -> list:
    return [e for e in entities if _is_valid_entity(e)]


def _drop_publisher_entity(entities: list, publisher_name: str = "TechCrunch") -> list:
    """Remove publisher from entities so we use only the canonical publisher node."""
    norm = (publisher_name or "").strip().lower()
    if not norm:
        return entities
    return [e for e in entities if not (e.get("type") == "Organization" and (e.get("name") or "").strip().lower() == norm)]


def _drop_hosts_to_non_event(relationships: list, id_to_type: dict) -> list:
    """Keep HOSTS only when target is Event. Drop e.g. TechCrunch HOSTS Particle."""
    return [
        r for r in relationships
        if r.get("type") != "HOSTS" or id_to_type.get(r.get("target_id")) == "Event"
    ]


def _is_valid_relationship(r: dict, valid_ids: set) -> bool:
    """Relationship must have non-null type and evidence; both endpoints in valid_ids."""
    rtype = r.get("type")
    evidence = r.get("evidence")
    if rtype is None or evidence is None:
        return False
    if not isinstance(rtype, str) or not rtype.strip():
        return False
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    sid = r.get("source_id")
    tid = r.get("target_id")
    return bool(sid and tid and sid in valid_ids and tid in valid_ids)


def _filter_relationships(relationships: list, valid_ids: set) -> list:
    """Keep only relationships with valid type, evidence, and endpoints."""
    return [r for r in relationships if _is_valid_relationship(r, valid_ids)]


def run():
    articles = get_data_from_json_file(INPUT_FILE)
    if not isinstance(articles, list):
        articles = [articles]
    llm = generate_llm_instance()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for idx, art in enumerate(articles):
            title = art.get("title") or ""
            date = art.get("date") or art.get("date_raw") or ""
            url = art.get("url") or ""
            paragraphs = art.get("paragraphs") or []
            body = "\n\n".join(p for p in paragraphs if isinstance(p, str))
            article_id = f"article:{_slug(title) or idx}"
            article_node = {
                "id": article_id,
                "type": "Article",
                "name": title,
                "properties": {"title": title, "date": date, "url": url, "author": None},
            }
            print(f"Article {idx}/{len(articles)}: {title[:50]}...")
            try:
                prompt = EXTRACTION_PROMPT.format(title=title, date=date, body=body[:12000])
                response = llm.invoke(prompt)
                out = _parse_extraction(response.content)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  parse error: {e}")
                row = {
                    "article": article_node,
                    "publisher": PUBLISHER_NODE,
                    "entities": [],
                    "relationships": [{
                        "source_id": article_id,
                        "target_id": PUBLISHER_ID,
                        "type": "PUBLISHED_BY",
                        "evidence": "Structural: article published by publisher.",
                    }],
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                continue
            entities = _filter_entities(out.get("entities", []))
            entities = _drop_publisher_entity(entities, PUBLISHER_NODE.get("name"))
            raw_rels = out.get("relationships", [])
            if not isinstance(raw_rels, list):
                raw_rels = []
            relationships = [r for r in raw_rels if isinstance(r, dict) and (r.get("source_id") is not None or r.get("target_id") is not None)]
            # Prefix entity ids with article to avoid collisions
            eid_map = {}
            for e in entities:
                old_id = e.get("id", "")
                new_id = f"{article_id}:{old_id}"
                eid_map[old_id] = new_id
                e["id"] = new_id
            valid_ids = set(e["id"] for e in entities)
            id_to_type = {e["id"]: e.get("type") for e in entities}
            for r in relationships:
                r["source_id"] = eid_map.get(r.get("source_id"), r.get("source_id"))
                r["target_id"] = eid_map.get(r.get("target_id"), r.get("target_id"))
            relationships = _filter_relationships(relationships, valid_ids)
            relationships = _drop_hosts_to_non_event(relationships, id_to_type)
            # Structural edges (fixed evidence: metadata, not extracted from text)
            # FundingRound ANNOUNCED_IN Article
            for e in entities:
                if e.get("type") == "FundingRound":
                    relationships.append({
                        "source_id": e["id"],
                        "target_id": article_id,
                        "type": "ANNOUNCED_IN",
                        "evidence": "Structural: funding announced in this article.",
                    })
            # Article MENTIONS Company | Investor | Theme
            mention_types = {"Company", "Investor", "Theme"}
            for e in entities:
                if e.get("type") in mention_types:
                    relationships.append({
                        "source_id": article_id,
                        "target_id": e["id"],
                        "type": "MENTIONS",
                        "evidence": "Structural: entity mentioned in article.",
                    })
            # Article ABOUT Event
            for e in entities:
                if e.get("type") == "Event":
                    relationships.append({
                        "source_id": article_id,
                        "target_id": e["id"],
                        "type": "ABOUT",
                        "evidence": "Structural: article about this event.",
                    })
            # Article PUBLISHED_BY Organization (provenance)
            relationships.append({
                "source_id": article_id,
                "target_id": PUBLISHER_ID,
                "type": "PUBLISHED_BY",
                "evidence": "Structural: article published by publisher.",
            })
            row = {
                "article": article_node,
                "publisher": PUBLISHER_NODE,
                "entities": entities,
                "relationships": relationships,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
