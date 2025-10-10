"""
Lightweight Knowledge Graph utilities for re-ranking spell-correction candidates.

Design goals:
- Zero external dependencies (no networkx); store an adjacency dict.
- Load optional graph from `knowledge_graph.json` if present.
- Fall back to a minimal graph seeded from keywords and surface variants.
- Provide a scoring function to re-rank candidates based on context.

File format for knowledge_graph.json (optional):
{
  "nodes": ["Bangkok", "Thailand", "EGAT"],
  "edges": [["Bangkok", "Thailand"], ["EGAT", "Electricity"]]
}

If absent, we build a tiny graph where each keyword is linked to simple variants
like lowercase, titlecase (English) and itself.
"""

from __future__ import annotations

import os
import json
import re
from typing import Dict, Set, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KG_PATH = os.path.join(PROJECT_ROOT, "knowledge_graph.json")

_KG_ADJ: Dict[str, Set[str]] | None = None


def _norm(s: str) -> str:
    return (s or "").strip()


def _add_edge(adj: Dict[str, Set[str]], a: str, b: str):
    if not a or not b:
        return
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _english_variants(term: str) -> List[str]:
    # generate minimal surface-form variants for English tokens
    t = term.strip()
    if not t:
        return []
    var = {t, t.lower(), t.title(), t.upper()}
    return list(var)


def load_kg(keywords_th: List[str] | None = None, keywords_en: List[str] | None = None) -> Dict[str, Set[str]]:
    global _KG_ADJ
    if _KG_ADJ is not None:
        return _KG_ADJ

    # 1) Try user-provided knowledge_graph.json
    adj: Dict[str, Set[str]] = {}
    if os.path.exists(KG_PATH):
        try:
            with open(KG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = set(map(_norm, data.get("nodes", [])))
            edges = data.get("edges", []) or []
            for a, b in edges:
                a = _norm(a); b = _norm(b)
                if a and b:
                    _add_edge(adj, a, b)
            for n in nodes:
                adj.setdefault(n, set())
        except Exception:
            adj = {}

    # 2) Seed minimal graph from keywords if no file or file empty
    if not adj:
        keywords_th = keywords_th or []
        keywords_en = keywords_en or []
        for t in keywords_th:
            t = _norm(t)
            if t:
                adj.setdefault(t, set())
        for e in keywords_en:
            e = _norm(e)
            if not e:
                continue
            variants = _english_variants(e)
            for v in variants:
                _add_edge(adj, e, v)
    _KG_ADJ = adj
    return adj


def reload_kg() -> None:
    """Clear in-memory KG so next call reloads from disk/keywords."""
    global _KG_ADJ
    _KG_ADJ = None


def export_kg() -> Dict[str, List[List[str]] | List[str]]:
    """Return KG in a simple JSONable form {nodes:[...], edges:[[a,b],...]}."""
    adj = load_kg()
    nodes = sorted(adj.keys())
    edges: Set[Tuple[str, str]] = set()
    for a, nbrs in adj.items():
        for b in nbrs:
            if a <= b:
                edges.add((a, b))
            else:
                edges.add((b, a))
    return {"nodes": nodes, "edges": [[a, b] for a, b in sorted(edges)]}


WORD_RE_EN = re.compile(r"[A-Za-z']{2,}")


def re_rank_candidates(sentence: str,
                       candidates: List[dict],
                       present_th: List[str] | None = None,
                       present_en: List[str] | None = None,
                       keywords_th: List[str] | None = None,
                       keywords_en: List[str] | None = None) -> List[dict]:
    """
    Re-rank candidates using a lightweight KG and context tokens.

    Heuristics:
    - Prefer smaller edits (length diff) and shorter outputs (ties) as base.
    - Penalize changing tokens that match current context keywords (present_*).
    - Bonus if corrected token connects in KG to any English context token.
    - Mild penalty for altering presumed proper nouns (TitleCase, ALLCAPS).
    """
    if not candidates:
        return candidates

    present_th = present_th or []
    present_en = present_en or []
    adj = load_kg(keywords_th or [], keywords_en or [])

    # extract simple English context tokens from sentence for KG linking
    ctx_en = set(m.group(0) for m in WORD_RE_EN.finditer(sentence or ""))
    ctx_en_lower = {t.lower() for t in ctx_en}
    present_en_lower = {t.lower() for t in present_en}
    present_th_set = set(present_th)

    def base_len_score(o: str, c: str) -> Tuple[int, int]:
        return (abs(len(c) - len(o)), len(c))

    def proper_noun_penalty(tok: str) -> float:
        if not tok:
            return 0.0
        if tok.isupper():
            return 1.0
        if tok[:1].isupper() and tok[1:].islower():
            return 0.5
        return 0.0

    def kg_bonus(word: str) -> float:
        if not word:
            return 0.0
        nbrs = adj.get(word, set()) | adj.get(word.lower(), set()) | adj.get(word.title(), set()) | adj.get(word.upper(), set())
        if not nbrs:
            return 0.0
        # Any overlap with context → give a small boost
        if any((n in ctx_en) or (n.lower() in ctx_en_lower) for n in nbrs):
            return 1.0
        return 0.0

    def protect_penalty(o: str, c: str, lang: str) -> float:
        # If trying to change a keyword present in the local context, penalize
        if (lang or '').lower().startswith('th'):
            if (o in present_th_set) or (c in present_th_set):
                return 2.0
        else:
            if (o.lower() in present_en_lower) or (c.lower() in present_en_lower):
                return 2.0
        return 0.0

    scored: List[Tuple[Tuple[int, int], float, dict]] = []
    for it in candidates:
        o = (it.get("orig") or "").strip()
        c = (it.get("corr") or "").strip()
        lang = it.get("lang") or ""
        base = base_len_score(o, c)
        score = 0.0
        score -= proper_noun_penalty(o)
        score -= protect_penalty(o, c, lang)
        score += kg_bonus(c)
        # Slight preference for case-only fixes if not acronyms
        if o.lower() == c.lower() and o != c and not o.isupper():
            score += 0.25
        scored.append((base, score, it))

    # Sort by base (smaller edit, shorter) ascending, then by score descending
    scored.sort(key=lambda x: (x[0][0], x[0][1], -x[1]))
    return [it for _, _, it in scored]


def _neighbors_for(adj: Dict[str, Set[str]], term: str) -> Set[str]:
    """Return adjacency set including simple case variants for English."""
    t = (term or "").strip()
    if not t:
        return set()
    nbrs: Set[str] = set()
    for v in {t, t.lower(), t.title(), t.upper()}:
        nbrs |= adj.get(v, set())
    return nbrs


def get_kg_context_map(sentence: str,
                       present_th: List[str] | None = None,
                       present_en: List[str] | None = None,
                       max_terms: int = 5,
                       max_neighbors: int = 6) -> Dict[str, List[str]]:
    """
    Build a compact mapping of context terms → a few KG neighbors.

    - Terms are selected from present_en/present_th and English tokens in sentence.
    - Only include terms that exist in the KG and have at least one neighbor.
    - Limit number of terms and neighbors to keep prompts small.
    """
    adj = load_kg()
    present_th = present_th or []
    present_en = present_en or []

    # Collect potential terms: explicit present keywords plus English tokens
    ctx_en_tokens = [m.group(0) for m in WORD_RE_EN.finditer(sentence or "")] if sentence else []
    # Preserve insertion order while deduping
    seen: Dict[str, None] = {}
    for t in (present_en + ctx_en_tokens + present_th):
        s = (t or "").strip()
        if not s:
            continue
        if s not in seen:
            seen[s] = None

    out: Dict[str, List[str]] = {}
    for term in list(seen.keys())[: max_terms]:
        nbrs = list(_neighbors_for(adj, term))
        if not nbrs:
            continue
        # Prefer neighbors that also appear in sentence for relevance
        in_sent = []
        others = []
        sent_low = (sentence or "").lower()
        for n in nbrs:
            if not n:
                continue
            if (n in (sentence or "")) or (n.lower() in sent_low):
                in_sent.append(n)
            else:
                others.append(n)
        # Prioritize in-sentence neighbors then fill with others
        ordered = in_sent + others
        if not ordered:
            continue
        out[term] = ordered[: max_neighbors]
    return out


def ensure_nodes_edges(nodes: List[str] | None = None,
                       edges: List[Tuple[str, str]] | None = None,
                       persist: bool = False) -> None:
    """
    Ensure given nodes/edges exist in the in-memory KG. Optionally persist to file.
    Also adds English variant edges for any English node added.
    """
    nodes = nodes or []
    edges = edges or []
    adj = load_kg()

    # Add nodes and minimal variant edges for English terms
    for n in nodes:
        t = _norm(n)
        if not t:
            continue
        adj.setdefault(t, set())
        # Add variant links for English surface forms
        if WORD_RE_EN.fullmatch(t):
            for v in _english_variants(t):
                _add_edge(adj, t, v)

    # Add explicit edges
    for a, b in edges:
        a = _norm(a); b = _norm(b)
        if a and b:
            _add_edge(adj, a, b)

    # Optionally persist to JSON
    if persist:
        try:
            # Build stable lists
            nodes_out = sorted(adj.keys())
            uniq_edges: Set[Tuple[str, str]] = set()
            for a, nbrs in adj.items():
                for b in nbrs:
                    if a <= b:
                        uniq_edges.add((a, b))
                    else:
                        uniq_edges.add((b, a))
            data = {"nodes": nodes_out, "edges": [[a, b] for a, b in sorted(uniq_edges)]}
            with open(KG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # reload cache to reflect persisted state
            reload_kg()
        except Exception:
            # Non-fatal: keep in-memory changes
            pass
