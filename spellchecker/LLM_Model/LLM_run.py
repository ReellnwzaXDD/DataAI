import copy
import hashlib
import json
import logging
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import requests

PROMPT_VER = "sc-001"

logger = logging.getLogger(__name__)

_AGENT_BASE_URL = os.getenv("SPELLCHECKER_AGENT_BASE_URL") or os.getenv(
    "AGENT_BASE_URL", "http://192.168.36.44:5001"
)
_VALIDATE_PATH = os.getenv("SPELLCHECKER_VALIDATE_PATH", "/spellchecker/validate")
_REQUEST_TIMEOUT = float(os.getenv("SPELLCHECKER_LLM_TIMEOUT", "45"))
_MAX_BATCH = int(os.getenv("SPELLCHECKER_LLM_BATCH", "6"))
_CANDIDATE_LIMIT = int(os.getenv("LLM_TRIM_LIMIT", "5"))

_session = requests.Session()
_cache: Dict[str, Dict] = {}


def _build_url(path: str) -> str:
    base = _AGENT_BASE_URL.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _normalize_candidates(candidates: Sequence[dict]) -> List[dict]:
    normalized: List[dict] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        orig = str(cand.get("orig") or "").strip()
        corr = str(cand.get("corr") or "").strip()
        if not orig or not corr:
            continue
        entry = {"orig": orig, "corr": corr}
        if isinstance(cand.get("lang"), str) and cand["lang"]:
            entry["lang"] = cand["lang"]
        score = cand.get("score")
        if isinstance(score, (int, float)):
            entry["score"] = float(score)
        if isinstance(cand.get("reason"), str) and cand["reason"]:
            entry["reason"] = cand["reason"]
        normalized.append(entry)
    return normalized


def _hash_signature(context: str, candidates: Sequence[dict]) -> str:
    payload = {
        "prompt_ver": PROMPT_VER,
        "context": context,
        "candidates": [
            {"orig": c.get("orig"), "corr": c.get("corr"), "lang": c.get("lang")}
            for c in candidates
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def trim_candidates(candidates: Sequence[dict], limit: int | None = None) -> List[dict]:
    if limit is None:
        limit = _CANDIDATE_LIMIT
    unique: List[dict] = []
    seen: set[Tuple[str, str]] = set()
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        orig = str(cand.get("orig") or "").strip()
        corr = str(cand.get("corr") or "").strip()
        if not orig or not corr:
            continue
        key = (orig, corr)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    def sort_key(item: dict) -> Tuple[int, float]:
        score = item.get("score")
        if isinstance(score, (int, float)):
            # negative for descending order
            return (0, -float(score))
        return (1, 0.0)

    unique.sort(key=sort_key)
    if limit and limit > 0:
        return unique[:limit]
    return unique


def cheap_gate(context: str, candidates: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    auto_accept: List[dict] = []
    keep: List[dict] = []
    seen: set[Tuple[str, str]] = set()
    ctx_lower = (context or "").lower()

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        orig = str(cand.get("orig") or "").strip()
        corr = str(cand.get("corr") or "").strip()
        if not orig or not corr:
            continue
        key = (orig, corr)
        if key in seen:
            continue
        seen.add(key)

        orig_clean = orig.strip()
        corr_clean = corr.strip()

        if orig_clean == corr_clean:
            auto_accept.append(cand)
            continue
        if orig_clean.lower() == corr_clean.lower():
            auto_accept.append(cand)
            continue
        if len(orig_clean) <= 2 and orig_clean.lower() == corr_clean.lower():
            auto_accept.append(cand)
            continue
        if orig_clean.lower() in ctx_lower and corr_clean.lower() not in ctx_lower and len(orig_clean) <= 4:
            # likely typo fix bringing new token; keep for LLM
            keep.append(cand)
            continue
        keep.append(cand)

    return keep, auto_accept


def _post(endpoint: str, payload: dict) -> dict:
    url = _build_url(endpoint)
    try:
        response = _session.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Spellchecker LLM request failed: %s", exc)
        raise


def _call_validate(entries: List[dict]) -> List[dict]:
    if not entries:
        return []
    data = _post(_VALIDATE_PATH, {"entries": entries})
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError("Invalid response payload from spellchecker LLM endpoint")
    normalized: List[dict] = []
    for item in results:
        if isinstance(item, dict) and "apply" in item and "reject" in item:
            normalized.append(item)
        else:
            normalized.append({"apply": [], "reject": []})
    return normalized


def _llm_decide(context: str, candidates: Sequence[dict]) -> dict:
    normalized = _normalize_candidates(candidates)
    if not normalized:
        return {"apply": [], "reject": []}
    cache_key = _hash_signature(context, normalized)
    cached = _cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    try:
        result = _call_validate([{"context": context, "candidates": normalized}])[0]
    except Exception:
        result = {"apply": [], "reject": []}

    _cache[cache_key] = copy.deepcopy(result)
    return result


def llm_validate_cached(context: str, candidates: Sequence[dict]) -> dict:
    return _llm_decide(context, candidates)


def llm_validate_many(pairs: Iterable[Tuple[str, Sequence[dict]]]) -> List[dict]:
    pairs_list = list(pairs)
    if not pairs_list:
        return []

    results: List[dict] = [{"apply": [], "reject": []} for _ in pairs_list]
    batch_entries: List[dict] = []
    batch_index: List[int] = []

    for idx, (context, candidates) in enumerate(pairs_list):
        normalized = _normalize_candidates(candidates)
        if not normalized:
            continue
        cache_key = _hash_signature(context, normalized)
        cached = _cache.get(cache_key)
        if cached is not None:
            results[idx] = copy.deepcopy(cached)
            continue
        batch_entries.append({"context": context, "candidates": normalized})
        batch_index.append(idx)

        if len(batch_entries) == _MAX_BATCH:
            _flush_batch(batch_entries, batch_index, results)

    if batch_entries:
        _flush_batch(batch_entries, batch_index, results)

    return results


def _flush_batch(batch_entries: List[dict], batch_index: List[int], results: List[dict]) -> None:
    try:
        response_items = _call_validate(batch_entries)
    except Exception:
        # On error, leave defaults (empty apply/reject)
        batch_entries.clear()
        batch_index.clear()
        return

    for offset, item in enumerate(response_items):
        target_idx = batch_index[offset]
        normalized = item if isinstance(item, dict) else {"apply": [], "reject": []}
        results[target_idx] = normalized
        cache_key = _hash_signature(
            batch_entries[offset]["context"],
            batch_entries[offset]["candidates"],
        )
        _cache[cache_key] = copy.deepcopy(normalized)

    batch_entries.clear()
    batch_index.clear()


def init_llm() -> None:
    try:
        warmup()
    except Exception as exc:
        logger.warning("Spellchecker LLM warmup failed: %s", exc)


def warmup() -> None:
    sample_context = "Sentence: ตัวอย่างข้อความพร้อมคำผิด -> example text with typo"
    sample_candidates = [{"orig": "ผิด", "corr": "ถูก", "lang": "Thai"}]
    try:
        _call_validate([{"context": sample_context, "candidates": sample_candidates}])
    except Exception as exc:
        logger.debug("Spellchecker LLM warmup request failed: %s", exc)
