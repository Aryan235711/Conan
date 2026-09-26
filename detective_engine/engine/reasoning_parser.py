"""Reasoning output parser — gatekeeper between free-form LLM text and the reward engine.

Three layers, applied in sequence:

    Layer A — Structure extraction
        Regex-based section detection.  Handles markdown headings, numbered lists,
        inline labels, and mixed formatting.  Extracts raw text into the canonical
        dict shape that ``RewardScorer.score()`` expects.

    Layer B — Structural validity (hard filter)
        Rejects outputs missing core sections, containing empty phases,
        or exhibiting absurd lengths.  Returns ``None`` for unparseable garbage.

    Layer C — Semantic grounding (soft filter)
        Checks observation density, hypothesis diversity, cross-reference
        presence.  Blocks "perfect format, zero thinking" exploits.

Usage:

    from detective_engine.engine.reasoning_parser import parse_reasoning_output

    result = parse_reasoning_output(raw_llm_text)
    if result is None:
        # skip this sample — unparseable or semantically empty
        ...
    else:
        analysis, raw_text = result["analysis"], result["raw_text"]
        reward = scorer.score(case_id, analysis)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParserConfig:
    """All tunable thresholds for parse validation."""

    # Layer B — structural
    min_observations: int = 2
    min_anomalies: int = 1
    min_hypotheses_keys: int = 1
    min_explanations_per_key: int = 1
    min_reasons: int = 1
    min_section_token_length: int = 3       # reject 1-2 word sections
    max_missing_sections: int = 1           # reject if ≥2 core sections missing

    # Layer C — semantic grounding
    min_observation_tokens: int = 4         # each observation must have ≥N tokens
    min_nontrivial_observations: int = 2    # ≥N observations passing token threshold
    hypothesis_similarity_limit: float = 0.60  # reject if all hypotheses overlap > this
    min_cross_references: int = 1           # ≥1 hypothesis must reference an observation


DEFAULT_CONFIG = ParserConfig()


# ---------------------------------------------------------------------------
# Section name aliases — what LLMs might call each phase
# ---------------------------------------------------------------------------

_SECTION_ALIASES: dict[str, list[str]] = {
    "observations": [
        "observations", "raw observations", "phase 1",
        "what i see", "what i observe", "initial observations",
        "evidence observations", "facts", "raw facts",
    ],
    "anomalies": [
        "anomalies", "tensions", "oddities", "phase 2",
        "anomalies and tensions", "anomalies & tensions",
        "contradictions observed", "what doesn't fit",
        "inconsistencies", "red flags",
    ],
    "hypotheses": [
        "hypotheses", "explanations", "phase 3",
        "multiple explanations", "possible explanations",
        "competing hypotheses", "theories", "scenarios analysis",
    ],
    "elimination_target": [
        "elimination", "eliminated", "phase 6",
        "elimination target", "rejected scenario",
        "scenario eliminated", "i eliminate", "ruling out",
        "conclusion", "final answer", "answer",
    ],
    "reasons": [
        "reasons", "reasoning", "reasoning chain",
        "evidence chain", "justification", "why",
        "because", "supporting evidence",
    ],
    "false_narrative_rejection": [
        "false narrative", "narrative rejection",
        "phase 5", "rejected narrative", "wrong explanation",
        "surface story", "obvious but wrong",
    ],
    "contradiction_notes": [
        "contradictions", "contradiction detection",
        "phase 4", "contradiction notes", "pairs in tension",
        "conflicting facts",
    ],
}

# Flatten for quick lookup: alias → canonical section name
_ALIAS_MAP: dict[str, str] = {}
for canonical, aliases in _SECTION_ALIASES.items():
    for alias in aliases:
        _ALIAS_MAP[alias] = canonical


# ---------------------------------------------------------------------------
# Layer A — Structure extraction
# ---------------------------------------------------------------------------

# Pattern: markdown heading, numbered item, or label with colon
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+(.+)"           # ## Heading
    r"|(\d+)[\.\)]\s+(.+)"    # 1. or 1) prefix
    r"|([A-Za-z][\w\s&/]+?):\s*$"  # Label:
    r")",
    re.MULTILINE,
)

# Pattern: list items (bullet, dash, asterisk, numbered)
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]\s+|\d+[\.\)]\s+)(.+)",
    re.MULTILINE,
)


def _normalize_section_name(raw: str) -> str | None:
    """Map a raw heading to a canonical section name, or None if unrecognized."""
    cleaned = raw.strip().lower().rstrip(":")
    # Direct alias match
    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]
    # Substring match — "Phase 1: Raw Observations" contains "raw observations"
    for alias, canonical in _ALIAS_MAP.items():
        if alias in cleaned:
            return canonical
    return None


def _split_sections(text: str) -> dict[str, str]:
    """Split text into sections by detecting headings/labels."""
    # Find all heading-like lines and their positions
    markers: list[tuple[int, str]] = []

    for line_match in re.finditer(r"^(.+)$", text, re.MULTILINE):
        line = line_match.group(1).strip()
        # Check markdown heading
        hm = re.match(r"^#{1,4}\s+(.+)", line)
        if hm:
            section = _normalize_section_name(hm.group(1))
            if section:
                markers.append((line_match.start(), section))
                continue

        # Check "Label:" pattern (line ends with colon, or is just a label)
        lm = re.match(r"^([A-Za-z][\w\s&/,]+?)\s*:\s*$", line)
        if lm:
            section = _normalize_section_name(lm.group(1))
            if section:
                markers.append((line_match.start(), section))
                continue

        # Check "**Label**" or "**Label:**"
        bm = re.match(r"^\*\*(.+?)\*\*\s*:?\s*$", line)
        if bm:
            section = _normalize_section_name(bm.group(1))
            if section:
                markers.append((line_match.start(), section))
                continue

    # Extract text between markers
    sections: dict[str, str] = {}
    for i, (pos, name) in enumerate(markers):
        # Find end of this section's header line
        line_end = text.index("\n", pos) if "\n" in text[pos:] else len(text)
        next_pos = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[line_end:next_pos].strip()
        # If section already exists, append (handles split sections)
        if name in sections:
            sections[name] += "\n" + body
        else:
            sections[name] = body

    return sections


def _extract_list_items(text: str) -> list[str]:
    """Extract individual items from a section body (bullets, numbers, or lines)."""
    items = []
    for m in _LIST_ITEM_RE.finditer(text):
        item = m.group(1).strip()
        if item:
            items.append(item)

    # If no list items found, split on newlines (fallback)
    if not items:
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line.split()) >= 2:
                items.append(line)

    return items


def _extract_hypotheses(text: str) -> dict[str, list[str]]:
    """Parse hypotheses section into {observation: [explanations]} dict.

    Handles formats like:
        "observation text": exp1, exp2
        For "observation": - exp1 - exp2
        1. observation — exp1; exp2
    """
    hypotheses: dict[str, list[str]] = {}
    current_key: str | None = None
    current_exps: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for "For 'obs':" or "obs:" or "**obs**:" pattern
        key_match = re.match(
            r"^(?:for\s+)?[\"'](.+?)[\"']\s*:|"
            r"^(?:for\s+)?(.+?):\s*$|"
            r"^\*\*(.+?)\*\*\s*:",
            line, re.IGNORECASE,
        )
        if key_match:
            # Save previous
            if current_key and current_exps:
                hypotheses[current_key] = current_exps
            current_key = (key_match.group(1) or key_match.group(2) or key_match.group(3)).strip()
            current_exps = []
            # Check if explanations are on the same line after the colon
            after_colon = line.split(":", 1)[1].strip() if ":" in line else ""
            if after_colon:
                for exp in re.split(r"[;,]|\bor\b", after_colon):
                    exp = exp.strip().lstrip("-*• ")
                    if exp and len(exp.split()) >= 2:
                        current_exps.append(exp)
            continue

        # List item under current key
        item_match = re.match(r"^\s*[-*•]\s+(.+)|^\s*\d+[\.\)]\s+(.+)", line)
        if item_match and current_key:
            exp = (item_match.group(1) or item_match.group(2)).strip()
            if exp:
                current_exps.append(exp)
            continue

        # Bare line under current key — treat as explanation
        if current_key and len(line.split()) >= 3:
            current_exps.append(line)

    # Save last key
    if current_key and current_exps:
        hypotheses[current_key] = current_exps

    # Fallback: if no keys parsed, treat all items as explanations under "general"
    if not hypotheses:
        items = _extract_list_items(text)
        if items:
            hypotheses["general"] = items

    return hypotheses


def _extract_single(text: str) -> str:
    """Extract a single-value section (elimination target, false narrative)."""
    items = _extract_list_items(text)
    if items:
        return items[0]
    # Fallback: first non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line.split()) >= 2:
            return line
    return text.strip()


def _layer_a(raw_text: str) -> dict[str, Any] | None:
    """Layer A: extract structured sections from free-form text."""
    sections = _split_sections(raw_text)

    if not sections:
        # Last resort: try treating the whole text as a flat list
        # and infer sections from content keywords
        return None

    analysis: dict[str, Any] = {
        "observations": [],
        "anomalies": [],
        "hypotheses": {},
        "elimination_target": "",
        "reasons": [],
        "false_narrative_rejection": "",
        "contradiction_notes": [],
    }

    if "observations" in sections:
        analysis["observations"] = _extract_list_items(sections["observations"])

    if "anomalies" in sections:
        analysis["anomalies"] = _extract_list_items(sections["anomalies"])

    if "hypotheses" in sections:
        analysis["hypotheses"] = _extract_hypotheses(sections["hypotheses"])

    if "elimination_target" in sections:
        analysis["elimination_target"] = _extract_single(sections["elimination_target"])

    if "reasons" in sections:
        analysis["reasons"] = _extract_list_items(sections["reasons"])

    if "false_narrative_rejection" in sections:
        analysis["false_narrative_rejection"] = _extract_single(
            sections["false_narrative_rejection"]
        )

    if "contradiction_notes" in sections:
        analysis["contradiction_notes"] = _extract_list_items(
            sections["contradiction_notes"]
        )

    return analysis


# ---------------------------------------------------------------------------
# Layer B — Structural validity
# ---------------------------------------------------------------------------

_CORE_SECTIONS = ["observations", "hypotheses", "elimination_target", "reasons"]


def _layer_b(analysis: dict[str, Any], config: ParserConfig) -> str | None:
    """Layer B: hard structural rejection.  Returns error string or None if valid."""

    # Count missing core sections
    missing = []
    if len(analysis.get("observations", [])) == 0:
        missing.append("observations")
    if not analysis.get("hypotheses"):
        missing.append("hypotheses")
    if not analysis.get("elimination_target", "").strip():
        missing.append("elimination_target")
    if len(analysis.get("reasons", [])) == 0:
        missing.append("reasons")

    if len(missing) > config.max_missing_sections:
        return f"missing {len(missing)} core sections: {', '.join(missing)}"

    # Minimum counts
    obs = analysis.get("observations", [])
    if len(obs) < config.min_observations:
        return f"only {len(obs)} observations (need ≥{config.min_observations})"

    hyps = analysis.get("hypotheses", {})
    if len(hyps) < config.min_hypotheses_keys:
        return f"only {len(hyps)} hypothesis keys (need ≥{config.min_hypotheses_keys})"

    for key, exps in hyps.items():
        if len(exps) < config.min_explanations_per_key:
            return f"hypothesis '{key[:30]}' has {len(exps)} explanations (need ≥{config.min_explanations_per_key})"

    reasons = analysis.get("reasons", [])
    if len(reasons) < config.min_reasons:
        return f"only {len(reasons)} reasons (need ≥{config.min_reasons})"

    # Absurd section lengths — reject 1-2 word items
    for obs_item in obs:
        if len(obs_item.split()) < config.min_section_token_length:
            return f"observation too short: '{obs_item}'"

    elim = analysis.get("elimination_target", "")
    if len(elim.split()) < config.min_section_token_length:
        return f"elimination target too short: '{elim}'"

    return None


# ---------------------------------------------------------------------------
# Layer C — Semantic grounding
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    """Lowercase token set, filtering stopwords and short words."""
    stops = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "has", "have", "had", "do", "does", "did", "will", "would",
             "could", "should", "may", "might", "can", "shall", "must",
             "not", "no", "and", "or", "but", "if", "then", "so", "as",
             "at", "by", "for", "in", "of", "on", "to", "up", "it",
             "its", "this", "that", "with", "from", "into", "than"}
    return {
        w for w in text.lower().split()
        if len(w) > 2 and w not in stops
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _layer_c(analysis: dict[str, Any], config: ParserConfig) -> str | None:
    """Layer C: semantic grounding checks.  Returns error string or None if valid."""

    # C1: Observation density — enough non-trivial observations
    obs = analysis.get("observations", [])
    nontrivial = [o for o in obs if len(o.split()) >= config.min_observation_tokens]
    if len(nontrivial) < config.min_nontrivial_observations:
        return (
            f"only {len(nontrivial)} non-trivial observations "
            f"(need ≥{config.min_nontrivial_observations} with ≥{config.min_observation_tokens} tokens each)"
        )

    # C2: Hypothesis diversity — not all near-identical
    hyps = analysis.get("hypotheses", {})
    all_exps = [exp for exps in hyps.values() for exp in exps]
    if len(all_exps) >= 2:
        exp_sets = [_token_set(e) for e in all_exps]
        # Check if ALL pairs are too similar
        high_sim_count = 0
        pair_count = 0
        for i in range(len(exp_sets)):
            for j in range(i + 1, len(exp_sets)):
                pair_count += 1
                if _jaccard(exp_sets[i], exp_sets[j]) > config.hypothesis_similarity_limit:
                    high_sim_count += 1
        if pair_count > 0 and high_sim_count == pair_count:
            return "all hypotheses are near-identical (monoculture)"

    # C3: Cross-reference — hypotheses must connect to observations
    # Check both: hypothesis KEYS (often quoted observations) AND explanations
    obs_tokens = set()
    for o in obs:
        obs_tokens |= _token_set(o)

    cross_refs = 0
    for key, exps in hyps.items():
        key_tokens = _token_set(key)
        # Key itself references observations (common: key IS an observation)
        if len(obs_tokens & key_tokens) >= 2:
            cross_refs += 1
            continue
        # Explanations reference observations
        for exp in exps:
            exp_tokens = _token_set(exp)
            if len(obs_tokens & exp_tokens) >= 2:
                cross_refs += 1
                break

    if cross_refs < config.min_cross_references:
        return (
            f"only {cross_refs} hypothesis groups reference observations "
            f"(need ≥{config.min_cross_references})"
        )

    # C4: Reasoning chain references evidence or observations
    reasons = analysis.get("reasons", [])
    reason_tokens = set()
    for r in reasons:
        reason_tokens |= _token_set(r)
    if not (obs_tokens & reason_tokens):
        return "reasoning chain shares no content with observations"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Successful parse output."""
    analysis: dict[str, Any]
    raw_text: str
    sections_found: list[str]
    warnings: list[str]


def parse_reasoning_output(
    raw_text: str,
    config: ParserConfig | None = None,
) -> ParseResult | None:
    """Parse free-form LLM reasoning text into a structured analysis dict.

    Returns ``ParseResult`` on success, ``None`` if the output is unparseable
    or fails structural/semantic validation.

    The caller should check the ``warnings`` list for soft issues that didn't
    cause rejection but indicate quality concerns.
    """
    if not raw_text or not raw_text.strip():
        return None

    cfg = config or DEFAULT_CONFIG
    warnings: list[str] = []

    # Layer A — structure extraction
    analysis = _layer_a(raw_text)
    if analysis is None:
        return None

    sections_found = [
        k for k, v in analysis.items()
        if (isinstance(v, list) and v) or (isinstance(v, dict) and v) or (isinstance(v, str) and v.strip())
    ]

    # Layer B — structural validity
    b_error = _layer_b(analysis, cfg)
    if b_error is not None:
        return None

    # Layer C — semantic grounding
    c_error = _layer_c(analysis, cfg)
    if c_error is not None:
        # Semantic failure is a soft reject — report but still return None
        return None

    # Warn about optional sections that are missing
    if not analysis.get("contradiction_notes"):
        warnings.append("no contradiction notes found")
    if not analysis.get("false_narrative_rejection"):
        warnings.append("no false narrative rejection found")
    if not analysis.get("anomalies"):
        warnings.append("no anomalies section found")

    return ParseResult(
        analysis=analysis,
        raw_text=raw_text,
        sections_found=sections_found,
        warnings=warnings,
    )
