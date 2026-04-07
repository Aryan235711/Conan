"""Case loader — reads JSON case files and insight definitions from disk.

Community contributions go into detective_engine/cases/ as JSON files.
Each file can define one or more cases + associated insights.

Standard JSON format:
{
  "insights": [ ... ],
  "cases": [ ... ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import CaseDefinition, EvidenceMeta, Insight


_DEFAULT_CASES_DIR = Path(__file__).resolve().parent.parent / "cases"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class CaseLoader:
    """Load cases and insights from JSON files."""

    def __init__(self, cases_dir: Path | str | None = None):
        self.cases_dir = Path(cases_dir) if cases_dir else _DEFAULT_CASES_DIR

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def load_all(self) -> tuple[list[CaseDefinition], list[Insight]]:
        """Load every .json file in the cases directory.

        Returns (cases, insights) — both sorted by ID / key.
        """
        cases: list[CaseDefinition] = []
        insights: list[Insight] = []

        if not self.cases_dir.is_dir():
            return cases, insights

        for path in sorted(self.cases_dir.glob("*.json")):
            file_cases, file_insights = self.load_file(path)
            cases.extend(file_cases)
            insights.extend(file_insights)

        # de-duplicate by id / key (last wins)
        seen_cases: dict[str, CaseDefinition] = {}
        for c in cases:
            seen_cases[c.id] = c
        seen_insights: dict[str, Insight] = {}
        for i in insights:
            seen_insights[i.key] = i

        return list(seen_cases.values()), list(seen_insights.values())

    @staticmethod
    def load_file(path: Path) -> tuple[list[CaseDefinition], list[Insight]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = [CaseDefinition.from_dict(c) for c in data.get("cases", [])]
        insights = [Insight.from_dict(i) for i in data.get("insights", [])]

        # V11: attach optional evidence_meta as a runtime attribute on each case.
        # CaseDefinition is frozen, so we use object.__setattr__ to attach the
        # evidence_map without modifying the dataclass definition.  The runner
        # reads this via getattr(case, "evidence_map", {}).
        case_meta: dict[str, dict[str, EvidenceMeta]] = {}
        for raw_case in data.get("cases", []):
            cid = raw_case.get("id", "")
            meta_list = raw_case.get("evidence_meta", [])
            if meta_list:
                case_meta[cid] = {
                    m["id"]: EvidenceMeta.from_dict(m) for m in meta_list
                }

        for case in cases:
            emap = case_meta.get(case.id, {})
            object.__setattr__(case, "evidence_map", emap)

        return cases, insights

    @staticmethod
    def save_case_file(
        path: Path,
        cases: list[CaseDefinition],
        insights: list[Insight],
    ) -> None:
        """Write cases + insights back to a JSON file (for tooling)."""
        from dataclasses import asdict
        data = {
            "insights": [asdict(i) for i in insights],
            "cases": [asdict(c) for c in cases],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
