"""Case validator — enforces quality standards on community-contributed JSON cases.

Checks:
    1. Required fields present and non-empty
    2. Minimum 1 contradiction
    3. Minimum 1 forbidden pattern
    4. Minimum 1 concept rule in solution
    5. Dependency integrity — requires_all references only known insight keys
    6. teaches references are unique across all loaded cases
    7. No circular dependencies
    8. Evidence has ≥3 items
    9. Scenarios has ≥2 items
    10. Analysis protocol has ≥3 steps

Returns a CaseValidationReport with all issues found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .case_loader import CaseLoader
from .models import CaseDefinition, Insight


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single quality issue found in a case file."""
    case_id: str
    severity: str   # "error" | "warning"
    message: str


@dataclass
class CaseValidationReport:
    """Aggregated validation results."""
    issues: list[ValidationIssue] = field(default_factory=list)
    cases_checked: int = 0
    errors: int = 0
    warnings: int = 0
    valid: bool = True

    def summary(self) -> str:
        lines = [
            f"CASE VALIDATION REPORT",
            f"{'━' * 44}",
            f"  Cases checked: {self.cases_checked}",
            f"  Errors:        {self.errors}",
            f"  Warnings:      {self.warnings}",
            f"  Status:        {'✅ VALID' if self.valid else '❌ INVALID'}",
        ]
        if self.issues:
            lines.append("")
            for issue in self.issues:
                icon = "❌" if issue.severity == "error" else "⚠️"
                lines.append(f"  {icon} [{issue.case_id}] {issue.message}")
        lines.append(f"{'━' * 44}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CaseQualityValidator:
    """Validates quality standards for community case contributions."""

    def validate_all(
        self,
        cases: list[CaseDefinition],
        insights: list[Insight],
    ) -> CaseValidationReport:
        """Validate all loaded cases against quality standards."""
        report = CaseValidationReport(cases_checked=len(cases))

        all_insight_keys = {i.key for i in insights}
        all_taught = {}  # key → case_id that teaches it

        for case in cases:
            self._validate_case(case, all_insight_keys, all_taught, report)

        # Check for circular dependencies
        self._check_circular(cases, all_insight_keys, report)

        report.errors = sum(1 for i in report.issues if i.severity == "error")
        report.warnings = sum(1 for i in report.issues if i.severity == "warning")
        report.valid = report.errors == 0

        return report

    def validate_directory(self, cases_dir: Path | str | None = None) -> CaseValidationReport:
        """Load and validate all cases from a directory."""
        loader = CaseLoader(cases_dir=cases_dir)
        cases, insights = loader.load_all()
        return self.validate_all(cases, insights)

    # ------------------------------------------------------------------
    # per-case checks
    # ------------------------------------------------------------------

    def _validate_case(
        self,
        case: CaseDefinition,
        known_insights: set[str],
        taught_by: dict[str, str],
        report: CaseValidationReport,
    ) -> None:
        cid = case.id

        # Required fields
        if not case.title.strip():
            report.issues.append(ValidationIssue(cid, "error", "Missing title."))
        if not case.summary.strip():
            report.issues.append(ValidationIssue(cid, "error", "Missing summary."))
        if not case.category.strip():
            report.issues.append(ValidationIssue(cid, "error", "Missing category."))
        if not case.hidden_truth.strip():
            report.issues.append(ValidationIssue(cid, "warning", "No hidden_truth — weakens training value."))

        # Evidence minimum
        if len(case.evidence) < 3:
            report.issues.append(ValidationIssue(
                cid, "error",
                f"Evidence has {len(case.evidence)} items (minimum 3).",
            ))

        # Scenarios minimum
        if len(case.scenarios) < 2:
            report.issues.append(ValidationIssue(
                cid, "warning",
                f"Scenarios has {len(case.scenarios)} items (recommended ≥2).",
            ))

        # Protocol minimum
        if len(case.analysis_protocol) < 3:
            report.issues.append(ValidationIssue(
                cid, "warning",
                f"Analysis protocol has {len(case.analysis_protocol)} steps (recommended ≥3).",
            ))

        # Contradictions
        if len(case.contradictions) < 1:
            report.issues.append(ValidationIssue(
                cid, "error",
                "No contradictions defined. Every case needs ≥1.",
            ))

        # Forbidden patterns
        if len(case.solution.forbidden_patterns) < 1:
            report.issues.append(ValidationIssue(
                cid, "error",
                "No forbidden patterns defined. Every case needs ≥1.",
            ))

        # Concept rules
        if len(case.solution.required_concept_rules) < 1:
            report.issues.append(ValidationIssue(
                cid, "error",
                "No concept rules defined. Every case needs ≥1.",
            ))

        # Dependency integrity
        for req in case.requires_all:
            if req not in known_insights:
                report.issues.append(ValidationIssue(
                    cid, "error",
                    f"requires_all references unknown insight '{req}'.",
                ))

        # Teach uniqueness
        for teach in case.teaches:
            if teach in taught_by and taught_by[teach] != cid:
                report.issues.append(ValidationIssue(
                    cid, "warning",
                    f"Insight '{teach}' is also taught by {taught_by[teach]}.",
                ))
            taught_by[teach] = cid

        # Solution unlock key
        if not case.solution.unlock_key.strip():
            report.issues.append(ValidationIssue(
                cid, "error", "Solution has no unlock_key.",
            ))

        # False narrative consistency
        if case.solution.must_reject_false_narrative and not case.false_narrative.strip():
            report.issues.append(ValidationIssue(
                cid, "error",
                "must_reject_false_narrative=true but no false_narrative defined.",
            ))

    # ------------------------------------------------------------------
    # circular dependency check
    # ------------------------------------------------------------------

    def _check_circular(
        self,
        cases: list[CaseDefinition],
        known_insights: set[str],
        report: CaseValidationReport,
    ) -> None:
        """Detect circular dependencies in the case DAG."""
        # Build dependency graph: insight_key → set of required insight_keys
        teaches_map: dict[str, str] = {}  # insight_key → case_id
        requires_map: dict[str, list[str]] = {}  # case_id → [insight_keys]

        for case in cases:
            requires_map[case.id] = list(case.requires_all)
            for t in case.teaches:
                teaches_map[t] = case.id

        # Check for cycles via DFS
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(cid: str) -> bool:
            if cid in in_stack:
                return True  # cycle!
            if cid in visited:
                return False
            visited.add(cid)
            in_stack.add(cid)

            for req in requires_map.get(cid, []):
                upstream_cid = teaches_map.get(req)
                if upstream_cid and dfs(upstream_cid):
                    return True

            in_stack.discard(cid)
            return False

        for case in cases:
            if dfs(case.id):
                report.issues.append(ValidationIssue(
                    case.id, "error",
                    "Circular dependency detected in case graph.",
                ))
                break  # one cycle error is enough
