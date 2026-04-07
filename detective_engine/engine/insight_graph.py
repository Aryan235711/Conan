"""Insight dependency graph — DAG with topological ordering, multiple paths,
and ASCII visualisation.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict

from .models import CaseDefinition, Insight


class InsightGraph:
    """Manages the insight inventory and case-unlock dependencies as a DAG."""

    def __init__(
        self,
        cases: dict[str, CaseDefinition],
        insights: dict[str, Insight],
    ):
        self.cases = cases
        self.insights = insights
        self.unlocked: set[str] = set()

        # Pre-compute DAG edges: insight_key → set of case_ids that teach it
        self._taught_by: dict[str, str] = {}
        for c in cases.values():
            for key in c.teaches:
                self._taught_by[key] = c.id

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def is_unlocked(self, case_id: str) -> bool:
        return all(r in self.unlocked for r in self.cases[case_id].requires_all)

    def missing_for(self, case_id: str) -> list[str]:
        return [r for r in self.cases[case_id].requires_all if r not in self.unlocked]

    def available_cases(self) -> list[CaseDefinition]:
        return [c for c in self.cases.values() if self.is_unlocked(c.id)]

    def topological_order(self) -> list[str]:
        """Return case IDs in a valid topological order (Kahn's algorithm)."""
        # Build adjacency: parent_case → child_cases
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {cid: 0 for cid in self.cases}

        for cid, c in self.cases.items():
            for req_key in c.requires_all:
                parent = self._taught_by.get(req_key)
                if parent and parent in self.cases:
                    adj[parent].append(cid)
                    in_degree[cid] += 1

        queue: deque[str] = deque(
            cid for cid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def all_paths(self, start: str, end: str) -> list[list[str]]:
        """Find all paths from *start* case to *end* case in the DAG."""
        adj: dict[str, list[str]] = defaultdict(list)
        for cid, c in self.cases.items():
            for req_key in c.requires_all:
                parent = self._taught_by.get(req_key)
                if parent and parent in self.cases:
                    adj[parent].append(cid)

        paths: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            if node == end:
                paths.append(list(path))
                return
            for child in adj.get(node, []):
                if child not in path:
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(start, [start])
        return paths

    def parallel_branches(self) -> list[list[str]]:
        """Identify groups of cases that can be solved in parallel."""
        topo = self.topological_order()
        # Assign depth (longest path from any root)
        depth: dict[str, int] = {}
        for cid in topo:
            c = self.cases[cid]
            if not c.requires_all:
                depth[cid] = 0
            else:
                max_parent_depth = 0
                for req_key in c.requires_all:
                    parent = self._taught_by.get(req_key)
                    if parent and parent in depth:
                        max_parent_depth = max(max_parent_depth, depth[parent])
                depth[cid] = max_parent_depth + 1

        # Group by depth
        layers: dict[int, list[str]] = defaultdict(list)
        for cid, d in depth.items():
            layers[d].append(cid)
        return [layers[d] for d in sorted(layers)]

    # ------------------------------------------------------------------
    # mutations
    # ------------------------------------------------------------------

    def unlock(self, key: str) -> list[str]:
        """Unlock an insight and return list of newly-unlocked case IDs."""
        self.unlocked.add(key)
        newly_unlocked: list[str] = []
        for cid, c in self.cases.items():
            if c.requires_all and all(r in self.unlocked for r in c.requires_all):
                was_locked = any(r not in (self.unlocked - {key}) for r in c.requires_all)
                if was_locked:
                    newly_unlocked.append(cid)
        return newly_unlocked

    # ------------------------------------------------------------------
    # visualisation
    # ------------------------------------------------------------------

    def render_dag(self) -> str:
        """Render an ASCII DAG showing insight flow and case dependencies."""
        branches = self.parallel_branches()
        lines: list[str] = ["INSIGHT DAG:"]

        for layer_idx, layer in enumerate(branches):
            if layer_idx > 0:
                # Draw arrows from previous layer
                prev = branches[layer_idx - 1]
                for cid in layer:
                    parents: list[str] = []
                    for req in self.cases[cid].requires_all:
                        p = self._taught_by.get(req)
                        if p:
                            parents.append(p)
                    arrows = ", ".join(parents)
                    lines.append(f"       {'│' * len(prev)}")
                    lines.append(f"       └─── {arrows} ──▶")

            for cid in layer:
                c = self.cases[cid]
                status = "✓" if self.is_unlocked(cid) else "✗"
                teaches = ", ".join(c.teaches)
                reqs = ", ".join(c.requires_all) if c.requires_all else "—"
                lines.append(
                    f"  [{status}] {cid}: {c.title}"
                    f"  (teaches: {teaches} | needs: {reqs})"
                )

        return "\n".join(lines)

    def render_case_table(self) -> str:
        topo = self.topological_order()
        branches = self.parallel_branches()
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              CASE BOARD — INSIGHT DAG                   ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        for layer_idx, layer in enumerate(branches):
            if layer_idx > 0:
                lines.append("║  ────────── ↓ ──────────                                ║")
            parallel_tag = " (parallel)" if len(layer) > 1 else ""
            lines.append(f"║  Layer {layer_idx}{parallel_tag}:")
            for cid in layer:
                c = self.cases[cid]
                status = "🔓" if self.is_unlocked(cid) else "🔒"
                reqs = ", ".join(c.requires_all) or "none"
                meta = ""
                if c.contradictions:
                    meta += f" | {len(c.contradictions)} contradiction(s)"
                if c.solution.forbidden_patterns:
                    meta += " | has traps"
                if c.solution.insight_usage_rules:
                    meta += f" | {len(c.solution.insight_usage_rules)} insight check(s)"
                lines.append(
                    f"║    {status} {cid}: {c.title}"
                )
                lines.append(
                    f"║       {c.category} | requires: {reqs}{meta}"
                )

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export_json(self) -> str:
        topo = self.topological_order()
        branches = self.parallel_branches()
        return json.dumps({
            "insights": [asdict(i) for i in self.insights.values()],
            "unlocked": sorted(self.unlocked),
            "topological_order": topo,
            "parallel_layers": branches,
            "cases": [
                {
                    "id": c.id,
                    "title": c.title,
                    "requires_all": c.requires_all,
                    "teaches": c.teaches,
                    "false_narrative": c.false_narrative,
                    "hidden_truth": c.hidden_truth,
                    "num_contradictions": len(c.contradictions),
                    "num_forbidden": len(c.solution.forbidden_patterns),
                    "num_insight_checks": len(c.solution.insight_usage_rules),
                    "must_reject_fn": c.solution.must_reject_false_narrative,
                }
                for c in self.cases.values()
            ],
        }, indent=2)
