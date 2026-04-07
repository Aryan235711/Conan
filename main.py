#!/usr/bin/env python3
"""Detective Reasoning Engine — main entry point.

Usage:
    python main.py                         # interactive (LLM off)
    python main.py --llm deepseek-r1       # interactive + LLM judge via ollama
    python main.py --llm qwen2.5-coder --endpoint http://localhost:1234/v1/chat/completions
    python main.py --list                  # list all cases and exit
    python main.py --graph                 # print insight graph JSON and exit
    python main.py --validate              # validate all case files and exit
    python main.py --profile               # show user skill profile and exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package is importable when running from the Conan/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detective_engine.engine.case_loader import CaseLoader
from detective_engine.engine.case_runner import CaseRunner
from detective_engine.engine.case_validator import CaseQualityValidator
from detective_engine.engine.insight_graph import InsightGraph
from detective_engine.engine.llm_judge import LLMJudge
from detective_engine.engine.reasoning_graph import ReasoningGraphValidator
from detective_engine.engine.user_profile import UserProfile
from detective_engine.engine.validator import Validator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detective Reasoning Engine — Adversarial Intelligence Framework"
    )
    parser.add_argument(
        "--llm",
        metavar="MODEL",
        default=None,
        help="Enable LLM judge with this model name (e.g. deepseek-r1, qwen2.5-coder).",
    )
    parser.add_argument(
        "--endpoint",
        metavar="URL",
        default=None,
        help="OpenAI-compatible endpoint for the LLM (default: use ollama CLI).",
    )
    parser.add_argument(
        "--cases-dir",
        metavar="DIR",
        default=None,
        help="Path to the cases directory (default: detective_engine/cases/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all cases and exit.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print the insight dependency graph as JSON and exit.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all case files against quality standards and exit.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show user skill profile and exit.",
    )
    parser.add_argument(
        "--profile-path",
        metavar="FILE",
        default=None,
        help="Path to the user profile JSON (default: detective_engine/data/profile.json).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="LLM response timeout in seconds (default: 60).",
    )

    args = parser.parse_args()

    # Load cases from JSON
    loader = CaseLoader(cases_dir=args.cases_dir)
    cases, insights = loader.load_all()

    if not cases:
        print("No cases found. Check your cases directory.")
        sys.exit(1)

    # Non-interactive: validate
    if args.validate:
        validator = CaseQualityValidator()
        report = validator.validate_all(cases, insights)
        print(report.summary())
        sys.exit(0 if report.valid else 1)

    # Non-interactive: profile
    if args.profile:
        profile = UserProfile(profile_path=args.profile_path)
        print(profile.render_profile())
        return

    # Build components
    cases_dict = {c.id: c for c in cases}
    insights_dict = {i.key: i for i in insights}

    graph = InsightGraph(cases=cases_dict, insights=insights_dict)
    validator = Validator()
    llm = LLMJudge(
        model=args.llm or "deepseek-r1",
        endpoint=args.endpoint,
        enabled=args.llm is not None,
        timeout=args.timeout,
    )
    reasoning = ReasoningGraphValidator()
    profile = UserProfile(profile_path=args.profile_path)

    runner = CaseRunner(
        graph=graph,
        validator=validator,
        llm_judge=llm,
        reasoning_validator=reasoning,
        user_profile=profile,
    )

    # Non-interactive modes
    if args.list:
        print(graph.render_case_table())
        return
    if args.graph:
        print(graph.export_json())
        return

    # Interactive
    runner.run()


if __name__ == "__main__":
    main()
