"""Command-line entry point for requirement quality analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyzer import DEFAULT_EXPECTED_ROLES, analyze_requirements, synthesize_with_openai


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze business requirements for quality risks.")
    parser.add_argument("file", type=Path, help="UTF-8 .txt, .md, or .csv requirements file")
    parser.add_argument("--roles", default=", ".join(DEFAULT_EXPECTED_ROLES), help="Expected roles, comma-separated")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print a machine-readable report")
    parser.add_argument("--llm", action="store_true", help="Add an OpenAI grounded narrative using OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    result = analyze_requirements(text, [role.strip() for role in args.roles.split(",")], args.file.name)
    if args.llm:
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error("--llm requires OPENAI_API_KEY")
        result.llm_review = synthesize_with_openai(result, api_key, args.model)

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Reviewed {result.summary['requirement_count']} requirements; found {result.summary['finding_count']} issues.")
        for finding in result.findings:
            ids = f" ({', '.join(finding.requirement_ids)})" if finding.requirement_ids else ""
            print(f"[{finding.severity}] {finding.category}: {finding.title}{ids}")
        if result.llm_review:
            print("\n" + result.llm_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
