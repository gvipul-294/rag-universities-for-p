from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .ingest import build_index, load_registry
from .models import CandidateProfile
from .retrieve import answer, profile_to_question, search
from .settings import get_settings


def _print_sources(results: object) -> None:
    print("\nRetrieved official sources:")
    for number, result in enumerate(results, start=1):  # type: ignore[arg-type]
        record = result.record
        print(f"[S{number}] {record['university']} ({record['country']}) — {record['url']} [{result.score:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a source-backed PhD RAG index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sources", help="List the curated official source registry.")
    ingest = subparsers.add_parser("ingest", help="Fetch official pages and rebuild the FAISS index.")
    ingest.add_argument("--country", action="append", help="Country to ingest; repeat to select several.")
    ingest.add_argument("--limit", type=int, help="Limit sources for a quick smoke test.")
    ingest.add_argument("--delay", type=float, default=0.4, help="Seconds between page requests (default: 0.4).")
    ask = subparsers.add_parser("ask", help="Ask a question against the local index.")
    ask.add_argument("question")
    ask.add_argument("--country", action="append", help="Limit retrieval to a country; repeat to select several.")
    ask.add_argument("--top-k", type=int, default=12)
    recommend = subparsers.add_parser("recommend", help="Recommend research leads from a profile JSON file.")
    recommend.add_argument("--profile", required=True, type=Path)
    recommend.add_argument("--top-k", type=int, default=12)
    agent = subparsers.add_parser(
        "agent",
        help="Use an Ollama agent that calls the local RAG MCP tools.",
    )
    agent.add_argument("question", nargs="?", help="Question for the local RAG agent.")
    agent.add_argument("--profile", type=Path, help="Candidate profile JSON; cannot be combined with a question.")
    agent.add_argument("--country", action="append", help="Limit retrieval to a country; repeat to select several.")
    agent.add_argument("--top-k", type=int, default=12)
    agent.add_argument("--model", help="Ollama model override, e.g. mistral.")
    args = parser.parse_args()
    # Progress goes to stderr; command output (answers, manifest) stays on stdout.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("phd_rag").setLevel(logging.INFO)
    settings = get_settings()

    if args.command == "sources":
        sources = load_registry(settings.registry_path)
        for item in sources:
            print(f"{item.country:15} | {item.university:45} | {item.url}")
        print(f"\n{len(sources)} official pages in registry")
    elif args.command == "ingest":
        manifest = build_index(settings, set(args.country) if args.country else None, args.limit, args.delay)
        print(json.dumps(manifest, indent=2))
    elif args.command == "ask":
        reply, results = answer(settings, args.question, args.top_k, countries=args.country or None)
        print(reply)
        _print_sources(results)
    elif args.command == "recommend":
        profile = CandidateProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
        reply, results = answer(
            settings,
            profile_to_question(profile),
            args.top_k,
            countries=profile.preferred_countries or None,
        )
        print(reply)
        _print_sources(results)
    elif args.command == "agent":
        if bool(args.question) == bool(args.profile):
            parser.error("agent requires exactly one of a question or --profile PROFILE")
        if args.profile:
            profile = CandidateProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
            question = profile_to_question(profile)
            countries = profile.preferred_countries or None
            candidate_profile_json = args.profile.read_text(encoding="utf-8")
        else:
            question = args.question
            countries = args.country or None
            candidate_profile_json = None
        # Imported only for this command so ordinary RAG commands do not require
        # the optional MCP agent runtime to be loaded at startup.
        from .agentic import run_agentic_matcher

        outcome = run_agentic_matcher(
            settings,
            question,
            args.top_k,
            countries=countries,
            model=args.model,
            candidate_profile_json=candidate_profile_json,
        )
        print(f"Agentic RAG matcher (Ollama model: {outcome.model})")
        for event in outcome.trace:
            print(event)
        print(f"\n{outcome.answer}")
        _print_sources(outcome.sources)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
