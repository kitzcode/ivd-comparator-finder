#!/usr/bin/env python3
"""
IVD Comparator Finder CLI

Commands:
  find     Analyte -> device table (v1)
  ingest   Fetch + parse 510(k) Summary PDFs for devices (v2)
  ask      Grounded Q&A over indexed summaries (v2)
  compare  Structured performance extraction table (v3)
  labs     Reference-lab directory lookup (v3)
  status   Show what has been indexed

Examples:
  python cli.py find "Group A Strep"
  python cli.py ingest --knumbers K173653 K141757
  python cli.py ingest --analyte "Group A Strep" --limit 10
  python cli.py ask "What LoD did K173653 report?" --knumbers K173653
  python cli.py compare --knumbers K173653 K141757 K201269
  python cli.py compare --analyte "Group A Strep" --limit 5
  python cli.py labs "Group A Strep"
  python cli.py labs "Group A Strep" --labs arup mayo
  python cli.py status
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


# ---------------------------------------------------------------------------
# Shared table helpers
# ---------------------------------------------------------------------------

def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = _col_widths(headers, rows)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# find command
# ---------------------------------------------------------------------------

_FIND_HEADERS = ["K-Number", "Device Name", "Applicant", "Decision Date", "Prod Code", "Regulation", "Summary URL"]


def _device_row(dev) -> list[str]:
    return [
        dev.k_number or "—",
        (dev.device_name or "")[:55],
        (dev.applicant_name or "")[:30],
        str(dev.decision_date) if dev.decision_date else "—",
        dev.product_code or "—",
        dev.regulation_number or "—",
        dev.summary_url or "—",
    ]


def cmd_find(args: argparse.Namespace) -> None:
    from finder.pipeline import find_devices

    extra = [s.strip() for s in args.synonyms.split(",") if s.strip()] if args.synonyms else None

    print(f"\nSearching for: {args.analyte!r}")
    resolution, devices = find_devices(
        args.analyte,
        extra_synonyms=extra,
        resolve_urls=args.urls,
        medical_specialty=args.specialty or None,
    )

    print(f"\nSynonyms used: {', '.join(resolution.synonyms_used)}")
    print(f"Product codes found: {', '.join(p.product_code for p in resolution.product_codes)}")
    print(f"\n*** {resolution.note} ***\n")

    if not devices:
        print("No devices found.")
        return

    output = devices[: args.limit] if args.limit else devices
    print_table(_FIND_HEADERS, [_device_row(d) for d in output])
    print(f"\n{len(output)} device(s) shown (of {len(devices)} total).")


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    from finder.pipeline import find_devices, ingest_summaries
    from finder.sources.summaries import fetch_summary_pdf
    from finder.parse.pdf import extract_pdf
    from finder.parse.sections import chunk_pdf
    from finder.index.store import store_chunks, is_indexed
    from finder.sources.summaries import resolve_summary_url

    devices = []

    if args.knumbers:
        # Build minimal Device stubs from K-numbers
        from finder.models import Device
        from finder.sources.openfda import get_510k_by_knumber
        for k in args.knumbers:
            rec = get_510k_by_knumber(k)
            if rec:
                devices.append(Device(
                    k_number=k,
                    device_name=rec.get("device_name", ""),
                    applicant_name=rec.get("applicant_name", ""),
                    product_code=rec.get("product_code", ""),
                ))
            else:
                print(f"WARNING: {k} not found in openFDA — skipping")

    elif args.analyte:
        _, devices = find_devices(args.analyte)
        if args.limit:
            devices = devices[: args.limit]
        print(f"Found {len(devices)} devices for {args.analyte!r}.")
    else:
        print("ERROR: provide --knumbers or --analyte", file=sys.stderr)
        sys.exit(1)

    if not devices:
        print("No devices to ingest.")
        return

    print(f"\nIngesting {len(devices)} device(s) …\n")
    results = ingest_summaries(devices, progress_cb=print, skip_already_indexed=not args.force)

    ok = sum(1 for r in results if r.status == "ok")
    no_summary = sum(1 for r in results if r.status == "no_summary")
    image_only = sum(1 for r in results if r.status == "image_only")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.note == "already indexed")

    print(f"\nDone. ok={ok}  no_summary={no_summary}  image_only={image_only}  error={errors}  skipped={skipped}")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    from finder.extract import extract_performance, format_performance_table
    from finder.pipeline import find_devices
    from finder.index.store import list_indexed

    if args.knumbers:
        k_numbers = args.knumbers
        # Collect device/product-code labels from the index manifest
        indexed = list_indexed()
        device_names: dict[str, str] = {}
        product_codes: dict[str, str] = {}
        from finder.sources.openfda import get_510k_by_knumber
        for k in k_numbers:
            rec = get_510k_by_knumber(k)
            if rec:
                device_names[k] = rec.get("device_name", "")
                product_codes[k] = rec.get("product_code", "")
    elif args.analyte:
        _, devices = find_devices(args.analyte)
        if args.limit:
            devices = devices[: args.limit]
        # Only compare indexed devices
        indexed = list_indexed()
        devices = [d for d in devices if d.k_number in indexed]
        k_numbers = [d.k_number for d in devices]
        device_names = {d.k_number: d.device_name for d in devices}
        product_codes = {d.k_number: d.product_code for d in devices}
        if not k_numbers:
            print("No indexed devices found. Run: python cli.py ingest --analyte '...'")
            return
    else:
        print("ERROR: provide --knumbers or --analyte", file=sys.stderr)
        sys.exit(1)

    llm = None
    if args.model:
        llm = _make_anthropic_llm(args.model)

    table = extract_performance(k_numbers, device_names=device_names, product_codes=product_codes, llm=llm)
    print(format_performance_table(table, verbose=args.verbose))


def cmd_labs(args: argparse.Namespace) -> None:
    from finder.sources.labs import find_reference_labs, format_lab_results
    labs_filter = args.labs or None
    tests = find_reference_labs(args.analyte, labs=labs_filter)
    print(format_lab_results(tests))


def cmd_ask(args: argparse.Namespace) -> None:
    from finder.qa import ask, format_answer

    llm = None
    if args.model:
        llm = _make_anthropic_llm(args.model)

    answer = ask(
        args.question,
        k_numbers=args.knumbers or None,
        product_codes=args.product_codes or None,
        top_k=args.top_k,
        sections=args.sections or None,
        llm=llm,
    )
    print(format_answer(answer))


def _make_anthropic_llm(model: str):
    """Return a simple Anthropic SDK callable for use with qa.ask()."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic to use --model", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    def call(system_prompt: str, user_prompt: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    return call


def cmd_status(args: argparse.Namespace) -> None:
    from finder.index.store import list_indexed, load_chunks

    manifest = list_indexed()
    if not manifest:
        print("Nothing indexed yet. Run: python cli.py ingest --analyte 'Group A Strep'")
        return

    rows = []
    for k, status in sorted(manifest.items()):
        count = len(load_chunks(k)) if status == "ok" else 0
        rows.append([k, status, str(count)])

    print_table(["K-Number", "Status", "Chunks"], rows)
    print(f"\n{len(manifest)} device(s) in index.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IVD Comparator Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # find
    p_find = sub.add_parser("find", help="Analyte -> device table")
    p_find.add_argument("analyte", help="Analyte or assay name")
    p_find.add_argument("--synonyms", default="", help="Comma-separated extra synonyms")
    p_find.add_argument("--urls", action="store_true", help="Resolve Summary PDF URLs (slow)")
    p_find.add_argument("--specialty", default="", help="Filter by medical specialty description")
    p_find.add_argument("--limit", type=int, default=0, help="Limit output rows (0=all)")

    # compare
    p_compare = sub.add_parser("compare", help="Structured performance extraction table")
    grp_c = p_compare.add_mutually_exclusive_group(required=True)
    grp_c.add_argument("--knumbers", nargs="+", metavar="K")
    grp_c.add_argument("--analyte", help="Extract for all indexed devices matching analyte")
    p_compare.add_argument("--limit", type=int, default=0)
    p_compare.add_argument("--model", default="", help="Anthropic model for LLM-assisted extraction")
    p_compare.add_argument("--verbose", action="store_true", help="Show source URLs in output")

    # labs
    p_labs = sub.add_parser("labs", help="Reference-lab directory lookup")
    p_labs.add_argument("analyte", help="Analyte name to search for")
    p_labs.add_argument("--labs", nargs="+", choices=["arup", "mayo"], help="Which labs to query")

    # ask
    p_ask = sub.add_parser("ask", help="Grounded Q&A over indexed summaries")
    p_ask.add_argument("question", help="Question to answer")
    p_ask.add_argument("--knumbers", nargs="+", metavar="K", help="Scope to these K-numbers")
    p_ask.add_argument("--product-codes", nargs="+", metavar="PC", dest="product_codes")
    p_ask.add_argument("--sections", nargs="+", help="Restrict to section names")
    p_ask.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_ask.add_argument(
        "--model",
        default="",
        help="Anthropic model ID for LLM-backed answers (e.g. claude-haiku-4-5-20251001). "
             "Omit for keyword-only mode (returns top chunk verbatim).",
    )

    # ingest
    p_ingest = sub.add_parser("ingest", help="Fetch + parse 510(k) Summary PDFs")
    grp = p_ingest.add_mutually_exclusive_group(required=True)
    grp.add_argument("--knumbers", nargs="+", metavar="K", help="Specific K-numbers to ingest")
    grp.add_argument("--analyte", help="Ingest summaries for all devices matching analyte")
    p_ingest.add_argument("--limit", type=int, default=0, help="Limit devices when using --analyte")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if already indexed")

    # status
    sub.add_parser("status", help="Show index status")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "labs":
        cmd_labs(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "find":
        cmd_find(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
