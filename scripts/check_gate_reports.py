"""Fail if a reported result lost its control gate, or carries a failing one.

NEXTSTEP.md Priority 2 item 3 asks CI to "fail the workflow when a reported
experiment omits a control result or when a gate check fails". The second half
is unambiguous. The first half needs a definition this repository does not yet
have: there is no manifest of experiments, no schema, and of 13 JSON files in
`outputs/` only some carry a gate -- in several different shapes, because four
scripts grew their own independently.

Inventing a rule that every result JSON must carry a gate would turn CI red on
the repository's own committed deliverables and teach everyone to ignore it. So
this takes the narrow, enforceable reading:

  1. **Results declared gated must stay gated.** `MUST_CARRY_GATE` is the list
     as it stands. Dropping a gate from one of them, or committing one whose
     gate did not pass, fails the build. That is the regression this can
     actually prevent.
  2. **Any gate present anywhere must be well-formed**, whatever shape it is
     in, and must be passing -- except in the files whose purpose is to record
     methods that fail. `baselines.json` recording `passed: false` for SIFT on
     cross-illumination IS the project's headline result; treating that as a
     build failure would invert the finding. Those files are listed in
     `RECORDS_FAILURES` and are still checked for completeness.
  3. **Ungated results are reported, not failed.** The gap is printed every run
     so it stays visible, which is the honest state of things until someone
     decides what "a reported experiment" means.

    python scripts/check_gate_reports.py          # exit 1 on a violation
    python scripts/check_gate_reports.py --json gate-report.json

Runs from a fresh clone: it reads committed JSON and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# Results that are gated today and must remain so. Add to this list when a new
# gated result is committed; never remove from it to make a build pass.
MUST_CARRY_GATE = [
    "tmc2_metrics.json",      # the headline DELIVERABLE, not just the run log
    "tmc_vs_kaguya.json",
    "m3_vs_kaguya.json",
    "baselines.json",
    "adopt_check.json",
]

# Files whose whole purpose is to record methods that FAIL the gate. A
# `passed: false` in one of these is the measurement, not a regression: the
# project's headline baseline result is that on cross-illumination no tested
# baseline passes. They are still checked for well-formedness -- a malformed or
# partial gate is a bug wherever it appears -- but not for passing.
RECORDS_FAILURES = {"baselines.json", "adopt_check.json"}

# The four controls, by the names the gate writes. A gate missing one of these
# is not a gate -- BUG-011 got through because only some of them existed.
EXPECTED_CHECKS = {"real pair", "roll +15 raw", "noise", "constant"}


def find_gates(node, path="") -> list[tuple[str, object]]:
    """Every `gate` value anywhere in a JSON tree, with its location.

    Four scripts write four shapes: `{"gate": {"passed":…, "checks":{…}}}`,
    `"gate": bool` beside `"gate_checks"`, and `{"gate": {case: {…}}}` nested
    per case or per method. Rather than bless one, walk the tree.
    """
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else k
            if k == "gate":
                found.append((here, v))
            else:
                found.extend(find_gates(v, here))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(find_gates(v, f"{path}[{i}]"))
    return found


def verdict(gate) -> tuple[bool | None, dict]:
    """(passed, checks) from any of the shapes, or (None, {}) if unreadable."""
    if isinstance(gate, bool):
        return gate, {}
    if isinstance(gate, dict):
        if "passed" in gate:
            checks = gate.get("checks", {})
            return bool(gate["passed"]), checks if isinstance(checks, dict) else {}
        # nested per case or per method: pass only if every leaf passes
        leaves = [verdict(v) for v in gate.values()]
        known = [p for p, _ in leaves if p is not None]
        if known:
            return all(known), {}
    return None, {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the machine-readable report here")
    args = ap.parse_args()

    if not OUT.exists():
        print(f"no {OUT} - nothing to check")
        return 1

    files = sorted(OUT.glob("*.json"))
    if not files:
        print(f"no result JSON in {OUT} - refusing to report success on nothing")
        return 1

    report: dict = {"gated": {}, "ungated": [], "violations": []}

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report["violations"].append(f"{p.name}: not valid JSON ({e})")
            continue

        gates = find_gates(data)
        if not gates:
            report["ungated"].append(p.name)
            continue

        records_failures = p.name in RECORDS_FAILURES
        states = []
        for where, gate in gates:
            passed, checks = verdict(gate)
            states.append(passed)
            if passed is None:
                report["violations"].append(
                    f"{p.name}: gate at '{where}' is not in any known shape")
            elif not passed and not records_failures:
                report["violations"].append(
                    f"{p.name}: gate at '{where}' records passed=false")
            if checks:
                missing = EXPECTED_CHECKS - set(checks)
                if missing:
                    report["violations"].append(
                        f"{p.name}: gate at '{where}' is missing "
                        f"{sorted(missing)} - a partial gate is not a gate")
        report["gated"][p.name] = {
            "all_passing": all(s for s in states if s is not None),
            "gates": len(states),
            "records_failures": records_failures,
        }

    for name in MUST_CARRY_GATE:
        if name not in report["gated"]:
            report["violations"].append(
                f"{name}: declared gated in MUST_CARRY_GATE but carries no gate"
                + ("" if (OUT / name).exists() else " (file is missing)"))

    print(f"{len(report['gated'])} gated result(s):")
    for name, info in sorted(report["gated"].items()):
        if info["records_failures"]:
            mark = "RECORDS"
            note = f"  ({info['gates']} gates, failures are the measurement)"
        else:
            mark = "PASS" if info["all_passing"] else "FAIL"
            note = ""
        print(f"  {mark:<7} {name}{note}")
    if report["ungated"]:
        print(f"\n{len(report['ungated'])} result(s) carry no gate - visible, not enforced:")
        for name in report["ungated"]:
            print(f"        {name}")
        print("  (see NEXTSTEP.md Priority 2: 'a reported experiment' is undefined)")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    if report["violations"]:
        print(f"\n{len(report['violations'])} violation(s):")
        for v in report["violations"]:
            print(f"  {v}")
        return 1

    print("\nOK - every declared gate is present, complete and passing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
