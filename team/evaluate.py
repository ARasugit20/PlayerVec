"""Evaluate team fingerprints and matchup diagnosis quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from team.diagnose import FixtureDiagnostician
from team.fingerprint import STYLE_DIMENSIONS, TeamFingerprintEngine

DOCS_DIR = Path(__file__).parent.parent / "docs"


def fingerprint_sanity_checks(engine: TeamFingerprintEngine) -> list[dict]:
    """Qualitative checks that team DNA reflects known style differences."""
    fps = engine.all_fingerprints()
    results = []

    if len(fps) < 2:
        return [{"check": "squad_count", "pass": False, "detail": "Need >= 2 teams"}]

    all_press = {t: fp.style_dna.get("press_intensity", 0) for t, fp in fps.items()}
    all_finish = {t: fp.style_dna.get("finishing", 0) for t, fp in fps.items()}

    press_spread = max(all_press.values()) - min(all_press.values())
    finish_spread = max(all_finish.values()) - min(all_finish.values())

    results.append(
        {
            "check": "press_intensity_variance",
            "pass": bool(press_spread > 0.05),
            "detail": f"Press DNA spread across teams: {press_spread:.3f}",
        }
    )
    results.append(
        {
            "check": "finishing_variance",
            "pass": bool(finish_spread > 0.05),
            "detail": f"Finishing DNA spread across teams: {finish_spread:.3f}",
        }
    )

    archetype_diversity = []
    for t, fp in fps.items():
        n_archetypes = sum(1 for v in fp.archetype_mix.values() if v > 0.05)
        archetype_diversity.append(n_archetypes)
    avg_div = np.mean(archetype_diversity)
    results.append(
        {
            "check": "archetype_diversity",
            "pass": bool(avg_div >= 2),
            "detail": f"Avg archetype pockets per squad: {avg_div:.1f}",
        }
    )
    return results


def matchup_coverage(diagnostician: FixtureDiagnostician) -> dict:
    engine = diagnostician.engine
    teams = engine.all_teams()
    n_briefs = 0
    n_with_gaps = 0
    n_with_adjustments = 0

    for i, ta in enumerate(teams):
        for tb in teams[i + 1 :]:
            brief = diagnostician.diagnose(ta, tb)
            n_briefs += 1
            if brief.structural_gaps:
                n_with_gaps += 1
            if brief.adjustments:
                n_with_adjustments += 1

    return {
        "total_matchups": n_briefs,
        "matchups_with_structural_gaps": n_with_gaps,
        "matchups_with_adjustments": n_with_adjustments,
        "gap_rate": round(n_with_gaps / max(n_briefs, 1), 3),
    }


def run_evaluation() -> dict:
    engine = TeamFingerprintEngine()
    diagnostician = FixtureDiagnostician(engine)

    sanity = fingerprint_sanity_checks(engine)
    coverage = matchup_coverage(diagnostician)

    return {
        "teams_analyzed": len(engine.all_teams()),
        "style_dimensions": list(STYLE_DIMENSIONS.keys()),
        "fingerprint_sanity": sanity,
        "matchup_coverage": coverage,
        "honest_limits": [
            "Team fingerprints are minute-weighted aggregates — not formation-aware.",
            "Adjustment cards are scouting hypotheses, not proven tactical prescriptions.",
            "Outcome prediction requires match-level event data and supervised modeling (future work).",
        ],
        "future_outcome_modeling": {
            "features": "style_gap_vector + archetype_mix_diff + home_away",
            "target": "win_probability or xG_differential",
            "data_needed": "StatsBomb match results + lineups",
            "approach": "Logistic regression baseline on historical matchups with similar style gaps",
        },
    }


def write_evaluation_md(report: dict, path: Path = DOCS_DIR / "RESEARCH.md") -> None:
    existing = path.read_text() if path.exists() else ""
    if "## Matchup Diagnosis Layer" in existing:
        base = existing.split("## Matchup Diagnosis Layer")[0].rstrip()
    else:
        base = existing.rstrip()

    sanity_rows = "\n".join(
        f"| {s['check']} | {'PASS' if s['pass'] else 'FAIL'} | {s['detail']} |"
        for s in report["fingerprint_sanity"]
    )
    cov = report["matchup_coverage"]

    section = f"""

## Matchup Diagnosis Layer

Team fingerprints aggregate minute-weighted player embeddings into interpretable style DNA.
Fixture briefs compare Team A vs Team B and surface structural gaps and adjustment cards.

### Fingerprint Sanity Checks

| Check | Result | Detail |
|-------|--------|--------|
{sanity_rows}

### Matchup Coverage

- Teams analyzed: {report['teams_analyzed']}
- Total pairwise matchups: {cov['total_matchups']}
- Matchups with structural gaps: {cov['matchups_with_structural_gaps']} ({cov['gap_rate']:.0%})
- Matchups with adjustment cards: {cov['matchups_with_adjustments']}

### What we claim (and don't)

| Claim | Supported? |
|-------|------------|
| "These teams play different styles" | Yes — style DNA + archetype mix |
| "You're missing archetype Y" | Yes — structural gap on embeddings |
| "Player Z is best stylistic counter" | Yes — wildcard picks via cosine distance |
| "Do X tactic and you win" | No — needs match events + outcomes |
| "Guaranteed tournament path" | No — too many confounders |

### Future: Outcome Modeling (optional extension)

- **Features:** {report['future_outcome_modeling']['features']}
- **Target:** {report['future_outcome_modeling']['target']}
- **Data:** {report['future_outcome_modeling']['data_needed']}
- **Approach:** {report['future_outcome_modeling']['approach']}

```bash
python -m team.evaluate
python -m team.diagnose  # via API: GET /fixture-brief?team_a=&team_b=
```
"""
    path.write_text(base + section)
    print(f"Updated {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report")
    args = parser.parse_args()

    report = run_evaluation()
    print(json.dumps(report, indent=2))
    write_evaluation_md(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
