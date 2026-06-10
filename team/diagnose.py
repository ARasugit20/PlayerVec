"""Fixture diagnosis: team style gaps and adjustment recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from team.fingerprint import STYLE_DIMENSIONS, TeamFingerprint, TeamFingerprintEngine

COSINE_GAP_THRESHOLD = 0.15


@dataclass
class StyleGap:
    dimension: str
    team_a_value: float
    team_b_value: float
    delta_pct: float
    description: str


@dataclass
class StructuralGap:
    archetype: str
    opponent_share: float
    your_share: float
    description: str


@dataclass
class AdjustmentCard:
    category: str
    title: str
    detail: str
    suggested_players: list[str] = field(default_factory=list)


@dataclass
class WildcardPick:
    player: str
    position: str
    archetype: str
    fills_gap: str
    similarity_to_opponent: float


@dataclass
class FixtureBrief:
    team_a: str
    team_b: str
    team_a_fingerprint: dict
    team_b_fingerprint: dict
    style_clashes: list[dict]
    structural_gaps: list[dict]
    exploit_vectors: list[str]
    adjustments: list[dict]
    wildcard_picks: list[dict]
    summary: str

    def to_dict(self) -> dict:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "team_a_fingerprint": self.team_a_fingerprint,
            "team_b_fingerprint": self.team_b_fingerprint,
            "style_clashes": self.style_clashes,
            "structural_gaps": self.structural_gaps,
            "exploit_vectors": self.exploit_vectors,
            "adjustments": self.adjustments,
            "wildcard_picks": self.wildcard_picks,
            "summary": self.summary,
        }


class FixtureDiagnostician:
    def __init__(self, engine: TeamFingerprintEngine | None = None):
        self.engine = engine or TeamFingerprintEngine()

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _style_clashes(self, fp_a: TeamFingerprint, fp_b: TeamFingerprint) -> list[StyleGap]:
        gaps = []
        dim_labels = {
            "press_intensity": "press intensity",
            "progression": "progressive play",
            "aerial_direct": "aerial/direct play",
            "chance_creation": "chance creation",
            "finishing": "finishing threat",
            "width": "width and carries",
        }
        for dim in STYLE_DIMENSIONS:
            va = fp_a.style_dna.get(dim, 0.0)
            vb = fp_b.style_dna.get(dim, 0.0)
            base = max(va, vb, 0.01)
            delta_pct = ((vb - va) / base) * 100
            if abs(delta_pct) < 8:
                continue
            label = dim_labels.get(dim, dim)
            if delta_pct > 0:
                desc = f"{fp_b.team}'s {label} is {abs(delta_pct):.0f}% stronger than {fp_a.team}'s."
            else:
                desc = f"{fp_a.team}'s {label} is {abs(delta_pct):.0f}% stronger than {fp_b.team}'s."
            gaps.append(StyleGap(dim, va, vb, round(delta_pct, 1), desc))
        return sorted(gaps, key=lambda g: abs(g.delta_pct), reverse=True)[:5]

    def _structural_gaps(self, fp_a: TeamFingerprint, fp_b: TeamFingerprint) -> list[StructuralGap]:
        gaps = []
        all_archetypes = set(fp_a.archetype_mix) | set(fp_b.archetype_mix)
        for archetype in all_archetypes:
            share_a = fp_a.archetype_mix.get(archetype, 0.0)
            share_b = fp_b.archetype_mix.get(archetype, 0.0)
            if share_b > share_a + 0.12:
                gaps.append(
                    StructuralGap(
                        archetype=archetype,
                        opponent_share=share_b,
                        your_share=share_a,
                        description=(
                            f"{fp_b.team} has {share_b:.0%} of squad minutes in '{archetype}' "
                            f"vs your {share_a:.0%} — structural gap."
                        ),
                    )
                )
        return sorted(gaps, key=lambda g: g.opponent_share - g.your_share, reverse=True)[:4]

    def _exploit_vectors(
        self, fp_a: TeamFingerprint, fp_b: TeamFingerprint, clashes: list[StyleGap]
    ) -> list[str]:
        vectors = []
        for gap in clashes[:3]:
            if gap.delta_pct > 15:
                vectors.append(
                    f"Opponent edge in {gap.dimension.replace('_', ' ')}: "
                    f"they may control phases where this style dominates."
                )
            elif gap.delta_pct < -15:
                vectors.append(
                    f"Your edge in {gap.dimension.replace('_', ' ')}: "
                    f"lean into this mismatch early."
                )

        b_progressors = fp_b.archetype_mix.get("deep_progressor", 0)
        a_press = fp_a.archetype_mix.get("high_presser", 0)
        if b_progressors > 0.2 and a_press < 0.15:
            vectors.append(
                f"Their midfield is progressor-heavy ({b_progressors:.0%}); "
                f"your press cluster is thin ({a_press:.0%}) — risk of midfield bypass."
            )

        b_wide = fp_b.archetype_mix.get("wide_carrier", 0)
        a_press = fp_a.style_dna.get("press_intensity", 0)
        if b_wide > 0.15 and a_press < np.percentile([fp_a.style_dna.get("press_intensity", 0)], 50):
            vectors.append(
                "Their wide carriers may stretch a back line that doesn't press aggressively wide."
            )

        return vectors[:4]

    def _find_opponent_anchor(self, team_b: str, archetype: str):
        players = self.engine.team_players(team_b)
        candidates = [p for p in players if p.archetype == archetype]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.minutes)

    def _wildcard_picks(
        self, team_a: str, team_b: str, structural_gaps: list[StructuralGap]
    ) -> list[WildcardPick]:
        picks = []
        players_a = self.engine.team_players(team_a)

        for gap in structural_gaps[:3]:
            anchor = self._find_opponent_anchor(team_b, gap.archetype)
            if anchor is None:
                continue
            best = None
            best_dist = float("inf")
            for p in players_a:
                if p.archetype == gap.archetype:
                    continue
                dist = self._cosine_distance(p.embedding, anchor.embedding)
                if dist < best_dist:
                    best_dist = dist
                    best = p
            if best and best_dist < 0.5:
                picks.append(
                    WildcardPick(
                        player=best.player,
                        position=best.position,
                        archetype=best.archetype,
                        fills_gap=gap.archetype,
                        similarity_to_opponent=round(1 - best_dist, 3),
                    )
                )
        return picks[:3]

    def _adjustments(
        self,
        team_a: str,
        team_b: str,
        clashes: list[StyleGap],
        structural_gaps: list[StructuralGap],
        wildcards: list[WildcardPick],
    ) -> list[AdjustmentCard]:
        cards = []

        for gap in clashes:
            if gap.delta_pct > 20:
                cards.append(
                    AdjustmentCard(
                        category="style_lever",
                        title=f"Neutralize their {gap.dimension.replace('_', ' ')}",
                        detail=(
                            f"Opponent leads by {gap.delta_pct:.0f}% on {gap.dimension.replace('_', ' ')}. "
                            f"Consider a second presser or deeper block to cut supply."
                        ),
                    )
                )
            elif gap.delta_pct < -20:
                cards.append(
                    AdjustmentCard(
                        category="style_lever",
                        title=f"Lean into your {gap.dimension.replace('_', ' ')}",
                        detail=(
                            f"You lead by {abs(gap.delta_pct):.0f}% — push this advantage "
                            f"in transitions and set-piece moments."
                        ),
                    )
                )

        for sg in structural_gaps[:2]:
            anchor = self._find_opponent_anchor(team_b, sg.archetype)
            wildcard_names = [w.player for w in wildcards if w.fills_gap == sg.archetype]
            detail = f"No minute-weighted '{sg.archetype}' pocket to match theirs."
            if anchor:
                detail += f" Their anchor: {anchor.player} ({anchor.minutes:.0f} min)."
            if wildcard_names:
                detail += f" Closest fit on your squad: {', '.join(wildcard_names)}."
            cards.append(
                AdjustmentCard(
                    category="lineup",
                    title=f"Fill the '{sg.archetype}' gap",
                    detail=detail,
                    suggested_players=wildcard_names,
                )
            )

        if not cards:
            cards.append(
                AdjustmentCard(
                    category="general",
                    title="Balanced stylistic matchup",
                    detail="No dominant structural gap — execution and form likely decide this fixture.",
                )
            )
        return cards[:5]

    def _embedding_gap_report(self, team_a: str, team_b: str) -> str | None:
        players_a = self.engine.team_players(team_a)
        players_b = self.engine.team_players(team_b)
        if not players_a or not players_b:
            return None
        b_top = max(players_b, key=lambda p: p.minutes)
        best_dist = min(self._cosine_distance(p.embedding, b_top.embedding) for p in players_a)
        if best_dist > COSINE_GAP_THRESHOLD:
            return (
                f"No {team_a} player within {COSINE_GAP_THRESHOLD} cosine distance of "
                f"{team_b}'s {b_top.player} ({b_top.archetype}) — structural mismatch."
            )
        return None

    def diagnose(self, team_a: str, team_b: str) -> FixtureBrief:
        fp_a = self.engine.fingerprint(team_a)
        fp_b = self.engine.fingerprint(team_b)

        clashes = self._style_clashes(fp_a, fp_b)
        structural = self._structural_gaps(fp_a, fp_b)
        exploits = self._exploit_vectors(fp_a, fp_b, clashes)
        wildcards = self._wildcard_picks(fp_a.team, fp_b.team, structural)
        adjustments = self._adjustments(fp_a.team, fp_b.team, clashes, structural, wildcards)

        embed_gap = self._embedding_gap_report(fp_a.team, fp_b.team)
        if embed_gap:
            exploits.append(embed_gap)

        top_clash = clashes[0].description if clashes else "Even stylistic profile"
        summary = (
            f"Fixture brief: {fp_a.team} vs {fp_b.team}. "
            f"Key clash: {top_clash} "
            f"Structural gaps: {len(structural)}. "
            f"Suggested adjustments: {len(adjustments)}."
        )

        return FixtureBrief(
            team_a=fp_a.team,
            team_b=fp_b.team,
            team_a_fingerprint=fp_a.to_dict(),
            team_b_fingerprint=fp_b.to_dict(),
            style_clashes=[
                {
                    "dimension": g.dimension,
                    "team_a_value": round(g.team_a_value, 3),
                    "team_b_value": round(g.team_b_value, 3),
                    "delta_pct": g.delta_pct,
                    "description": g.description,
                }
                for g in clashes
            ],
            structural_gaps=[
                {
                    "archetype": g.archetype,
                    "opponent_share": g.opponent_share,
                    "your_share": g.your_share,
                    "description": g.description,
                }
                for g in structural
            ],
            exploit_vectors=exploits,
            adjustments=[
                {
                    "category": a.category,
                    "title": a.title,
                    "detail": a.detail,
                    "suggested_players": a.suggested_players,
                }
                for a in adjustments
            ],
            wildcard_picks=[
                {
                    "player": w.player,
                    "position": w.position,
                    "archetype": w.archetype,
                    "fills_gap": w.fills_gap,
                    "similarity_to_opponent": w.similarity_to_opponent,
                }
                for w in wildcards
            ],
            summary=summary,
        )

    def tournament_outlook(self, team: str) -> dict:
        fp = self.engine.fingerprint(team)
        all_fps = self.engine.all_fingerprints()
        opponents = [t for t in all_fps if t != fp.team]

        matchups = []
        for opp in opponents:
            brief = self.diagnose(fp.team, opp)
            difficulty = len(brief.structural_gaps) + len(
                [c for c in brief.style_clashes if c["delta_pct"] > 15]
            )
            matchups.append(
                {
                    "opponent": opp,
                    "difficulty_score": difficulty,
                    "top_gap": brief.structural_gaps[0]["description"]
                    if brief.structural_gaps
                    else "No major gap",
                    "key_adjustment": brief.adjustments[0]["title"] if brief.adjustments else None,
                }
            )

        matchups.sort(key=lambda m: m["difficulty_score"], reverse=True)
        archetypes = list(fp.archetype_mix.keys())
        redundancy = [a for a in archetypes if fp.archetype_mix.get(a, 0) > 0.35]

        return {
            "team": fp.team,
            "fingerprint": fp.to_dict(),
            "hardest_matchups": matchups[:5],
            "squad_redundancy": redundancy,
            "outlook_summary": (
                f"{fp.team}'s hardest stylistic matchup among loaded squads: "
                f"{matchups[0]['opponent'] if matchups else 'N/A'}. "
                f"Redundant archetype pockets: {', '.join(redundancy) or 'none'}."
            ),
        }
