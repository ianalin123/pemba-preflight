"""Verdicts → operator sentences + machine JSON. The demo speaks these."""
SEVERITY = [(0.25, "mild"), (0.6, "moderate"), (1.01, "severe")]
TEMPLATES = {
    "healthy": "{joint}: nominal.",
    "friction": "{joint}: {sev} friction increase — check for grit or icing.",
    "stiffness": "{joint}: {sev} stiffness — lubricant may be cold; warm up before walking.",
    "obstruction": "{joint}: motion obstructed ({sev}) — check for transport lock or snag.",
    "derate": "{joint}: {sev} torque deficit — motor derating, check temperature and battery.",
    "unknown": "{joint}: {sev} anomaly detected — cause unclassified, inspect before walking.",
}


def _sev_word(s: float) -> str:
    return next(w for lim, w in SEVERITY if s < lim)


def verdict_sentence(joint: str, cls: str, severity: float) -> str:
    pretty = joint.replace("_joint", "").replace("_", " ")
    return TEMPLATES[cls].format(joint=pretty, sev=_sev_word(severity))


def go_no_go(verdicts: list[tuple[str, str, float]]) -> str:
    bad = [v for v in verdicts if v[1] != "healthy" and v[2] > 0.5]
    if not bad:
        return "Pre-flight PASS. Clear to walk."
    return "Pre-flight FAIL. " + " ".join(verdict_sentence(*v) for v in bad)
