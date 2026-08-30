from preflight.report import verdict_sentence


def test_sentences():
    assert "nominal" in verdict_sentence("left_knee_joint", "healthy", 0.0).lower()
    s = verdict_sentence("waist_yaw_joint", "obstruction", 0.8)
    assert "waist" in s and "obstruct" in s.lower() and "severe" in s.lower()
