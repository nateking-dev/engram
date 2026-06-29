import numpy as np

from engram.activation import Candidate, activation, base_level, heuristic_importance


def test_base_level_decay_zero_is_flat():
    # d=0 removes decay -> every age scores 0 (the -decay ablation).
    assert base_level(5, 100, 0.0) == 0.0
    assert base_level(99, 100, 0.0) == 0.0


def test_base_level_older_is_lower():
    # With decay, an older chunk (smaller index, larger age) is less active.
    recent = base_level(90, 100, 0.5)
    old = base_level(10, 100, 0.5)
    assert recent > old


def test_base_level_stronger_decay_punishes_age_more():
    gentle = base_level(10, 100, 0.25)
    harsh = base_level(10, 100, 1.0)
    assert harsh < gentle  # harsher decay -> lower activation for the same old chunk


def test_heuristic_importance_bounded_and_signal():
    plain = heuristic_importance("the weather is nice today")
    salient = heuristic_importance("Remember: deadline is Monday, allergic to penicillin")
    assert 0.0 <= plain <= 1.0
    assert 0.0 <= salient <= 1.0
    assert salient > plain


def test_activation_combines_terms():
    q = np.array([1.0, 0.0], dtype=np.float32)
    near = Candidate("a", 50, 0.0, np.array([1.0, 0.0], dtype=np.float32))
    far = Candidate("b", 50, 0.0, np.array([0.0, 1.0], dtype=np.float32))
    scores = activation([near, far], 100, q, decay=0.0, beta=0.0, gamma=1.0)
    assert scores[0] > scores[1]  # spreading favors the semantically near chunk


def test_importance_term_can_flip_ranking():
    q = np.array([1.0, 0.0], dtype=np.float32)
    # 'b' is semantically far but flagged important; with high beta it should win.
    near = Candidate("a", 50, 0.0, np.array([1.0, 0.0], dtype=np.float32))
    far_imp = Candidate("b", 50, 1.0, np.array([0.0, 1.0], dtype=np.float32))
    no_imp = activation([near, far_imp], 100, q, decay=0.0, beta=0.0, gamma=1.0)
    with_imp = activation([near, far_imp], 100, q, decay=0.0, beta=5.0, gamma=1.0)
    assert no_imp[0] > no_imp[1]
    assert with_imp[1] > with_imp[0]
