"""
Unit tests for detect_anteater_pacing_2.py

These tests validate the pacing detection logic implemented in:
- get_episodes_from_poses
- get_pacing_type_from_episode
- find_anteater_behaviour

We build helper mappings from letters A-F to bounding boxes inside a 5x5 grid cells:
A-> [1,1,2,2], B-> [6,1,7,2], C-> [1,6,2,7], D-> [6,6,7,7], E-> [1,11,2,12], F-> [6,11,7,12]

Timestamps are spaced 1 second starting at 2024-01-01T00:00:00Z unless specified otherwise.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from backend.detect_anteater_pacing_2 import (
    find_anteater_behaviour,
    get_episodes_from_poses,
    get_pacing_type_from_episode,
)

# Mapping from box letter to representative bbox (x1,y1,x2,y2) with centroid clearly in the target cell
LETTER_TO_BB = {
    "A": [1, 1, 2, 2],
    "B": [6, 1, 7, 2],
    "C": [1, 6, 2, 7],
    "D": [6, 6, 7, 7],
    "E": [1, 11, 2, 12],
    "F": [6, 11, 7, 12],
}

ISO_START = "2024-01-01T00:00:00Z"


def iso_t(n_seconds: int) -> str:
    """Return ISO8601 UTC timestamp offset by n_seconds from ISO_START."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=n_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_pose(letter: str, second: int, pose: str = "NON-RECUMBENT", behaviour: str = "") -> dict:
    """Construct a pose dictionary compatible with detect_anteater_pacing_2 interfaces."""
    return {
        "timestamp": iso_t(second),
        "bb": LETTER_TO_BB[letter],
        "pose": pose,
        "behaviour": behaviour,
    }


def sequence_to_poses(letters, start_second=0, pose="NON-RECUMBENT"):
    """Create a list of pose dicts, one per second with provided sequence of letters."""
    return [make_pose(letter, start_second + i, pose=pose) for i, letter in enumerate(letters)]


def test_main_sequence_has_at_least_one_pacing_start_and_maybe_continuations(capfd):
    """
    Build the long sequence provided by the task and assert that the algorithm labels
    at least one PACING_START and potentially PACING continuations based on overlapping episodes.

    Sequence: {A,B,C,D,E,F,D,B,D,F,D,B,D,F,D,B,D,F,D,A,C,A,C,E,C,A,C,E}, 1 per second.
    """
    long_letters = [
        "A", "B", "C", "D", "E", "F", "D", "B", "D", "F",
        "D", "B", "D", "F", "D", "B", "D", "F", "D", "A",
        "C", "A", "C", "E", "C", "A", "C", "E",
    ]
    poses = sequence_to_poses(long_letters, start_second=0, pose="NON-RECUMBENT")
    # Use deepcopy to ensure function does not mutate our original reference
    poses_copy = deepcopy(poses)

    # Run behaviour detection
    find_anteater_behaviour(poses_copy)

    # Capture printed logs for visibility
    out, err = capfd.readouterr()
    print(out)

    # Examine behaviours detected on the pose array
    behaviours = [p.get("behaviour", "") for p in poses_copy]
    # Assert at least one PACING_START was set somewhere
    assert "PACING_START" in behaviours, f"Expected at least one PACING_START; got behaviours: {behaviours}"
    # Optionally there may be PACING continuation labels based on overlaps and deletions of episodes
    # We don't strictly require it, but it's useful to see if present.
    # Not an assertion but we include a helpful print
    if "PACING" in behaviours:
        print("Detected PACING continuation in main sequence.")


def test_pacing_start_for_B_D_F_D_B_episode():
    """
    Deterministic micro-test for PACING_START:
    Construct the minimal sequence [B, D, F, D, B] over 4s duration.
    Since the episode first==last (B) and duration < 5s and previous_behaviour!='PACING',
    get_pacing_type_from_episode should return PACING_START and find_anteater_behaviour
    should label the first item of that episode as PACING_START.
    """
    letters = ["B", "D", "F", "D", "B"]  # 5 frames from t=0..4s
    poses = sequence_to_poses(letters, start_second=0, pose="NON-RECUMBENT")

    # Validate sliding episodes creation
    episodes = get_episodes_from_poses(poses)
    assert len(episodes) == 1, "With 5 items exactly, there should be a single 5-item episode."
    episode = episodes[0]

    # Ask the pacing type directly for previous_behaviour = ""
    pacing_type = get_pacing_type_from_episode(episode, previous_behaviour="")
    assert pacing_type == "PACING_START"

    # Now check find_anteater_behaviour marking
    poses_copy = deepcopy(poses)
    find_anteater_behaviour(poses_copy)
    # The algorithm marks the first item of the episode index 0
    assert poses_copy[0]["behaviour"] == "PACING_START"


def test_pacing_continue_on_overlapping_same_anchor():
    """
    Deterministic micro-test for PACING continuation:
    Use overlapping episodes that both start and end on the same box within <5s.
    Example pattern: [B, D, F, D, B, D, F, D, B] with 1s spacings.
    The first episode (0..4) returns to B -> PACING_START at index 0.
    After skipping of next 3 episodes, the next considered episode should again be a B-return,
    causing 'PACING' continuation set on its first item.
    """
    letters = ["B", "D", "F", "D", "B", "D", "F", "D", "B"]  # 9 frames (0..8s)
    poses = sequence_to_poses(letters, start_second=0, pose="NON-RECUMBENT")

    poses_copy = deepcopy(poses)
    find_anteater_behaviour(poses_copy)

    behaviours = [p.get("behaviour", "") for p in poses_copy]
    # Expect one PACING_START at the first item
    assert behaviours[0] == "PACING_START"
    # Because episodes after the first are pruned (skip next 3), the next processed episode
    # aligns with the second 'B' return; we expect a PACING continuation label somewhere later.
    assert "PACING" in behaviours, f"Expected at least one PACING continuation label; got {behaviours}"


def test_pacing_stopped_when_next_episode_first_not_equal_last():
    """
    Deterministic micro-test for PACING_STOPPED:
    After being in pacing, if a subsequent episode evaluated has first!=last (i.e., does not
    return to same box within 5s) and previous_behaviour=='PACING', then
    get_pacing_type_from_episode should yield 'PACING_STOPPED'. Note the find_anteater_behaviour
    does not set a behaviour label for STOPPED; we validate the function output directly.
    """
    # Construct two episodes. First returns to B, second breaks the return.
    # To build explicit episodes we need at least 9 frames. We'll handcraft:
    # Episode 1: [B, D, F, D, B] -> returns to B in 4s -> PACING_START
    # Episode 2: [B, D, F, D, A] -> first!=last (B!=A) while previously pacing -> PACING_STOPPED
    letters = ["B", "D", "F", "D", "B", "B", "D", "F", "D", "A"]  # 10 frames
    poses = sequence_to_poses(letters, start_second=0, pose="NON-RECUMBENT")
    episodes = get_episodes_from_poses(poses)
    assert len(episodes) == 6  # 10-4 = 6 episodes of length 5

    # First episode should start pacing
    ep1 = episodes[0]
    t1 = get_pacing_type_from_episode(ep1, previous_behaviour="")
    assert t1 == "PACING_START"

    # The very next episode (index 1) begins with 'D' and ends 'B' -> same; but due to pruning in the main function,
    # to test STOPPED deterministically we evaluate a later episode that ends at 'A' while previous_behaviour='PACING'
    # Choose ep index 5-1 = 4 (0-based) to include a different end:
    found_stopped = False
    for i in range(1, len(episodes)):
        t = get_pacing_type_from_episode(episodes[i], previous_behaviour="PACING")
        if t == "PACING_STOPPED":
            found_stopped = True
            break
    assert found_stopped, "Expected at least one episode to yield PACING_STOPPED when previous behaviour is PACING."


def test_episodes_beginning_with_recumbent_are_skipped(capfd):
    """
    Episodes whose first item has pose == 'RECUMBENT' are skipped by find_anteater_behaviour.
    We construct a valid pacing episode starting with RECUMBENT, and ensure no behaviour labels are set.
    """
    # Prepare a valid pacing episode [B, D, F, D, B] but mark the first item as RECUMBENT.
    letters = ["B", "D", "F", "D", "B"]
    poses = sequence_to_poses(letters, start_second=0, pose="NON-RECUMBENT")
    poses[0]["pose"] = "RECUMBENT"

    poses_copy = deepcopy(poses)
    find_anteater_behaviour(poses_copy)

    # No behaviour labels should be set because the only episode was skipped
    behaviours = [p.get("behaviour", "") for p in poses_copy]
    assert all(b == "" for b in behaviours), f"Expected no behaviours labeled; got {behaviours}"
