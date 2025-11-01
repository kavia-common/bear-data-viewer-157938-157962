from datetime import datetime
from typing import Literal

BEHAVIOUR = Literal["PACING_START", "PACING", "PACING_STOPPED"]

BOXES = {
    "BOX_A": {0, 0, 5, 0},
    "BOX_B": {5, 0, 10, 0},
    "BOX_C": {0, 5, 5, 5},
    "BOX_D": {5, 5, 10, 5},
    "BOX_E": {0, 10, 5, 10},
    "BOX_F": {5, 10, 10, 10},
}

# Test data for anteater pacing detection, such that the centre point
# of the bounding boxes moves from BOX_B to BOX_D to BOX_F and back to BOX_B,
# indicating pacing behaviour.
ANTEATER_POSE_TEST_DATA = [
    {
        "timestamp": "2024-01-01T00:00:00Z",
        "bb": [5, 0, 6, 1],  # centroid 5.5,0.5 -> BOX_B
        "pose": "RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:01Z",
        "bb": [7, 2, 9, 4],  # centroid 8,3 -> BOX_B
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:02Z",
        "bb": [6, 6, 8, 8],  # centroid 7,7 -> BOX_D
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:03Z",
        "bb": [6, 11, 8, 13],  # centroid 7,12 -> BOX_F
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:04Z",
        "bb": [8, 5, 10, 7],  # centroid 9,6 -> BOX_D
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:05Z",
        "bb": [5, 3, 7, 5],  # centroid 6,4 -> BOX_B
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    },
    {
        "timestamp": "2024-01-01T00:00:06Z",
        "bb": [1, 1, 3, 3],  # centroid 2,2 -> BOX_A
        "pose": "NON-RECUMBENT",
        "behaviour": ""
    }
]

def get_episodes_from_poses(poses):
    """
    Return a list of 3-item episodes (sliding windows) from the input poses.
    Each episode is a list: [poses[i], poses[i+1], poses[i+2]].
    """
    poses = list(poses)
    if len(poses) < 5:
        return []
    return [[poses[i], poses[i + 1], poses[i + 2], poses[i + 3], poses[i + 4]] for i in range(len(poses) - 4)]


def get_pacing_type_from_episode(episode, previous_behaviour):
    """
    Determine pacing type for a 4-item episode.
    - Compute centroids of bboxes and map them to BOX_A..BOX_F (5x5 grid).
    - Print the sequence of boxes.
    - Return one of: "PACING_START", "PACING", "PACING_STOPPED", or None.
    """
    def box_for_point(cx, cy):
        # Define top-left corner for each 5x5 box
        box_tl = {
            "BOX_A": (0, 0),
            "BOX_B": (5, 0),
            "BOX_C": (0, 5),
            "BOX_D": (5, 5),
            "BOX_E": (0, 10),
            "BOX_F": (5, 10),
        }
        for name, (x0, y0) in box_tl.items():
            if x0 <= cx < x0 + 5 and y0 <= cy < y0 + 5:
                return name
        return None

    boxes = []
    for item in episode:
        bb = item.get("bb", [])
        if not bb or len(bb) < 4:
            boxes.append(None)
            continue
        x1, y1, x2, y2 = bb
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        boxes.append(box_for_point(cx, cy))

    print("\n Boxes in episode:", boxes)

    if not boxes or boxes[0] is None or boxes[-1] is None:
        return None

    first, last = boxes[0], boxes[-1]
    start_ts = episode[0].get("timestamp")
    end_ts = episode[-1].get("timestamp")
    if not start_ts or not end_ts:
        return None

    try:
        start_dt = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%SZ")
        end_dt = datetime.strptime(end_ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        try:
            start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        except Exception:
            return None

    duration = (end_dt - start_dt).total_seconds()

    # Require duration < 5 seconds for PACING_START and PACING
    if first == last:
        if duration < 5:
            if previous_behaviour == "PACING":
                return "PACING"
            else:
                return "PACING_START"
        return None
    else:
        if previous_behaviour == "PACING":
            return "PACING_STOPPED"
        return None
    

def find_anteater_behaviour(poses):
    episodes = get_episodes_from_poses(poses)
    previous_behaviour = ""
    for idx, episode in enumerate(episodes):
        print("\n episode",episode)
        if episode[0]["pose"] == "RECUMBENT":
            print("Skip episode with RECUMBENT pose")
            continue
        else:
            print("Processing episode for PACING detection:", episode)
            pacing_type = get_pacing_type_from_episode(episode, previous_behaviour)
            if pacing_type == "PACING_START":
                previous_behaviour = "PACING"
                poses[idx]["behaviour"] = "PACING_START"
                print("Detected pacing behaviour:", pacing_type)
                # Skip next 3 episodes by removing them from the episodes list
                del episodes[idx + 1: idx + 4]
                print("Skipping next 3 episodes.")
            elif pacing_type == "PACING":
                previous_behaviour = "PACING"
                poses[idx]["behaviour"] = "PACING"
                del episodes[idx + 1: idx + 4]
                print("Detected pacing behaviour:", pacing_type)
            elif pacing_type == "PACING_STOPPED":
                previous_behaviour = ""
                print ("Pacing behaviour stopped.")
            else:
                print ("No pacing behaviour detected.")


if __name__ == "__main__":
    # Simple test to print the test data
    for entry in ANTEATER_POSE_TEST_DATA:
        print(entry)
    find_anteater_behaviour(ANTEATER_POSE_TEST_DATA)
