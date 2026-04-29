import numpy as np
import pandas as pd
from pathlib import Path

# ===== CONFIG =====
SUBJECT = "100206"
BASE_DIR = Path("paper/data/raw")  # change this
TR = 0.72

# Number of volumes per scan (HCP standard)
# adjust if needed
SCAN_VOLUMES = {
    "EMOTION": 176,
    "GAMBLING": 253,
    "LANGUAGE": 316,
    "MOTOR": 284,
    "RELATIONAL": 232,
    "SOCIAL": 274,
    "WM": 405,
}
# BLOCK FILE DEFINITIONS
BLOCK_FILES = {
    "EMOTION": ["fear.txt", "neut.txt"],
    "GAMBLING": ["win.txt", "loss.txt"],
    "LANGUAGE": ["story.txt", "math.txt"],
    "MOTOR": ["lf.txt", "rf.txt", "lh.txt", "rh.txt", "t.txt", "cue.txt"],
    "RELATIONAL": ["relation.txt", "match.txt"],
    "SOCIAL": ["mental.txt", "rnd.txt"],
    "WM": [
        "0bk_body.txt", "0bk_faces.txt", "0bk_places.txt", "0bk_tools.txt",
        "2bk_body.txt", "2bk_faces.txt", "2bk_places.txt", "2bk_tools.txt",
    ],
}


# ===== LOAD FUNCTION =====
def load_ev(path):
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["onset", "duration", "amplitude"]
    )


# ===== CORE FUNCTION =====
def build_task_vector(scan_dir, scan_name):
    n_vols = SCAN_VOLUMES[scan_name]
    task_vec = np.zeros(n_vols, dtype=int)

    for fname in BLOCK_FILES[scan_name]:
        fpath = scan_dir / fname
        if not fpath.exists():
            continue

        df = load_ev(fpath)

        for _, row in df.iterrows():
            onset = row["onset"]
            duration = row["duration"]

            start_vol = int(np.floor(onset / TR)) + 1
            end_vol = int(np.floor((onset + duration) / TR))

            start_vol = max(0, start_vol)
            end_vol = min(n_vols - 1, end_vol)

            if end_vol >= start_vol:
                task_vec[start_vol:end_vol + 1] = 1

    return task_vec


# ===== MAIN =====
def main():
    subj_dir = BASE_DIR / SUBJECT / "EVs"

    for scan_name in BLOCK_FILES.keys():
        scan_dir = subj_dir / scan_name

        if not scan_dir.exists():
            continue

        print(f"Processing {scan_name}")

        vec = build_task_vector(scan_dir, scan_name)

        out_file = f"paper/data/task_indices/{SUBJECT}_{scan_name}_task_vector.txt"
        np.savetxt(out_file, vec, fmt="%d")

        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()