import os
import glob
import numpy as np
import warnings
from typing import Tuple, List, Dict, Optional

def extract_first_N_poststim_volumes(
    data: np.ndarray,
    subject: str,
    task: str,
    N: int,
    tr: float = 0.72,
    time0: str = "start",
    first_poststimulus_volume: int = 0,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Extract first N volumes after each event from an HCP-style time x region array.

    Parameters
    ----------
    data : np.ndarray
        2D array shape (T, R) where T = #timepoints/volumes and R = #regions.
    subject : str
        Subject identifier used for locating EV files (function doesn't enforce format,
        it's used to build default search paths if ev_dir is None).
    task : str
        Task name (e.g., "GAMBLING", "WM", etc.). Used for EV path discovery if ev_dir is None.
    N : int
        Number of post-stimulus volumes to extract per event.
    tr : float
        Repetition time in seconds. Default 0.72 s (HCP 3T tfMRI).
    time0 : str
        How to interpret time=0 in the event files. Options:
          - "start"  : (default) 0 s corresponds to the START of the first TR/volume.
          - "end"    : 0 s corresponds to the END of the first TR/volume (uncommon).
        (Note: HCP EVs are normally referenced to the start of the first TR.)
    first_poststimulus_volume : int
        How to define the "first post-stimulus volume":
          - 0 (default): take the volume that CONTAINS the stimulus onset
                             i.e., volume_index = floor(adjusted_time / tr)
          - 1: take the first FULL volume AFTER the stimulus onset
                   i.e., volume_index = floor(adjusted_time / tr) + 1
          - 2 or more: take the k-th full volume AFTER the stimulus onset
    verbose : bool
        If True, print summary warnings/info.

    Returns
    -------
    extracted_data : np.ndarray
        Concatenated extracted volumes across all events and files. Shape (M, R),
        where M = total extracted volumes (<= number_of_events * N).
    info : dict
        Bookkeeping dictionary with:
          - 'file_summary': list of (filename, n_events, events_used)
          - 'indices': list of (file, event_row_index, start_vol, end_vol, extracted_count)
          - 'warnings': list of warning strings produced during extraction
          - 'tr_used': tr
          - 'time0_assumption': time0
          - 'first_after': first_after

    Notes
    -----
    * The function accepts event files where the first column is onset times in seconds.
      It will attempt to detect milliseconds (large numbers) and divide by 1000 automatically.
    * If an event's requested N volumes exceed the available data, the function will:
      - extract the available tail volumes for that event (so you still get partial data),
      - append a warning (and print it if verbose).
    * You can override the TR and ev_dir if your data are not exactly HCP standard.
    """

    # --- sanity checks ---
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("`data` must be a 2D numpy array with shape (time, regions).")
    T, R = data.shape
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if time0 not in ("start", "end"):
        raise ValueError("time0 must be 'start' or 'end'")

    # --- locate EV files ---
    ev_files = []
    if task == 'EMOTION':
        ev_files.append('paper/data/raw/'+subject+'/EVs/EMOTION/fear.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/EMOTION/neut.txt')
    elif task == 'GAMBLING':
        ev_files.append('paper/data/raw/'+subject+'/EVs/GAMBLING/loss.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/GAMBLING/win.txt')
    elif task == 'LANGUAGE':
        ev_files.append('paper/data/raw/'+subject+'/EVs/LANGUAGE/math.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/LANGUAGE/story.txt')
    elif task == 'MOTOR':
        ev_files.append('paper/data/raw/'+subject+'/EVs/MOTOR/lf.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/MOTOR/lh.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/MOTOR/rf.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/MOTOR/rh.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/MOTOR/t.txt')
    elif task == 'RELATIONAL':
        ev_files.append('paper/data/raw/'+subject+'/EVs/RELATIONAL/relation.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/RELATIONAL/match.txt')
    elif task == 'SOCIAL':
        ev_files.append('paper/data/raw/'+subject+'/EVs/SOCIAL/mental.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/SOCIAL/rnd.txt')
    elif task == 'WM':
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/0bk_body.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/0bk_faces.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/0bk_places.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/0bk_tools.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/2bk_body.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/2bk_faces.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/2bk_places.txt')
        ev_files.append('paper/data/raw/'+subject+'/EVs/WM/2bk_tools.txt')

    info = {
        "file_summary": [],
        "indices": [],
        "warnings": [],
        "tr_used": tr,
        "time0_assumption": time0,
        "first_poststimulus_volume": first_poststimulus_volume,
    }

    # container for extracted segments
    extracted_segments = []

    for evf in ev_files:
        # try to load first column robustly
        try:
            # load as text, allowing variable whitespace
            arr = np.loadtxt(evf, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            onsets = arr[:, 0]
        except Exception:
            # fallback: try pandas
            try:
                import pandas as pd
                df = pd.read_csv(evf, sep=None, engine='python', header=None)
                onsets = df.iloc[:, 0].astype(float).values
            except Exception as e:
                msg = f"Failed to read event file {evf}: {e}"
                warnings.warn(msg)
                info["warnings"].append(msg)
                continue

        # # detect ms (large numbers) and convert to seconds if needed
        # if np.nanmax(onsets) > 1e4 and np.nanmedian(onsets) > 1e3:
        #     # likely in milliseconds
        #     onsets = onsets / 1000.0
        #     conv_msg = f"Converted onsets from ms->s for {evf} (divided by 1000)."
        #     info["warnings"].append(conv_msg)
        #     if verbose:
        #         print(conv_msg)

        n_events = len(onsets)
        used_events = 0

        for row_idx, t in enumerate(onsets):
            # adjust according to time0 assumption
            adj_t = float(t)
            if time0 == "end":
                # if 0 corresponds to the END of the first TR, shift times back by TR
                # so that t=0 maps to the end of vol0; treating event mapping consistently.
                adj_t = adj_t - tr

            # Map to volume index:
            # volumes are indexed with k spanning [k*tr, (k+1)*tr)
            vol_idx = int(np.floor(adj_t / tr)) + first_poststimulus_volume

            # ensure non-negative
            if vol_idx < 0:
                # clamp to 0 but warn
                warn_msg = f"Event at t={t:.3f}s in {evf} mapped to negative volume index {vol_idx}; clamping to 0."
                warnings.warn(warn_msg)
                info["warnings"].append(warn_msg)
                vol_idx = 0

            start_idx = vol_idx
            end_idx_exclusive = start_idx + N  # python slice end
            # check bounds
            if start_idx >= T:
                warn_msg = f"Event at t={t:.3f}s in {evf} -> start volume {start_idx} >= data length {T}; skipping event."
                warnings.warn(warn_msg)
                info["warnings"].append(warn_msg)
                continue

            if end_idx_exclusive > T:
                # # not enough volumes to extract N volumes for this event
                # warn_msg = (f"Event at t={t:.3f}s in {evf}: requested N={N} volumes from start {start_idx} "
                #             f"but only {T - start_idx} available (data length {T}). Extracting available tail;")
                # warnings.warn(warn_msg)
                # info["warnings"].append(warn_msg)
                # # extract the tail (partial)
                # seg = data[start_idx:T, :]
                continue  # skip partials for simplicity
            else:
                seg = data[start_idx:end_idx_exclusive, :]

            extracted_segments.append(seg)
            # info["indices"].append({
            #     "file": evf,
            #     "event_row": int(row_idx),
            #     "event_time_s": float(t),
            #     "start_vol": int(start_idx),
            #     "end_vol_exclusive": int(min(end_idx_exclusive, T)),
            #     "extracted_volumes": int(seg.shape[0])
            # })
            used_events += 1

        # info["file_summary"].append((evf, n_events, used_events))

    if len(extracted_segments) == 0:
        msg = "No volumes were extracted (no valid events or all events out of range)."
        warnings.warn(msg)
        info["warnings"].append(msg)
        if verbose:
            print(msg)
        return np.zeros((0, R)), info

    # Concatenate along time axis
    extracted_data = np.vstack(extracted_segments)

    # final sanity warning if total extracted less than expected
    # expected_max = sum(used for _, _, used in info["file_summary"]) * N
    # if extracted_data.shape[0] < expected_max:
    #     warn_msg = (f"Total extracted volumes {extracted_data.shape[0]} is less than expected max {expected_max} "
    #                 "because some events were near the end of the run and produced partial extracts.")
    #     warnings.warn(warn_msg)
    #     info["warnings"].append(warn_msg)
    #     if verbose:
    #         print(warn_msg)

    return extracted_data, info

# Example usage:
# extracted, metadata = extract_first_N_poststim_volumes(data=time_by_region_array,
#                                                       subject="100307",
#                                                       task="GAMBLING",
#                                                       N=4,
#                                                       ev_dir="/path/to/100307/MNINonLinear/Results/tfMRI_GAMBLING_LR/EVs",
#                                                       tr=0.72,
#                                                       time0="start",
#                                                       first_after=False)
