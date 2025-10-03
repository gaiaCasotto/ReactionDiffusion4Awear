#TO RUN:
# python client_eeg.py  --host 127.0.0.1 --port 5000 --fs 256

#!/usr/bin/env python3
"""
Stream simulated EEG samples to your Flask server in real time.

Server:
  python rd_taichi_eeg_classify.py --fs 256 --buffer-s 8 --port 5000

Client (this script):
  python eeg_stream_client.py --host 127.0.0.1 --port 5000 --fs 256 --chunk 64 --duration 120 --demo-profile

POSTs JSON to: http://{host}:{port}/ingest
Body: {"samples": [float, ...]}
"""


'''
----------------- INTERPRETATION OF THE DATAFRAME COLUMNS ---------------------
['waveform_value', 'segment', 'time_UTC', 'timestamp', 'time_sample', 'focus_type',
 'TGA_ist_dB', 'GLOBAL_TABR_max_dB', 'GLOBAL_TABR_min_dB', 'TABR_avg_dB', 'TABR_ist_dB',
 'GLOBAL_FOCUS_min_dB', 'GLOBAL_TGA_min_dB', 'GLOBAL_TGA_max_dB', 'TGA_avg_dB',
 'GLOBAL_FOCUS_max_dB', 'FOCUS_ist_dB']


Raw + metadata
waveform_value → raw EEG sample amplitude (µV).
segment        → segment identifier (e.g. "seg_0", "seg_1").
time_UTC, timestamp, time_sample → different time references.
focus_type     → label placeholder ("no_label" in your data).


TABR features (Theta–Alpha Band Ratio)
TABR_ist_dB → instantaneous Theta/Alpha ratio, in dB.
TABR_avg_dB → average Theta/Alpha ratio across the segment.
GLOBAL_TABR_max_dB → maximum Theta/Alpha ratio across channels/segment.
GLOBAL_TABR_min_dB → minimum Theta/Alpha ratio across channels/segment.
👉 Used for relaxation vs cognitive load:
Higher alpha (lower TABR) = calmer.
Higher theta/alpha ratio (higher TABR) = drowsiness or stress.



TGA features (Theta–Gamma/Alpha mix, sometimes Theta–Gamma Ratio or Theta–Gamma–Alpha index)
TGA_ist_dB → instantaneous Theta vs Gamma/Alpha ratio in dB.
TGA_avg_dB → average Theta vs Gamma/Alpha ratio.
GLOBAL_TGA_max_dB → maximum ratio in the segment.
GLOBAL_TGA_min_dB → minimum ratio in the segment.
👉 Gamma power ↑ relative to theta is often associated with stress/arousal.


FOCUS features (proprietary Awear metric)
FOCUS_ist_dB → instantaneous "focus band ratio" in dB.
GLOBAL_FOCUS_max_dB, GLOBAL_FOCUS_min_dB → extrema across segment.
👉 This is likely a composite marker Awear defines for attention / mental workload (probably based on beta/gamma vs alpha/theta). 

'''

import argparse
import time
import sys
import pandas as pd
import numpy    as np
import requests
import os
import threading

from datetime import datetime, timedelta, timezone
from typing   import List, Tuple

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv         import load_dotenv
load_dotenv()

sys.path.insert(0, "./src")
print("path: ", sys.path)
#to make it find awear_neuroscience folder
from awear_neuroscience.data_extraction.firestore_loader import query_eeg_data, process_eeg_records

cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
firebase_admin.initialize_app(cred)
firestore_client = firestore.Client()
print("FIREBASE INITIALIZED")

#---------each device will be linked to an email----------
emails_str = os.getenv("EMAILS", "")
if not emails_str:
    print("No EMAILS found in environment variables. Using DOCUMENT_NAME as fallback.")
    available_emails = [os.getenv("DOCUMENT_NAME")]
else:
    available_emails = [email.strip() for email in emails_str.split(",")]
    if len(available_emails) > 1:
        print("more than your own email, smt wong")
    
print("available emails ", available_emails)
email = available_emails[0]
print("only email: ", email)

def get_past_data(delta_hours):
    now = datetime.now()
    time_ranges = [(now - timedelta(hours=delta_hours), now)]

    print(f"Querying EEG data for {email} between {time_ranges[0][0]} and {time_ranges[0][1]}")
    # ----------- Query and process records --------------
    raw_records = query_eeg_data(
        firestore_client=firestore_client,
        collection_name=os.getenv("COLLECTION_NAME"),
        document_name=email,  # Use selected email instead of env variable
        subcollection_name=os.getenv("SUB_COLLECTION_NAME"),
        time_ranges=time_ranges,
    )

    print(f"Retrieved {len(raw_records)} raw records.")


    compact_df = process_eeg_records(raw_records)
    print(f"compt Processed DataFrame shape: {compact_df.shape}")
    compact_df.head()
    print(compact_df.columns.tolist())


    long_df = process_eeg_records(raw_records, return_long=True)
    print(f"long Processed DataFrame shape: {long_df.shape}, {long_df.shape[0]//256}")
    long_df.head()
    print(long_df.columns.tolist())
    print("XXxxxxxxxxxxx\n" , long_df)

    print("compact : ", compact_df['focus_type'].unique())
    print("long : " , long_df['focus_type'].unique())

    with pd.option_context('display.max_columns', None):
        print("first line : \n",long_df.head(1))

    ''' ---------- to plot waveform segment ------------
    if not long_df.empty:
        from awear_neuroscience.utils.plot_utils import plot_eeg_waveform, plot_eeg_waveform_matplotlib
        
        visualize = input("\nVisualize waveform? (y/n): ").strip().lower()
        if visualize in ['y', 'yes']:
            # Plot one segment
            #plot_eeg_waveform(long_df, segment_id="seg_0")
            plot_eeg_waveform_matplotlib(long_df, segment_id="seg_0")
        else:
            print("Skipping visualization.")
    else:
        print("No data to visualize.")
    '''

    # Build a flat, time-ordered sample array from long_df
    ordered     = long_df.sort_values(["segment", "time_sample"], kind="mergesort")
    samples_all = ordered["waveform_value"].astype(np.float32).to_numpy()

    # drop NaNs (replace with 0.0)
    if np.isnan(samples_all).any():
        print("[WARN] NaNs in waveform_value; filling with 0.0")
        samples_all = np.nan_to_num(samples_all, nan=0.0)

    return samples_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",         type=str,   default="127.0.0.1", help="Flask server host")
    ap.add_argument("--port",         type=int,   default=5000,        help="Flask server port")
    ap.add_argument("--fs",           type=float, default=256.0,       help="Sample rate (Hz)")
    ap.add_argument("--chunk",        type=int,   default=64,          help="Samples per POST")
    ap.add_argument("--duration",     type=float, default=60.0,        help="Seconds to stream (<=0 = infinite)")
    ap.add_argument("--noise-std",    type=float, default=0.05,        help="Gaussian noise std dev")
    ap.add_argument("--lf-freq",      type=float, default=10.0,        help="Low-frequency tone (Hz)")
    ap.add_argument("--hf-freq",      type=float, default=25.0,        help="High-frequency tone (Hz)")
    ap.add_argument("--demo-profile", action="store_true",             help="Cycle CALM→MOD→HIGH→EXTREME automatically")
    ap.add_argument("--fixed-hf",     type=float, default=0.2,         help="HF weight if not using --demo-profile (0..1)")
    ap.add_argument("--timeout",      type=float, default=2.0,         help="HTTP request timeout (s)")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/ingest"
    print(f"Streaming to {url} at fs={args.fs} Hz, chunk={args.chunk} samples... (Ctrl-C to stop)")

    
    # Profile configuration
    #profile = demo_profile_segments() if args.demo_profile else [(float("inf"), float(np.clip(args.fixed_hf, 0.0, 1.0)))]
    #mode = "demo" if args.demo_profile else "fixed"

    samples_all = get_past_data(100)
    # Real-time pacing
    chunk_period = args.chunk / args.fs  # seconds per chunk
    next_send = time.perf_counter() + chunk_period
    #start_sample = 0
    t0 = time.perf_counter()
    end_time = t0 + (args.duration if args.duration > 0 else 10**12)

    total_samples = samples_all.shape[0]
    idx = 0 #current read pointer

    try:
        while True:
            now = time.perf_counter()
            if now >= end_time or idx >= total_samples:
                print("Done (duration reached).")
                break
            
            # Take the next chunk from real data
            end_idx = min(idx + args.chunk, total_samples)
            chunk = samples_all[idx:end_idx]
            idx = end_idx


            '''-----old artificial chunking code-------
            # Generate next chunk
            chunk = make_chunk(
                fs=args.fs,
                start_sample_idx=start_sample,
                n=args.chunk,
                mode=mode,
                lf_freq=args.lf_freq,
                hf_freq=args.hf_freq,
                noise_std=args.noise_std,
                t0=t0,
                profile=profile,
            )
            start_sample += args.chunk
            '''

            # Send
            try:
                r = requests.post(url, json={"samples": chunk.tolist()}, timeout=args.timeout)
                if r.status_code != 200:
                    print(f"[WARN] Bad status {r.status_code}: {r.text[:200]}...")
            except Exception as e:
                print(f"[ERROR] POST failed: {e}")

            # Pace to real time (account for time spent)
            next_send += chunk_period
            sleep_for = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We’re behind; reset schedule to avoid drift
                next_send = time.perf_counter()

    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")
    

if __name__ == "__main__":
    main()


'''
def demo_profile_segments() -> List[Tuple[float, float]]:
    """
    Returns a repeating profile of (duration_seconds, hf_weight) segments.
    hf_weight in [0..1] controls how much high-frequency power to add.
    """
    fuktuple = [
        (20.0, 0.04),
        (20.0, 0.05),  # CALM: mostly LF (alpha-ish ~10 Hz)
        (20.0, 0.25),  # MOD-STRESS
        (15.0, 0.26),
        (16.0, 0.27),
        (20.0, 0.55),  # HIGH-STRESS
        (20.0, 0.90),  # EXTREME-STRESS: mostly HF
    ]
    fuklist = list(fuktuple)
    random.shuffle(fuklist)
    fuktuple = tuple(fuklist)
    return fuktuple


def make_chunk(
    fs: float,
    start_sample_idx: int,
    n: int,
    mode: str,
    lf_freq: float,
    hf_freq: float,
    noise_std: float,
    t0: float,
    profile: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate a chunk of EEG-like samples.

    mode:
      - "demo": uses a time-based profile of HF/LF blends
      - "fixed": fixed blend using hf_weight from profile[0][1] if provided, else 0.2
    """
    # Time vector for this chunk
    idx = np.arange(start_sample_idx, start_sample_idx + n, dtype=np.float64)
    t = idx / fs

    if mode == "demo" and profile:
        # Determine which segment we are in based on wall time elapsed
        elapsed = time.perf_counter() - t0
        cycle = sum(seg[0] for seg in profile)
        in_cycle = elapsed % cycle
        acc = 0.0
        hf_w = profile[-1][1]
        for dur, w in profile:
            if in_cycle < acc + dur:
                hf_w = w
                break
            acc += dur
    else:
        hf_w = profile[0][1] if (profile and len(profile) > 0) else 0.2

    lf = np.sin(2.0 * math.pi * lf_freq * t, dtype=np.float64)  # 8–12 Hz alpha-ish
    hf = np.sin(2.0 * math.pi * hf_freq * t + 0.7, dtype=np.float64)  # 20–35 Hz beta/gamma-ish
    signal = (1.0 - hf_w) * lf + hf_w * hf

    if noise_std > 0:
        signal = signal + np.random.normal(0.0, noise_std, size=signal.shape)

    # Optional mild amplitude limiting to look realistic(ish)
    np.clip(signal, -3.0, 3.0, out=signal)
    return signal.astype(np.float32)

'''
