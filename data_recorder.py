# data_recorder.py
from __future__ import annotations
import os, io, csv, json, gzip, queue, threading, time, datetime as dt
from typing import Any, Mapping, Optional, Dict, Tuple

def _date_tag(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

class DataRecorder:
    """
    Files created in ./recordings:

    - raw_YYYY-MM-DD.ndjson.gz
      each line: {"ts": <local_unix>, "iso": "...", "packet": {...raw firestore doc...}}

    - hf_lf_YYYY-MM-DD.csv   (we'll open/prepare it but you only need it once you compute hf/lf)
      columns: ts,iso,hf,lf,ratio
    """

    def __init__(self, out_dir: str = "recordings", flush_every: int = 100):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.flush_every = flush_every

        # background thread infra
        import queue as _q
        self._q: "queue.Queue[tuple[str, dict]]" = _q.Queue(maxsize=10000)
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        # open handles
        self._hf_fp  = None       # CSV file handle
        self._hf_csv = None       # csv.writer
        self._raw_fp = None       # gzip handle
        self._cur_day = None      # str like "2025-10-30"

        self._csv_header_written = False
        self._since_flush = 0

        self._worker.start()

    # ---------- public API ----------

    def write_raw_packet(self, packet: Mapping[str, Any]):
        """
        Save 1 Firestore doc (what you called raw_data / doc.to_dict()).
        We'll timestamp it locally so we know when WE saw it.
        """
        now_ts = time.time()
        payload = {
            "ts": now_ts,
            "iso": dt.datetime.fromtimestamp(now_ts).isoformat(),
            "packet": dict(packet),  # shallow copy, must be JSON-serializable
        }
        self._q.put(("raw", payload), block=False)

    def write_hf_lf(self, ratio: float):
        """
        Optional (not yet called in your code).
        """
        now_ts = time.time()
        payload = {
            "ts": now_ts,
            "iso": dt.datetime.fromtimestamp(now_ts).isoformat(),
            "ratio": float(ratio),
        }
        self._q.put(("hf", payload), block=False)

    def close(self):
        self._stop_evt.set()
        self._worker.join(timeout=2.0)
        self._close_files()

    # ---------- internals ----------

    def _rotate_day_if_needed(self, ts: float):
        day = _date_tag(ts)
        if day == self._cur_day:
            return

        # new day → close/reopen
        self._close_files()
        self._cur_day = day

        # hf/lf CSV
        import csv
        hf_path = os.path.join(self.out_dir, f"hf_lf_{day}.csv")
        new_file = not os.path.exists(hf_path)
        self._hf_fp = open(hf_path, "a", newline="", encoding="utf-8")
        self._hf_csv = csv.writer(self._hf_fp)

        self._csv_header_written = (not new_file and os.path.getsize(hf_path) > 0)
        if not self._csv_header_written:
            self._hf_csv.writerow(["ts","iso","hf","lf","ratio"])
            self._csv_header_written = True

        # raw NDJSON.GZ
        raw_path = os.path.join(self.out_dir, f"raw_{day}.ndjson.gz")
        self._raw_fp = gzip.open(raw_path, "at", encoding="utf-8")

    def _close_files(self):
        if self._hf_fp:
            self._hf_fp.flush()
            self._hf_fp.close()
            self._hf_fp = None
            self._hf_csv = None
        if self._raw_fp:
            self._raw_fp.flush()
            self._raw_fp.close()
            self._raw_fp = None

    def _flush_maybe(self):
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            if self._hf_fp: self._hf_fp.flush()
            if self._raw_fp: self._raw_fp.flush()
            self._since_flush = 0

    def _worker_loop(self):
        """
        runs in background: consumes queue, rotates log files by day,
        writes rows/lines, flushes occasionally.
        """
        while not (self._stop_evt.is_set() and self._q.empty()):
            try:
                kind, payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            ts = float(payload.get("ts", time.time()))
            self._rotate_day_if_needed(ts)

            if kind == "raw":
                # append one JSON line (packet dump)
                self._raw_fp.write(json.dumps(payload, separators=(",", ":")) + "\n")  # type: ignore[arg-type]

            elif kind == "hf":
                # append one CSV row
                row = [payload["ts"], payload["iso"], payload["hf"], payload["lf"], payload["ratio"]]
                self._hf_csv.writerow(row)  # type: ignore[arg-type]

            self._flush_maybe()

        # final flush
        if self._hf_fp: self._hf_fp.flush()
        if self._raw_fp: self._raw_fp.flush()
