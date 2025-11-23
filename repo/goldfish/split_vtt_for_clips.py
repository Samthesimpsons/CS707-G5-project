import os, math, webvtt
from datetime import timedelta

def _ts_to_seconds(ts: str) -> float:
    # "HH:MM:SS.mmm" -> seconds (float)
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s.replace(',', '.'))

def _sec_to_vtt_ts(sec: float) -> str:
    """Convert seconds (float) to WebVTT timestamp HH:MM:SS.mmm"""
    if sec < 0:
        sec = 0.0
    td = timedelta(seconds=sec)
    s = str(td)

    # ensure we have milliseconds
    if '.' not in s:
        s += '.000'

    whole, frac = s.split('.')
    frac = (frac + '000')[:3]  # exactly 3 digits

    # ensure hours, minutes, seconds exist (zero-padded)
    parts = whole.split(':')
    while len(parts) < 3:
        parts.insert(0, '0')
    hh, mm, ss = [int(x) for x in parts]
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{frac}"


def split_vtt_for_clips(
    full_vtt_path: str,
    out_dir: str,
    clip_duration: float,
    total_duration: float = None,
    num_clips: int = None
):
    """
    Split a full-length VTT into numbered clip VTTs: 0001.vtt, 0002.vtt, ...
    """
    out_paths = []
    os.makedirs(out_dir, exist_ok=True)
    vtt = webvtt.read(full_vtt_path)

    if total_duration is None:
        last_end = 0.0
        for c in vtt:
            last_end = max(last_end, _ts_to_seconds(c.end))
        total_duration = last_end

    if num_clips is None:
        num_clips = math.ceil(total_duration / clip_duration)

    for i in range(num_clips):
        seg_start = i * clip_duration
        seg_end   = min((i + 1) * clip_duration, total_duration)

        # filenames as 0001.vtt, 0002.vtt, ...
        out_path = os.path.join(out_dir, f"{i+1:04d}.vtt")

        lines = ["WEBVTT", ""]
        for cue in vtt:
            c_start = _ts_to_seconds(cue.start)
            c_end   = _ts_to_seconds(cue.end)
            if c_end <= seg_start or c_start >= seg_end:
                continue

            new_start = max(c_start, seg_start) - seg_start
            new_end   = min(c_end, seg_end) - seg_start
            if new_end <= new_start:
                continue

            start_ts = _sec_to_vtt_ts(new_start)
            end_ts   = _sec_to_vtt_ts(new_end)

            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(cue.text.strip())
            lines.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        out_paths.append(out_path)
        print(f"Wrote {out_path}")

    return out_paths