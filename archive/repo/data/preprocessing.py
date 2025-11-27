import json
import re
import shutil
import pysrt  # type: ignore[import-not-found]

from pathlib import Path
from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from rapidfuzz import fuzz  # type: ignore[import-not-found]
from typing import Any, Dict, List, cast


def parse_html_script_new(html_path: Path) -> Dict[str, Any]:
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text() if title_tag else ""

    episode_num = html_path.stem
    scenes = []
    current_scene = {"scene_description": "Episode dialogue", "dialogue": []}

    # Skip terms in header/footer metadata
    skip_keywords = [
        "written", "transcribed", "originally", "minor",
        "adjustment", "story", "teleplay", "directed"
    ]
    max_speaker_len = 30

    def extract_multiple_speakers(text: str) -> List[Dict[str, str]]:
        """Extract all SPEAKER: dialogue pairs from a single line."""
        results = []
        pattern = re.compile(r"([A-Za-z][A-Za-z\s.'()/-]{0,30}):")

        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            speaker = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            dialogue = text[start:end].strip()

            # Clean + validate
            if any(k in speaker.lower() for k in skip_keywords):
                continue
            if len(speaker) > max_speaker_len:
                continue
            dialogue = re.sub(r"\s+", " ", dialogue).strip()
            if dialogue:
                results.append({"speaker": speaker.upper(), "text": dialogue})
        return results

    for p in soup.find_all("p"):
        raw = p.get_text(" ", strip=True)
        if not raw:
            continue

        # Scene detection like [Scene: ...]
        if raw.startswith("[Scene:") or raw.startswith("[Opening") or raw.startswith("[Closing"):
            if current_scene["dialogue"]:
                scenes.append(current_scene)
            current_scene = {"scene_description": raw, "dialogue": []}
            continue

        # Extract all speaker/dialogue segments in this paragraph
        pairs = extract_multiple_speakers(raw)
        if pairs:
            current_scene["dialogue"].extend(pairs)

    if current_scene["dialogue"]:
        scenes.append(current_scene)

    return {"episode": episode_num, "title": title, "scenes": scenes}


def parse_html_script(html_path: Path) -> Dict[str, Any]:
    """Parse HTML script file to extract scenes and dialogue.

    Handles multiple HTML format variations found across Friends episodes:

    Format 1: Standard with tags (e.g., 0101.html)
        <p><b>Speaker:</b> dialogue text</p>
        <p><strong>Speaker</strong>: dialogue text</p>

    Format 2: All caps without tags (e.g., 0203.html)
        <p>SPEAKER: dialogue text</p>

    Format 3: Multiple dialogues per paragraph with <br> (e.g., 0204.html)
        <p>SPEAKER: dialogue<br>SPEAKER: dialogue<br>...</p>

    Format 4: Malformed HTML with dialogue outside <p> tags (e.g., 0206.html)
        - Dialogue extracted from body text when paragraphs fail

    Format 5: Speaker and dialogue on separate lines (e.g., 0302.html)
        <b><p>Speaker:</b> dialogue on next line</p>

    Format 6: Dialogue split by <br> within tags (e.g., 0915.html)
        <p><B>Speaker</B>: text<br><B>Speaker</B>: text<br>...</p>

    All speaker names are normalized to uppercase for consistency.
    Metadata lines (written by, transcribed by, etc.) are filtered out.

    Args:
        html_path: Path to HTML script file.

    Returns:
        Dictionary with episode info, scenes, and dialogue:
        {
            "episode": "0101",
            "title": "...",
            "scenes": [
                {
                    "scene_description": "...",
                    "dialogue": [
                        {"speaker": "MONICA", "text": "..."},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text() if title_tag else ""

    episode_num = html_path.stem

    scenes: List[Dict[str, Any]] = []
    current_scene: Dict[str, Any] = {
        "scene_description": "Episode dialogue",
        "dialogue": [],
    }

    paragraphs = soup.find_all("p")

    for p in paragraphs:
        if p.find("br"):
            html_str = str(p)
            br_parts = re.split(r"<br\s*/?>", html_str, flags=re.IGNORECASE)

            lines = []
            for part in br_parts:
                part_soup = BeautifulSoup(part, "html.parser")
                part_text = part_soup.get_text(strip=True)
                if part_text:
                    lines.append(part_text)
        else:
            text = p.get_text(strip=True)
            lines = [text] if text else []

        for text in lines:
            is_scene_marker = (
                text.startswith("[Scene:")
                or text.startswith("[Opening")
                or text.startswith("[Closing")
            )

            if is_scene_marker:
                if current_scene and current_scene["dialogue"]:
                    scenes.append(current_scene)

                current_scene = cast(
                    Dict[str, Any], {"scene_description": text, "dialogue": []}
                )
                continue

            if ":" not in text:
                continue

            match = re.match(r"^([A-Za-z\s.]+?):\s*(.+)$", text)
            if match:
                speaker = match.group(1).strip()
                dialogue_text = match.group(2).strip()

                speaker = re.sub(r"<[^>]+>", "", speaker)
                speaker = speaker.upper()

                skip_keywords = [
                    "written",
                    "transcribed",
                    "originally",
                    "minor",
                    "adjustment",
                    "story",
                    "teleplay",
                    "directed",
                ]
                if any(keyword in speaker.lower() for keyword in skip_keywords):
                    continue

                max_speaker_name_length = 30
                if len(speaker) > max_speaker_name_length:
                    continue

                current_scene["dialogue"].append(
                    {"speaker": speaker, "text": dialogue_text}
                )

    if current_scene and current_scene["dialogue"]:
        scenes.append(current_scene)

    total_dialogue = sum(len(scene["dialogue"]) for scene in scenes)
    if not scenes or total_dialogue == 0:
        body = soup.find("body")
        if body:
            all_text = body.get_text(separator="\n")
            lines = [line.strip() for line in all_text.split("\n") if line.strip()]

            fallback_scene: Dict[str, Any] = {
                "scene_description": "Episode dialogue",
                "dialogue": [],
            }

            i = 0
            while i < len(lines):
                text = lines[i]

                is_scene_marker = (
                    text.startswith("[Scene:")
                    or text.startswith("[Opening")
                    or text.startswith("[Closing")
                )

                if is_scene_marker:
                    if fallback_scene["dialogue"]:
                        scenes.append(fallback_scene)
                    fallback_scene = cast(
                        Dict[str, Any], {"scene_description": text, "dialogue": []}
                    )
                    i += 1
                    continue

                if text.endswith(":") and i + 1 < len(lines):
                    speaker = text[:-1].strip()

                    speaker = re.sub(r"<[^>]+>", "", speaker)
                    skip_keywords = [
                        "written",
                        "transcribed",
                        "originally",
                        "minor",
                        "adjustment",
                        "story",
                        "teleplay",
                        "directed",
                    ]

                    max_speaker_name_length = 30
                    is_valid_speaker_name = re.match(r"^[A-Za-z\s.&]+$", speaker)

                    if (
                        len(speaker) <= max_speaker_name_length
                        and not any(
                            keyword in speaker.lower() for keyword in skip_keywords
                        )
                        and is_valid_speaker_name
                    ):

                        dialogue_parts = []
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j]
                            is_next_speaker = next_line.endswith(":") and re.match(
                                r"^[A-Za-z\s.&]+:$", next_line
                            )
                            is_next_scene = (
                                next_line.startswith("[Scene:")
                                or next_line.startswith("[Opening")
                                or next_line.startswith("[Closing")
                            )

                            if is_next_speaker or is_next_scene:
                                break
                            dialogue_parts.append(next_line)
                            j += 1

                        if dialogue_parts:
                            dialogue_text = " ".join(dialogue_parts)
                            fallback_scene["dialogue"].append(
                                {"speaker": speaker.upper(), "text": dialogue_text}
                            )
                        i = j
                        continue

                if ":" in text:
                    match = re.match(r"^([A-Za-z\s.]+?):\s*(.+)$", text)
                    if match:
                        speaker = match.group(1).strip()
                        dialogue_text = match.group(2).strip()

                        speaker = re.sub(r"<[^>]+>", "", speaker)
                        speaker = speaker.upper()

                        skip_keywords = [
                            "written",
                            "transcribed",
                            "originally",
                            "minor",
                            "adjustment",
                            "story",
                            "teleplay",
                            "directed",
                        ]
                        if any(keyword in speaker.lower() for keyword in skip_keywords):
                            i += 1
                            continue

                        if len(speaker) > 30:
                            i += 1
                            continue

                        fallback_scene["dialogue"].append(
                            {"speaker": speaker, "text": dialogue_text}
                        )

                i += 1

            if fallback_scene["dialogue"]:
                scenes.append(fallback_scene)

            if scenes and total_dialogue == 0:
                scenes = [s for s in scenes if len(s["dialogue"]) > 0]

    return {"episode": episode_num, "title": title, "scenes": scenes}


def parse_srt_file(srt_path: Path) -> Dict[str, Any]:
    """Parse SRT subtitle file to extract timing and text.

    Args:
        srt_path: Path to SRT subtitle file.

    Returns:
        Dictionary with episode info and subtitles list.
    """
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    subs = None

    for encoding in encodings:
        try:
            subs = pysrt.open(str(srt_path), encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if subs is None:
        raise ValueError(
            f"Could not decode file with any of the attempted encodings: {encodings}"
        )

    parent_dir = srt_path.parent.name
    filename = srt_path.stem

    season_match = re.search(r"season[_-]?(\d+)", parent_dir, re.IGNORECASE)
    episode_match = re.search(r"episode[_-]?(\d+)", filename, re.IGNORECASE)

    season_num = season_match.group(1).zfill(2) if season_match else "00"
    episode_num = episode_match.group(1).zfill(2) if episode_match else "00"
    episode_id = f"s{season_num}e{episode_num}"

    subtitles = []
    for sub in subs:
        subtitles.append(
            {
                "index": sub.index,
                "start_time": str(sub.start),
                "end_time": str(sub.end),
                "text": sub.text.replace("\n", " "),
            }
        )

    return {"episode": episode_id, "file_name": srt_path.name, "subtitles": subtitles}


def convert_all_html_scripts(input_dir: Path, output_dir: Path) -> None:
    """Convert all HTML scripts to JSON format.

    Args:
        input_dir: Directory containing HTML script files.
        output_dir: Directory to save JSON files.
    """
    print(f"\n{'='*80}")
    print("Converting HTML Scripts to JSON (Edersoncorbari)")
    print(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.glob("*.html"))

    episode_files = [f for f in html_files if re.match(r"^\d{4}", f.stem)]

    print(f"Found {len(episode_files)} episode scripts")

    for i, html_file in enumerate(episode_files, 1):
        try:
            if i % 10 == 0 or i == len(episode_files):
                print(f"Progress: {i}/{len(episode_files)}", end="\r")

            # data = parse_html_script(html_file)
            data = parse_html_script_new(html_file)

            output_file = output_dir / f"{html_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {html_file.name}: {e}")

    print(f"\n\nHTML scripts converted: {len(episode_files)} files")
    print(f"Output directory: {output_dir}")


def convert_all_srt_files(input_dir: Path, output_dir: Path) -> None:
    """Convert all SRT files to JSON format.

    Args:
        input_dir: Directory containing SRT files (with season subdirectories).
        output_dir: Directory to save JSON files.
    """
    print(f"\n{'='*80}")
    print("Converting SRT Subtitles to JSON (TVQA-Long)")
    print(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    srt_files = sorted(input_dir.rglob("*.srt"))

    print(f"Found {len(srt_files)} SRT files")

    for i, srt_file in enumerate(srt_files, 1):
        try:
            if i % 10 == 0 or i == len(srt_files):
                print(f"Progress: {i}/{len(srt_files)}", end="\r")

            data = parse_srt_file(srt_file)

            parent_dir = srt_file.parent.name
            filename = srt_file.stem

            season_match = re.search(r"season[_-]?(\d+)", parent_dir, re.IGNORECASE)
            episode_match = re.search(r"episode[_-]?(\d+)", filename, re.IGNORECASE)

            if season_match and episode_match:
                season_num = season_match.group(1).zfill(2)
                episode_num = episode_match.group(1).zfill(2)
                output_filename = f"{season_num}{episode_num}.json"
            else:
                output_filename = f"{srt_file.stem}.json"

            output_file = output_dir / output_filename
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {srt_file.name}: {e}")

    print(f"\n\nSRT files converted: {len(srt_files)} files")
    print(f"Output directory: {output_dir}")


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching by removing punctuation and normalizing whitespace.

    Removes stage directions in parentheses, all punctuation marks,
    collapses multiple spaces into one, and converts to lowercase.
    Used to improve matching accuracy between different subtitle formats.

    Args:
        text: Raw subtitle or dialogue text.

    Returns:
        Normalized text string suitable for fuzzy matching.
    """
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def is_dialogue_entry(subtitle_text: str) -> bool:
    """Determine if a subtitle entry contains dialogue rather than metadata.

    Filters out common non-dialogue patterns found in subtitle files
    including episode titles, credits, scene descriptions, and very
    short entries that are typically scene markers or transitions.

    Args:
        subtitle_text: Text from a TVQA subtitle entry.

    Returns:
        True if entry appears to be dialogue, False for metadata/titles/credits.
    """
    non_dialogue_patterns = [
        r"^the one where",
        r"english subtitles",
        r"subtitles by",
        r"^\[.*\]$",
        r"^opening credits",
        r"^closing credits",
    ]

    text_lower = subtitle_text.lower()

    for pattern in non_dialogue_patterns:
        if re.search(pattern, text_lower):
            return False

    if len(subtitle_text.strip()) < 3:
        return False

    return True


def find_best_subtitle_match(
    normalized_dialogue: str,
    tvqa_subtitles: list[dict[str, Any]],
    used_subtitle_indices: set[int],
    last_matched_index: int = -1,
    search_window_backward: int = 20,
) -> tuple[list[int], float]:
    """Find best matching TVQA subtitle(s) for a given dialogue using fuzzy matching.

    Attempts to match dialogue text to subtitles using Levenshtein ratio similarity.
    Can match across multiple consecutive subtitles when dialogue was split.
    Tests single subtitle matches first, then tries combining 2-3 consecutive
    subtitles to handle cases where one dialogue line was split into multiple
    subtitle entries.

    Enforces temporal ordering by only searching forward from the last matched index,
    with a small backward window for flexibility in case of minor ordering differences.

    Args:
        normalized_dialogue: Preprocessed dialogue text from edersoncorbari.
        tvqa_subtitles: List of all subtitle entries from TVQA.
        used_subtitle_indices: Set of indices already matched to avoid duplicates.
        last_matched_index: Index of the last matched subtitle (for temporal ordering).
        search_window_backward: How many indices backward to allow searching from last match.

    Returns:
        Tuple containing:
            - List of matched subtitle indices (empty if no good match found).
            - Confidence score (0-100) indicating match quality.
    """
    best_indices = []
    best_confidence = 0.0

    min_search_index = max(0, last_matched_index - search_window_backward)

    dialogue_candidates = [
        (idx, subtitle)
        for idx, subtitle in enumerate(tvqa_subtitles)
        if is_dialogue_entry(subtitle["text"])
        and idx not in used_subtitle_indices
        and idx >= min_search_index
    ]

    if not dialogue_candidates:
        return [], 0.0

    for candidate_idx, candidate_subtitle in dialogue_candidates:
        normalized_subtitle = normalize_text(candidate_subtitle["text"])
        confidence = fuzz.ratio(normalized_dialogue, normalized_subtitle)

        if confidence > best_confidence:
            best_confidence = confidence
            best_indices = [candidate_idx]

    for window_size in range(2, 4):
        for i in range(len(dialogue_candidates) - window_size + 1):
            window_indices = [dialogue_candidates[i + j][0] for j in range(window_size)]

            if any(idx in used_subtitle_indices for idx in window_indices):
                continue

            if window_indices != list(
                range(window_indices[0], window_indices[0] + window_size)
            ):
                continue

            combined_normalized = " ".join(
                normalize_text(tvqa_subtitles[idx]["text"]) for idx in window_indices
            )
            confidence = fuzz.ratio(normalized_dialogue, combined_normalized)

            if confidence > best_confidence:
                best_confidence = confidence
                best_indices = window_indices

    return best_indices, best_confidence


def sync_episode_timestamps(
    edersoncorbari_path: Path, tvqa_path: Path, confidence_threshold: float = 90.0
) -> Dict[str, Any]:
    """Sync timestamps from TVQA subtitles to edersoncorbari dialogue entries.

    Matches each dialogue line from edersoncorbari to corresponding TVQA
    subtitle(s) using fuzzy text matching. Adds timing information while
    preserving the clean structure and speaker labels from edersoncorbari.

    Only includes matches with confidence above the threshold to ensure quality.
    Enforces temporal ordering to prevent timestamp jumps.

    Args:
        edersoncorbari_path: Path to edersoncorbari JSON file with dialogue.
        tvqa_path: Path to TVQA JSON file with timestamps.
        confidence_threshold: Minimum confidence score (0-100) to include match.

    Returns:
        Synced episode data with added timestamp fields.
    """
    with open(edersoncorbari_path, encoding="utf-8") as f:
        ederson_data = json.load(f)

    with open(tvqa_path, encoding="utf-8") as f:
        tvqa_data = json.load(f)

    used_tvqa_indices: set[int] = set()
    last_matched_index = -1

    synced_data = ederson_data.copy()

    for scene_idx, scene in enumerate(synced_data["scenes"]):
        for dialogue_idx, dialogue in enumerate(scene["dialogue"]):
            normalized_dialogue = normalize_text(dialogue["text"])

            matched_indices, confidence = find_best_subtitle_match(
                normalized_dialogue,
                tvqa_data["subtitles"],
                used_tvqa_indices,
                last_matched_index,
            )

            if matched_indices and confidence >= confidence_threshold:
                used_tvqa_indices.update(matched_indices)
                last_matched_index = max(matched_indices)

                start_time = tvqa_data["subtitles"][matched_indices[0]]["start_time"]
                end_time = tvqa_data["subtitles"][matched_indices[-1]]["end_time"]

                synced_data["scenes"][scene_idx]["dialogue"][dialogue_idx][
                    "start_time"
                ] = start_time
                synced_data["scenes"][scene_idx]["dialogue"][dialogue_idx][
                    "end_time"
                ] = end_time
                synced_data["scenes"][scene_idx]["dialogue"][dialogue_idx][
                    "match_confidence"
                ] = round(confidence, 2)

                matched_texts = [
                    tvqa_data["subtitles"][idx]["text"] for idx in matched_indices
                ]
                synced_data["scenes"][scene_idx]["dialogue"][dialogue_idx][
                    "matched_tvqa_text"
                ] = " ".join(matched_texts)

    return synced_data


def sync_all_episodes(
    edersoncorbari_dir: Path,
    tvqa_dir: Path,
    synced_output_dir: Path,
) -> None:
    """Synchronize timestamps for all available episodes.

    Processes all episodes in the edersoncorbari directory that have
    corresponding TVQA subtitle files. Saves synced data for each episode.
    Prints progress and generates overall statistics at completion.

    Args:
        edersoncorbari_dir: Directory with edersoncorbari JSON files.
        tvqa_dir: Directory with TVQA JSON files.
        synced_output_dir: Directory to save synced subtitle files.
    """
    print(f"\n{'='*80}")
    print("Syncing TVQA Timestamps to Edersoncorbari Dialogue")
    print(f"{'='*80}\n")

    synced_output_dir.mkdir(parents=True, exist_ok=True)

    ederson_files = sorted(edersoncorbari_dir.glob("*.json"))

    print(f"Found {len(ederson_files)} episodes to sync\n")

    episodes_processed = 0
    total_dialogues = 0
    total_matched = 0
    total_unmatched = 0

    for ederson_file in ederson_files:
        episode_id = ederson_file.stem
        tvqa_file = tvqa_dir / f"{episode_id}.json"

        if not tvqa_file.exists():
            print(f"[Warning!] Skipping {episode_id}: No matching TVQA file")
            continue

        try:
            synced_data = sync_episode_timestamps(ederson_file, tvqa_file)

            output_file = synced_output_dir / f"{episode_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(synced_data, f, indent=2, ensure_ascii=False)

            episode_dialogues = sum(
                len(scene["dialogue"]) for scene in synced_data["scenes"]
            )
            episode_matched = sum(
                1
                for scene in synced_data["scenes"]
                for dialogue in scene["dialogue"]
                if "start_time" in dialogue
            )

            episodes_processed += 1
            total_dialogues += episode_dialogues
            total_matched += episode_matched
            total_unmatched += episode_dialogues - episode_matched

        except Exception as e:
            print(f"[ERROR] {e}")

    print("OVERALL SUMMARY:")
    print(f"Episodes processed: {episodes_processed}")
    print(f"Total dialogues: {total_dialogues}")
    print(f"Total matched: {total_matched}")
    print(f"Total unmatched: {total_unmatched}")
    if total_dialogues > 0:
        match_rate = total_matched / total_dialogues * 100
        print(f"Overall match rate: {match_rate:.2f}%")


def preprocess_data_pipeline() -> None:
    """Preprocess pipeline to convert scripts and subtitles to JSON, then sync timestamps.

    Pipeline stages:
    1. Convert HTML scripts (edersoncorbari) to JSON with speaker labels
    2. Convert SRT subtitles (TVQA) to JSON with timestamps
    3. Sync TVQA timestamps to edersoncorbari dialogue using fuzzy matching

    Creates output directories:
    - data/subtitles/edersoncorbari: Parsed scripts with speakers
    - data/subtitles/tvqa: Parsed subtitles with timestamps
    - data/subtitles/synced: Combined data with timestamps and speakers
    """
    data_dir = Path("./data")

    try:
        html_input_dir = data_dir / "edersoncorbari_subtitles"
        html_output_dir = data_dir / "subtitles" / "edersoncorbari"

        if html_input_dir.exists():
            convert_all_html_scripts(html_input_dir, html_output_dir)
            shutil.rmtree(html_input_dir)
        else:
            print(f"Warning: HTML scripts directory not found: {html_input_dir}")

        srt_input_dir = data_dir / "tvqa_subtitles"
        srt_output_dir = data_dir / "subtitles" / "tvqa"

        if srt_input_dir.exists():
            convert_all_srt_files(srt_input_dir, srt_output_dir)
            shutil.rmtree(srt_input_dir)
        else:
            print(f"Warning: SRT subtitles directory not found: {srt_input_dir}")

        edersoncorbari_dir = data_dir / "subtitles" / "edersoncorbari"
        tvqa_dir = data_dir / "subtitles" / "tvqa"
        synced_dir = data_dir / "subtitles" / "synced"

        if edersoncorbari_dir.exists() and tvqa_dir.exists():
            sync_all_episodes(edersoncorbari_dir, tvqa_dir, synced_dir)
        else:
            print("Warning: Cannot sync - missing source directories")
            if not edersoncorbari_dir.exists():
                print(f"  Missing: {edersoncorbari_dir}")
            if not tvqa_dir.exists():
                print(f"  Missing: {tvqa_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
