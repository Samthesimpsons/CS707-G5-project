import json
import re
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup


class Scene(TypedDict, total=False):
    """Dialogue scene with timing, location, and subject metadata.

    Attributes:
        scene_description: Raw scene heading text from the script.
        dialogue: List of speaker/text pairs extracted from the scene.
        clip_start: Manual clip start time (string or frame index).
        clip_end: Manual clip end time (string or frame index).
        srt_start: Manual subtitle start time (string or frame index).
        srt_end: Manual subtitle end time (string or frame index).
        context_label: Derived identifier for the scene.
        location: Parsed location from the scene heading.
        subjects: Unique speakers present in the scene.
        event_description: Narrative description of the scene.
    """

    scene_description: str
    dialogue: list[dict[str, str]]
    clip_start: str | int
    clip_end: str | int
    srt_start: int | str
    srt_end: int | str
    context_label: str
    location: str
    subjects: list[str]
    event_description: str


class EpisodeData(TypedDict):
    """Structured metadata for a Friends episode.

    Attributes:
        episode: Episode number or code (e.g., 0101).
        title: Episode title parsed from the script HTML.
        scenes: Ordered list of scenes with dialogue and annotations.
    """

    episode: str
    title: str
    scenes: list[Scene]


def parse_html_to_json(html_path: Path) -> EpisodeData:
    """Parse a single HTML Friends script into structured episode data.

    Args:
        html_path: Path to the HTML file containing the script.

    Returns:
        Episode metadata and scene/dialogue structure parsed from the HTML.

    Raises:
        FileNotFoundError: If the HTML file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text() if title_tag else ""

    episode_num = html_path.stem
    scenes: list[Scene] = []
    current_scene: Scene = {"scene_description": "Episode dialogue", "dialogue": []}

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
    max_speaker_len = 30

    def __extract_multiple_speakers(text: str) -> list[dict[str, str]]:
        """Extract all SPEAKER: dialogue pairs from a single line.

        Args:
            text: Raw dialogue line potentially containing multiple speakers.

        Returns:
            List of normalized speaker/text mappings found in the line.
        """
        results = []
        pattern = re.compile(r"([A-Za-z][A-Za-z\s.'()/-]{0,30}):")

        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            speaker = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            dialogue = text[start:end].strip()

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

        if (
            raw.startswith("[Scene:")
            or raw.startswith("[Opening")
            or raw.startswith("[Closing")
        ):
            if current_scene["dialogue"]:
                scenes.append(current_scene)
            current_scene = {"scene_description": raw, "dialogue": []}
            continue

        pairs = __extract_multiple_speakers(raw)
        if pairs:
            dialogue_list: list[dict[str, str]] = current_scene["dialogue"]
            dialogue_list.extend(pairs)

    if current_scene["dialogue"]:
        scenes.append(current_scene)

    return {"episode": episode_num, "title": title, "scenes": scenes}


def convert_all_html_scripts(
    input_dir: Path, output_dir: Path, overwrite: bool = False
) -> None:
    """Convert all HTML scripts to JSON format.

    Args:
        input_dir: Directory containing HTML script files.
        output_dir: Directory to save JSON files.
        overwrite: Recreate JSON even if it already exists (overwrites any manual edits).
    """
    print("Converting HTML Scripts to JSON (Edersoncorbari)")

    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.glob("*.html"))

    episode_files = [f for f in html_files if re.match(r"^\d{4}", f.stem)]

    print(f"Found {len(episode_files)} episode scripts")

    for i, html_file in enumerate(episode_files, 1):
        try:
            if i % 10 == 0 or i == len(episode_files):
                print(f"Progress: {i}/{len(episode_files)}", end="\r")

            data = parse_html_to_json(html_file)

            output_file = output_dir / f"{html_file.stem}.json"
            if output_file.exists() and not overwrite:
                continue
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {html_file.name}: {e}")

    print(f"\n\nHTML scripts converted: {len(episode_files)} files")
    print(f"Output directory: {output_dir}")


def add_new_annotations(data: EpisodeData):
    """Enrich dialogue scenes with labels, subjects, and normalized timing fields.

    Expects clip_start/clip_end and srt_start/srt_end to be filled in manually so
    context_label values can be derived deterministically.

    Args:
        data: Episode structure containing scenes and dialogue to annotate.

    Returns:
        Tuple of the annotated episode data and the episode id string.
    """
    episode = data.get("episode")
    scenes = data.get("scenes", [])
    bad_subject_strings = [
        "ALL",
        " AND ",
        ")",
        "(",
        "GUYS",
        "GIRLS",
        "BOTH",
        "ITNERCOM",
        " TO ",
    ]
    counter = 0
    for idx, scene in enumerate(scenes):
        speakers_seen = []
        for turn in scene.get("dialogue", []):
            sp = turn.get("speaker")
            if isinstance(sp, str):
                sp_clean = sp.strip()
                if sp_clean and sp_clean not in speakers_seen:
                    if not any(s in sp_clean for s in bad_subject_strings):
                        speakers_seen.append(sp_clean)

        scene_description = scene.get("scene_description")
        if not isinstance(scene_description, str):
            return ""
        m = re.search(r"\[Scene\s*[:\-]?\s*([^\],]+)", scene_description, flags=re.I)
        location = m.group(1).strip() if m else ""
        if scene.get("clip_start", "") != "":
            scene["context_label"] = (
                f"{episode}_scene_{str(counter).zfill(3)}_{location}"
            )
            counter += 1
        else:
            scene["context_label"] = ""
        scene["location"] = location
        scene["subjects"] = list(set(speakers_seen))
        scene["event_description"] = scene_description
        if scene.get("clip_start", "") == "":
            scene["clip_start"] = ""
        if scene.get("clip_end", "") == "":
            scene["clip_end"] = ""
        if scene.get("srt_start", 0) == 0:
            scene["srt_start"] = ""
        if scene.get("srt_end", 0) == 0:
            scene["srt_end"] = ""
    return data, episode


def _episode_has_clip_timings(episode: EpisodeData) -> bool:
    """Check whether an episode JSON already includes manual clip timing fields.

    Args:
        episode: Episode data to inspect.

    Returns:
        True if any scene contains clip or subtitle timing fields; otherwise False.
    """
    for scene in episode.get("scenes", []):
        if any(
            scene.get(field) not in ("", 0, None)
            for field in ("clip_start", "clip_end", "srt_start", "srt_end")
        ):
            return True
    return False


def preprocess_pipeline(
    require_clip_timings: bool = True, overwrite_html_json: bool = False
) -> None:
    """Convert scripts to JSON, require manual timings, then enrich annotations.

    Args:
        require_clip_timings: If True, warn when timing fields are missing in JSON inputs.
        overwrite_html_json: If True, regenerate JSON files even when they already exist.
    """
    data_root = Path("./data")
    html_input_dir = data_root / "edersoncorbari_subtitles"
    json_output_dir = data_root / "edersoncorbari_subtitles_json"
    annotated_output_dir = data_root / "annotated_tuples"

    convert_all_html_scripts(
        html_input_dir, json_output_dir, overwrite=overwrite_html_json
    )

    json_files = sorted(json_output_dir.glob("*.json"))
    if not json_files:
        print(
            f"No JSON scripts found in {json_output_dir}. "
            "Run Stage 1 (download) first to fetch HTML scripts."
        )
        return

    json_subs = []
    missing_timings = []
    for file in json_files:
        with open(file, encoding="utf-8") as f:
            subs = json.load(f)
        json_subs.append(subs)
        if require_clip_timings and not _episode_has_clip_timings(subs):
            missing_timings.append(file.name)

    if missing_timings:
        print(
            f"Clip timings are missing in {len(missing_timings)} files. Add clip_start/clip_end and srt_start/srt_end "
            f"in the JSON files under {json_output_dir} using the downloaded videos."
        )

    annotated_output_dir.mkdir(parents=True, exist_ok=True)
    for subs_data in json_subs:
        annotated_subs_data, episode = add_new_annotations(subs_data)
        save_dir = annotated_output_dir / f"{episode}.json"
        with open(save_dir, "w", encoding="utf-8") as f:
            json.dump(annotated_subs_data, f, indent=4, ensure_ascii=False)

    print(f"Annotated tuples saved to: {annotated_output_dir}")
