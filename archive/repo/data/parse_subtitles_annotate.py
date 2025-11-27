import json
import re

from pathlib import Path
from bs4 import BeautifulSoup  
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

            data = parse_html_script_new(html_file)

            output_file = output_dir / f"{html_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {html_file.name}: {e}")

    print(f"\n\nHTML scripts converted: {len(episode_files)} files")
    print(f"Output directory: {output_dir}")



def add_new_annotations(data: dict):
    episode = data.get("episode")
    scenes = data.get("scenes", [])
    bad_subject_strings = ["ALL", " AND ", ")", "(", "GUYS", "GIRLS", "BOTH", "ITNERCOM", " TO "]
    counter = 0
    for idx, scene in enumerate(scenes):
        ### Extract all unique subjects
        speakers_seen = []
        for turn in scene.get("dialogue", []):
            sp = turn.get("speaker")
            if isinstance(sp, str):
                sp_clean = sp.strip()
                if sp_clean and sp_clean not in speakers_seen:
                        if not any(s in sp_clean for s in bad_subject_strings):
                            speakers_seen.append(sp_clean)
        
        ### Extract the location from scene description
        scene_description = scene.get("scene_description")
        print(scene_description)
        if not isinstance(scene_description, str):
            return ""
        else:
            m = re.search(r"\[Scene\s*[:\-]?\s*([^\],]+)", scene_description, flags=re.I)
            try:
                location = m.group(1).strip()
            except AttributeError:
                location = ""
        if scene.get("clip_start", "") != "":
            scene["context_label"] = f"{episode}_scene_{str(counter).zfill(3)}_{location}"
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


if __name__ == "__main__":
    html_input_dir = Path("./edersoncorbari_subtitles_html")
    html_output_dir = Path("./edersoncorbari_new_annotated")

    if html_input_dir.exists():
        convert_all_html_scripts(html_input_dir, html_output_dir)

    subs_dir = Path("./edersoncorbari_new_annotated/season_01")
    output_dir = Path("./annotated_tuples/season_01")

    json_subs = []
    for file in sorted(subs_dir.rglob("*.json")):
        if 'bad' not in str(file):
            with open(file) as f:
                subs = json.load(f)
                json_subs.append(subs)
                f.close()                     

    output_dir.mkdir(exist_ok=True)
    for subs_data in json_subs:
        annotated_subs_data, episode = add_new_annotations(subs_data)
        save_dir = output_dir / f"{episode}.json"
        with open(save_dir, "w") as f:
            json.dump(annotated_subs_data, f, indent=4)