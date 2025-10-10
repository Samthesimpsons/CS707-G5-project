import shutil
import tarfile
import zipfile
from pathlib import Path


def combine_split_archives(data_dir: Path) -> Path:
    """Combine split video archives into a single tar.gz file.

    Args:
        data_dir: Directory containing the split archives.

    Returns:
        Path to the combined archive file.

    Raises:
        FileNotFoundError: If split archives are not found.
    """
    print(f"\n{'='*80}")
    print("Step: Combining Split Video Archives")
    print(f"{'='*80}\n")

    video_dir = data_dir / "tvqa-long-videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    archive_parts = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak"]
    missing_parts = []
    for part in archive_parts:
        part_path = video_dir / f"archive.tar.gz.{part}"
        if not part_path.exists():
            missing_parts.append(part)

    if missing_parts:
        raise FileNotFoundError(
            f"Missing archive parts: {', '.join(missing_parts)}\n"
            "Please re-run download.py first."
        )

    combined_archive = video_dir / "archive.tar.gz"
    with open(combined_archive, "wb") as outfile:
        for i, part in enumerate(archive_parts, 1):
            part_path = video_dir / f"archive.tar.gz.{part}"
            with open(part_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"\nCombined archive created: {combined_archive}")
    print(f"Size: {combined_archive.stat().st_size / (1024**3):.2f} GB")

    print("\nCleaning up split archive files...")
    for i, part in enumerate(archive_parts, 1):
        part_path = video_dir / f"archive.tar.gz.{part}"
        part_path.unlink()
    print("Split archive files deleted.")

    return combined_archive


def extract_video_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract Friends videos only from the combined archive.

    Args:
        archive_path: Path to the combined tar.gz archive.
        output_dir: Directory to extract videos to.

    Returns:
        Path to the extracted Friends videos directory.

    Raises:
        FileNotFoundError: If archive is not found.
    """
    print(f"\n{'='*80}")
    print("Step: Extracting Friends Videos Only")
    print(f"{'='*80}\n")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    friends_videos_dir = output_dir / "videos"

    if friends_videos_dir.exists() and list(friends_videos_dir.iterdir()):
        print(
            f"Friends videos directory already exists and is not empty: {friends_videos_dir}"
        )
        print("Skipping extraction step.")
        return friends_videos_dir

    friends_videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting Friends videos to: {friends_videos_dir}")
    print("Filtering for Friends videos only...")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()

            friends_members = [
                m
                for m in members
                if "/friends/" in m.name.lower()
                or m.name.lower().startswith("friends/")
            ]
            total_friends = len(friends_members)

            if total_friends == 0:
                print("Warning: No Friends videos found in archive!")
                return friends_videos_dir

            print(
                f"Found {total_friends} Friends video files out of {len(members)} total files"
            )
            print("Extracting Friends videos and reorganizing by season...")

            for i, member in enumerate(friends_members, 1):
                if i % 10 == 0 or i == total_friends:
                    print(f"Progress: {i}/{total_friends} files", end="\r")

                path_parts = member.name.split("/")
                season_part = None
                for part in path_parts:
                    if "season" in part.lower():
                        season_part = part
                        break

                if season_part and member.isfile():
                    season_dir = friends_videos_dir / season_part
                    season_dir.mkdir(parents=True, exist_ok=True)

                    filename = Path(member.name).name
                    dest_path = season_dir / filename

                    file_obj = tar.extractfile(member)
                    if file_obj:
                        dest_path.write_bytes(file_obj.read())
            print()

        print(f"\nFriends videos extracted to: {friends_videos_dir}")

        video_files = list(friends_videos_dir.rglob("*.mp4")) + list(
            friends_videos_dir.rglob("*.mkv")
        )
        print(f"Total Friends video files: {len(video_files)}")

        print("\nCleaning up combined archive file...")
        archive_path.unlink()
        print(f"Deleted: {archive_path}")

        video_archive_dir = output_dir / "tvqa-long-videos"
        if video_archive_dir.exists():
            print("\nCleaning up tvqa-long-videos directory...")
            shutil.rmtree(video_archive_dir)
            print(f"Deleted: {video_archive_dir}")

    except Exception as e:
        print(f"\nError extracting archive: {e}")
        raise

    return friends_videos_dir


def extract_subtitles(data_dir: Path) -> Path:
    """Extract Friends subtitle files only.

    Args:
        data_dir: Directory containing the subtitle zip file.

    Returns:
        Path to the extracted Friends subtitles directory.

    Raises:
        FileNotFoundError: If subtitle zip is not found.
    """
    print(f"\n{'='*80}")
    print("Step: Extracting Friends Subtitles Only")
    print(f"{'='*80}\n")

    subtitle_zip = data_dir / "tvqa_subtitles.zip"
    if not subtitle_zip.exists():
        raise FileNotFoundError(f"Subtitle zip not found: {subtitle_zip}")

    friends_subtitles_dir = data_dir / "tvqa_subtitles"
    temp_subtitles_dir = data_dir / "subtitles_temp"
    temp_subtitles_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(subtitle_zip, "r") as zip_ref:
            all_files = zip_ref.namelist()

            friends_files = [
                f
                for f in all_files
                if "/friends/" in f.lower() or f.lower().startswith("friends/")
            ]
            total_friends = len(friends_files)

            if total_friends == 0:
                print("Warning: No Friends subtitle files found in archive!")
                return friends_subtitles_dir

            print(
                f"Found {total_friends} Friends subtitle files out of {len(all_files)} total files"
            )
            print("Extracting Friends subtitles...")

            for i, file in enumerate(friends_files, 1):
                if i % 10 == 0 or i == total_friends:
                    print(f"Progress: {i}/{total_friends} files", end="\r")
                zip_ref.extract(file, temp_subtitles_dir)
            print()

        friends_subtitles_dir.mkdir(parents=True, exist_ok=True)

        friends_temp_path = None
        for path in temp_subtitles_dir.rglob("*"):
            if path.is_dir() and "friends" in path.name.lower():
                friends_temp_path = path
                break

        if friends_temp_path:
            for item in friends_temp_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(friends_temp_path)
                    dest_path = friends_subtitles_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)

        shutil.rmtree(temp_subtitles_dir)

        print(f"\nFriends subtitles extracted to: {friends_subtitles_dir}")
        subtitle_files = list(friends_subtitles_dir.rglob("*.srt")) + list(
            friends_subtitles_dir.rglob("*.vtt")
        )
        print(f"Total Friends subtitle files: {len(subtitle_files)}")

        print("\nCleaning up subtitle zip file...")
        subtitle_zip.unlink()
        print(f"Deleted: {subtitle_zip}")

    except Exception as e:
        print(f"\nError extracting subtitles: {e}")
        if temp_subtitles_dir.exists():
            shutil.rmtree(temp_subtitles_dir)
        raise

    return friends_subtitles_dir


def extract_data_pipeline() -> None:
    """Extract pipeline to combine archives and extract Friends content only.

    Pipeline stages:
    1. Combine split video archives into single tar.gz file
    2. Extract only Friends videos from combined archive
    3. Extract only Friends subtitles from zip file
    4. Clean up temporary files and archives
    """
    data_dir = Path("./data")

    try:
        combined_archive = combine_split_archives(data_dir)

        extract_video_archive(combined_archive, data_dir)

        extract_subtitles(data_dir)

    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
