import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)

from qwen_vl_utils import process_vision_info


CONFIG: dict[str, dict[str, object]] = {
    "qwen-vl-2": {
        "fps": 1.0,
        "num_frames": "",
        "max_frames": "",
        "max_channels": 64,
        "min_channels": 64,
        "width": 64,
        "height": 64,
    },
    "qwen-vl-25": {
        "fps": 1.0,
        "num_frames": "",
        "max_frames": "",
        "max_channels": "",
        "min_channels": "",
        "width": "",
        "height": "",
    },
    "video-llava": {
        "fps": 1.0,
        "num_frames": 8.0,
        "max_frames": "",
        "max_channels": "",
        "min_channels": "",
        "width": "",
        "height": "",
    },
    "video-llama-3": {
        "fps": 1.0,
        "num_frames": "",
        "max_frames": 128,
        "channels": "",
        "width": "",
        "height": "",
        "target_short_side": 64,
    },
    "intern-vl-3": {
        "fps": 1.0,
        "num_segments": 32,
    },
}

DEFAULT_MODEL_PATHS: dict[str, str] = {
    "qwen2.5-vl-7b-instruct": "./models/qwen2.5-vl-7b-instruct",
    "video-llava-7b": "./models/video-llava-7b",
    "videollama3-7b": "./models/videollama3-7b",
    "internvl3-8b": "./models/internvl3-8b",
}


class QwenVL:
    def __init__(self, model_path: str):
        """Initialize Qwen-VL model, processor, and runtime settings.

        Args:
            model_path: Filesystem path or Hugging Face id for the model weights.
        """
        self.model_path = model_path

        config = self.load_config()
        cfg = config.get("qwen-vl-25", {})

        self.fps = float(cfg.get("fps", 1.0))
        self.max_pixels = 8 * 28 * 28
        self.min_pixels = 8 * 28 * 28

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()

    def load_config(self):
        """Return static configuration for supported models."""
        return CONFIG

    def load_model(self):
        """Load the appropriate Qwen-VL variant based on model path.

        Returns:
            Initialized Qwen-VL causal model on the selected device.
        """
        if "qwen2.5".lower() in str(self.model_path).lower():
            print("Loading Qwen2.5-VL from pretrained")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        else:
            print("Loading Qwen2-VL from pretrained")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        return model

    def load_processor(self):
        """Load the processor with pixel constraints for Qwen-VL."""
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            max_pixels=self.max_pixels,
        )
        return processor

    def generate_prompt_message(
        self, text_prompt, video_paths
    ) -> list[dict[str, object]]:
        """Build chat-style multimodal messages for Qwen-VL inference.

        Args:
            text_prompt: User text prompt.
            video_paths: Ordered list of video file paths to include.

        Returns:
            Message payload consumable by the processor.
        """
        content = []
        for vpath in video_paths:
            content.append(
                {
                    "type": "video",
                    "video": vpath,
                    "fps": self.fps,
                }
            )
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
        return messages

    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens):
        """Run generation on Qwen-VL with provided videos and prompt.

        Args:
            text_prompt: Text question or instruction.
            video_paths: List of video paths to condition on.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple of decoded model outputs and elapsed time string.

        Raises:
            RuntimeError: If GPU inference fails.
        """
        print("Begin Inference...")
        messages = self.generate_prompt_message(
            text_prompt=text_prompt, video_paths=video_paths
        )

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        inputs = inputs.to(self.device)

        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"Model response: {output_text}")
        end = time.perf_counter()

        del inputs, video_inputs, image_inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"


class VideoLlava:
    def __init__(self, model_path: str):
        """Initialize Video-LLaVA model, processor, and frame sampling config.

        Args:
            model_path: Filesystem path or model id for weights.
        """
        self.model_path = model_path
        self.model_path = model_path

        config = self.load_config()
        cfg = config.get("video-llava", {})

        self.fps = 8
        self.num_frames = cfg.get("num_frames")
        self.max_pixels = ""
        self.min_pixels = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()

    def load_config(self):
        """Return static configuration for supported models."""
        return CONFIG

    def load_model(self):
        """Load Video-LLaVA generation model."""
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).to(self.device)
        return model

    def load_processor(self):
        """Load Video-LLaVA processor without rescaling."""
        processor = VideoLlavaProcessor.from_pretrained(
            self.model_path, do_rescale=False
        )
        return processor

    def read_video_pyav(self, container, indices):
        """Read selected frames from a PyAV container.

        Args:
            container: Opened PyAV container.
            indices: Frame indices to extract.

        Returns:
            Array of RGB frames matching requested indices.
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_clip_frames(self, video_path: str) -> np.ndarray:
        """Sample frames uniformly from a video file.

        Args:
            video_path: Path to the video to sample.

        Returns:
            Numpy array of sampled RGB frames.

        Raises:
            RuntimeError: If no frames can be read from the video.
        """
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = getattr(stream, "frames", None)
        print(f"Total Frames for Video: {total_frames}")
        if not total_frames or total_frames <= 0:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            total_frames = len(frames)
            if total_frames == 0:
                raise RuntimeError(f"No frames found in {video_path}")
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
            clip = np.stack([frames[i] for i in indices])
        else:
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
            clip = self.read_video_pyav(container, indices.tolist())
        container.close()
        return clip

    def generate_prompt_message(self, text_prompt) -> str:
        """Build text-only prompt for Video-LLaVA generation."""
        messages = f"USER: <video>\n{text_prompt}\nASSISTANT:"
        return messages

    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens):
        """Run Video-LLaVA inference over one or more videos.

        Args:
            text_prompt: Text question or instruction.
            video_paths: List of video paths to process.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple of decoded output text and elapsed time string.

        Raises:
            RuntimeError: If video sampling or generation fails.
        """
        if len(video_paths) > 1:
            clips_data = [self.sample_clip_frames(vp) for vp in video_paths]
            print(
                "Warning: model not explicitly trained for multi-video per prompt; results may be unpredictable."
            )
            clip = np.concatenate(clips_data, axis=0)
        else:
            container = av.open(video_paths[0])
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(
                int
            )
            print(f"Total Frames for Video: {total_frames} --- {len(indices)}")
            clip = self.read_video_pyav(container, indices)

        messages = self.generate_prompt_message(text_prompt=text_prompt)

        inputs = self.processor(text=messages, videos=clip, return_tensors="pt")
        inputs = inputs.to(self.device)

        print("Inference Start")
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"Model response: {output_text}")
        end = time.perf_counter()
        print("Inference End")

        del inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"


class VideoLLama3:
    def __init__(self, model_path: str):
        """Initialize Video-LLaMA3 model and preprocessing settings.

        Args:
            model_path: Filesystem path or model id for weights.
        """
        self.model_path = model_path

        config = self.load_config()
        cfg = config.get("video-llama-3", {})

        self.fps = float(cfg.get("fps", 1.0))
        self.max_frames = int(cfg.get("max_frames", 150))
        self.target_short_side = int(cfg.get("target_short_side", 64))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()

    def load_config(self):
        """Return static configuration for supported models."""
        return CONFIG

    def load_model(self):
        """Load Video-LLaMA3 model with remote code enabled."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    def load_processor(self):
        """Load the processor for Video-LLaMA3."""
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        return processor

    def generate_prompt_message(
        self, text_prompt, video_paths
    ) -> list[dict[str, object]]:
        """Compose chat conversation with embedded video references.

        Args:
            text_prompt: User text prompt.
            video_paths: List of video file paths to include.

        Returns:
            Message payload for Video-LLaMA3 conversation API.
        """
        content = []
        for vpath in video_paths:
            content.append(
                {
                    "type": "video",
                    "video": {
                        "video_path": vpath,
                        "fps": self.fps,
                        "max_frames": self.max_frames,
                        "size": self.target_short_side,
                    },
                }
            )
        content.append({"type": "text", "text": text_prompt})
        messages = [
            {"role": "system", "content": "You are a video answering assistant."},
            {"role": "user", "content": content},
        ]
        return messages

    def run_inference(
        self, text_prompt: str, video_paths: list[str], max_new_tokens: int = 128
    ):
        """Run Video-LLaMA3 generation on provided videos and prompt.

        Args:
            text_prompt: Text question or instruction.
            video_paths: List of video paths to process.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple of generated text and elapsed time string.
        """
        messages = self.generate_prompt_message(text_prompt, video_paths)
        inputs = self.processor(conversation=messages, return_tensors="pt")
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        print("Inference Start")
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        print(f"Model response: {output_text}")
        end = time.perf_counter()
        print("Inference End")
        del inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"


class InternVL3:
    def __init__(self, model_path: str):
        """Initialize InternVL3 model, tokenizer, and preprocessing settings.

        Args:
            model_path: Filesystem path or model id for weights.
        """
        self.model_path = model_path
        self.model_path = model_path

        config = self.load_config()
        cfg = config.get("intern-vl-3", {})

        self.num_segments = cfg.get("num_segments", 16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.pixel_values = None
        self.video_prefix = None

    def load_config(self):
        """Return static configuration for supported models."""
        return CONFIG

    def load_model(self):
        """Load InternVL3 model with optimized settings for inference.

        Returns:
            Initialized model ready for generation.
        """
        model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval()
        return model

    def load_tokenizer(self):
        """Load tokenizer with remote code support for InternVL3."""
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True, use_fast=False
        )
        return tokenizer

    @staticmethod
    def build_transform(input_size):
        """Build image preprocessing pipeline for InternVL3.

        Args:
            input_size: Target square resolution for resized tiles.

        Returns:
            Torchvision transform converting PIL images to normalized tensors.
        """
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        return transform

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    ):
        """Select the target grid ratio closest to the input aspect ratio.

        Args:
            aspect_ratio: Width/height of the input image.
            target_ratios: Candidate grid ratios.
            width: Input width.
            height: Input height.
            image_size: Target tile size.

        Returns:
            Tuple representing the chosen grid (cols, rows).
        """
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        """Tile an image into multiple patches while preserving aspect ratio.

        Args:
            image: PIL image to process.
            min_num: Minimum number of tiles.
            max_num: Maximum number of tiles.
            image_size: Square size for each tile.
            use_thumbnail: Whether to append a thumbnail tile.

        Returns:
            List of processed PIL image tiles.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = InternVL3.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        """Compute frame indices for uniform sampling across a clip.

        Args:
            bound: Optional (start, end) bounds in seconds.
            fps: Video frames per second.
            max_frame: Maximum frame index in the video.
            first_idx: Starting frame index.
            num_segments: Number of segments to sample.

        Returns:
            Numpy array of selected frame indices.
        """
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    @staticmethod
    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        """Load and preprocess frames from a video into tensor patches.

        Args:
            video_path: Path to the video file.
            bound: Optional (start, end) seconds to crop.
            input_size: Target tile size for preprocessing.
            max_num: Maximum number of tiles per frame.
            num_segments: Number of frames to sample.

        Returns:
            Tuple of pixel tensor and list of patch counts per frame.
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = InternVL3.build_transform(input_size=input_size)
        frame_indices = InternVL3.get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = InternVL3.dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    @staticmethod
    def process_video(video_path: str, num_segments: int):
        """Process a video into pixel tensors and prompt prefix.

        Args:
            video_path: Path to the video file.
            num_segments: Number of frames to sample.

        Returns:
            Tuple of pixel tensor, patch counts, and formatted video prefix string.
        """
        pixel_values, num_patches_list = InternVL3.load_video(
            str(video_path), num_segments=num_segments, max_num=1
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
        )
        return pixel_values, num_patches_list, video_prefix

    def generate_prompt_message(self, text_prompt: str, video_prefix: str) -> str:
        """Prepend video placeholders to the user prompt text."""
        return f"{video_prefix}{text_prompt}"

    def run_inference(
        self, text_prompt: str, video_paths: list[str], max_new_tokens: int = 128
    ):
        """Run InternVL3 inference on a single video clip.

        Args:
            text_prompt: Text question or instruction.
            video_paths: List containing a single video path.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple of generated response text and elapsed time in seconds.
        """
        assert len(video_paths) == 1
        pixel_values, num_patches_list, video_prefix = InternVL3.process_video(
            video_paths[0], self.num_segments
        )
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)
        question = self.generate_prompt_message(text_prompt, video_prefix)

        print("Inference Start")
        start_time = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            response, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        end_time = time.perf_counter()
        print(f"Model Response: {response}")
        print("Inference End")
        elapsed_time = end_time - start_time
        print(f"Script execution time: {elapsed_time:.4f} seconds")
        del history
        torch.cuda.empty_cache()
        return response, elapsed_time


MODEL_CLASSES: dict[str, type] = {
    "qwen2.5-vl-7b-instruct": QwenVL,
    "video-llava-7b": VideoLlava,
    "videollama3-7b": VideoLLama3,
    "internvl3-8b": InternVL3,
}


class EpisodicInference:
    def __init__(
        self,
        model,
        episodic_tuples_dir,
        qa_pairs_dir,
        video_dir,
        output_path: str,
        with_subs: bool,
        all_subs: bool,
        with_context: bool,
        run_id: str,
        run_datetime: str,
        checkpoint: str | None = None,
    ):
        """Coordinate running inference across QA datasets and videos.

        Args:
            model: Model instance implementing `run_inference`.
            episodic_tuples_dir: Directory containing episode tuples with events.
            qa_pairs_dir: Directory or file containing QA JSON data.
            video_dir: Root directory of video clips.
            output_path: Directory to save inference results.
            with_subs: Include subtitle context in prompts.
            all_subs: Use all subtitles instead of event-specific ones.
            with_context: Prepend show-specific context to questions.
            run_id: Identifier for the current run.
            run_datetime: Timestamp string for output organization.
            checkpoint: Optional index to resume QA processing.
        """
        self.episodic_tuples_dir = episodic_tuples_dir
        self.qa_pairs_dir = qa_pairs_dir
        self.video_dir = video_dir
        self.model = model
        self.output_path = output_path
        self.with_subs = with_subs
        self.all_subs = all_subs
        self.with_context = with_context
        self.responses = []
        self.run_id = run_id
        self.run_datetime = run_datetime

        self.qa_pairs = self.load_question_answer_pairs(qa_pairs_dir, checkpoint)

    def load_question_answer_pairs(self, qa_pairs_dir, checkpoint) -> list[dict]:
        """Load QA pairs from a JSON file or directory.

        Args:
            qa_pairs_dir: Path to QA JSON file or directory.
            checkpoint: Optional starting index for resuming.

        Returns:
            List of QA dictionaries.

        Raises:
            FileNotFoundError: If the provided path does not exist.
        """
        path = Path(qa_pairs_dir)
        entries: list[dict[str, object]] = []
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
        elif path.is_dir():
            for f in sorted(path.glob("*.json")):
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    entries += data if isinstance(data, list) else [data]
        else:
            raise FileNotFoundError(f"QA file/dir not found: {path}")

        if checkpoint is not None:
            entries = entries[int(checkpoint) :]

        return entries

    def extract_video_clips_paths(self, qa_json: dict) -> list:
        """Resolve video clip paths from QA metadata."""
        video_paths = []
        episodes = qa_json.get("episode_span", [])
        for ep in episodes:
            season_id = int(ep[:2])
            clip_id = int(ep[2:])
            clip_path = (
                Path(self.video_dir) / f"season_{season_id}" / f"episode_{clip_id}.mp4"
            )
            video_paths.append(str(clip_path))
        return video_paths

    def extract_subtitles(self, qa_json: dict, episode_id: int) -> list:
        """Retrieve subtitle dialogue associated with QA events or entire episode.

        Args:
            qa_json: QA item containing event references.
            episode_id: Episode identifier to load subtitles for.

        Returns:
            List of collected dialogue strings.

        Raises:
            FileNotFoundError: If the episodic tuple file is missing.
            AssertionError: If the episode id in file does not match expectation.
        """
        episodic_tuples_path = (
            Path(self.episodic_tuples_dir) / f"{episode_id}_with_events.json"
        )
        if not episodic_tuples_path.exists():
            raise FileNotFoundError(
                f"Episodic tuple not found for episode {episode_id}: {episodic_tuples_path}"
            )

        with open(episodic_tuples_path, "r", encoding="utf-8") as f:
            episode_tuple = json.load(f)
        assert (
            episode_tuple.get("episode") == episode_id
        ), f"Episode mismatch: expected {episode_id}, got {episode_tuple.get('episode')}"
        subtitles_output = []
        related_events = qa_json.get("related_events", [])
        collected_dialogues = []

        if self.all_subs:
            for scene in episode_tuple.get("scenes", []):
                for event_dialogue in scene.get("dialogue", []):
                    speaker = event_dialogue.get("speaker", "").strip()
                    text = event_dialogue.get("text", "").strip()
                    if speaker and text:
                        collected_dialogues.append(f"{speaker}: {text}")
        else:
            for scene in episode_tuple.get("scenes", []):
                scene_event_ids = [e["event_id"] for e in scene.get("events", [])]
                if any(event in scene_event_ids for event in related_events):
                    for event_dialogue in scene.get("dialogue", []):
                        speaker = event_dialogue.get("speaker", "").strip()
                        text = event_dialogue.get("text", "").strip()
                        if speaker and text:
                            collected_dialogues.append(f"{speaker}: {text}")

        subtitles_output.append(collected_dialogues)
        return subtitles_output

    def build_prompt_text(self, qa_json: dict, episode_id: int = None) -> tuple[str, str]:
        """Assemble the text prompt and options for a QA item.

        Args:
            qa_json: QA item containing question and options.
            episode_id: Episode identifier to fetch subtitles if requested.

        Returns:
            Tuple of prompt string and cleaned options text.
        """
        prompt_parts = []
        instruction = "Respond only with the number of the correct multiple-choice answer option (e.g., A)."

        if self.with_subs:
            subs = self.extract_subtitles(qa_json, episode_id)
            subs_text = str(subs)

        def clean_options_text(qa_json: dict):
            options = qa_json.get("options", [])
            cleaned_options = []
            for opt in options:
                cleaned = opt.strip()
                cleaned = cleaned.lstrip("ABCD1234").lstrip(". ").strip()
                cleaned_options.append(cleaned)
            labels = ["A", "B", "C", "D"]
            options_text = "\n".join(
                f"{labels[i]}. {opt}" for i, opt in enumerate(cleaned_options)
            )
            return options_text

        question = qa_json.get("question", "")
        if self.with_context is True:
            question = f"""
                This is an episode from the American television sitcom Friends created by David Crane and Marta Kauffman. Answer the following questions regarding this episode of Friends. \n{question}
            """
        options_text = clean_options_text(qa_json)

        if self.with_subs:
            prompt_parts.append(
                f"\n\nVideo Context (Subtitles):\n{subs_text}\n\n"
                f"Question:\n{question}\n\n"
                f"Answer Options:\n{options_text}\n"
                f"{instruction}\n\n"
            )
        else:
            prompt_parts.append(
                f"Question:\n{question}\n\n"
                f"Answer Options:\n{options_text}\n"
                f"{instruction}\n\n"
            )

        prompt = "\n\n".join(prompt_parts)
        return prompt, options_text

    def generate_batch_run(self, max_new_tokens: int = 128) -> list[dict]:
        """Run inference for all QA items and persist results to disk.

        Args:
            max_new_tokens: Maximum tokens for model generation.

        Returns:
            List of QA records with model responses.

        Raises:
            RuntimeError: If output path is not provided.
        """
        if self.output_path is None:
            raise RuntimeError("Output Save Path for Results is Not Specified")
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        print(f"\n{'='*80}")
        print(f"Batch Job Inference: Initializing --- {now_str}")
        print(f"{'='*80}\n")

        model_name = str(self.model.model_path).split("/")[-1]

        for idx, qa_json in enumerate(
            tqdm(self.qa_pairs, desc="Episodic Inference Batch Run")
        ):
            results = []
            episode_id = "finalised_qa"
            print(f"\n{'='*80}")
            print(
                f"Generating Batch Inference - Index: {idx} --- Episode: {episode_id}"
            )
            print(f"{'='*80}\n")
            qa_dataset = qa_json.get("qa_dataset", [])

            for idx, qa_data in enumerate(
                tqdm(qa_dataset, desc=f"Running QA Dataset - {episode_id}")
            ):
                question_id = qa_data.get("question_id")
                text_prompt, options_text = self.build_prompt_text(qa_data)
                video_paths = self.extract_video_clips_paths(qa_data)

                print(f"\n============{question_id}============")
                print(f"\n{qa_data}\n")
                response = {
                    "model": model_name,
                    "question": qa_data["question"],
                    "question_id": qa_data["question_id"],
                    "question_type": qa_data["question_type"],
                    "ground_truth_answer": qa_data["answer"],
                    "options_text": options_text,
                    "model_response": "",
                    "with_context": str(self.with_context),
                    "with_subs": str(self.with_subs),
                    "all_subs": str(self.all_subs),
                    "execution_time": "",
                    "video_clips": "",
                    "error": "",
                }
                try:
                    print(f"Videos to run inference on: {video_paths}")
                    generated_result, execution_time = self.model.run_inference(
                        text_prompt=text_prompt,
                        video_paths=video_paths,
                        max_new_tokens=max_new_tokens,
                    )
                    response["model_response"] = generated_result
                    response["execution_time"] = execution_time
                    response["video_clips"] = video_paths
                except torch.cuda.OutOfMemoryError as e:
                    response["error"] = f"{e}"
                    continue
                except Exception as e:
                    response["error"] = f"{e}"
                    raise

                qa_data["result"] = response
                results.append(qa_data)

            output_path = (
                Path(self.output_path)
                / model_name
                / f"{self.run_datetime}"
                / f"run_{self.run_id}"
            )
            output_path.mkdir(exist_ok=True, parents=True)
            save_path = (
                output_path
                / f"results_{Path(self.qa_pairs_dir).name}_{episode_id}_{now_str}.json"
            )

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"{self.qa_pairs_dir} Saved results to: {save_path}")
            print(f"Ending Batch Inference - Index: {idx} --- Episode: {episode_id}")
            torch.cuda.empty_cache()

        return results


@dataclass
class InferenceSettings:
    checkpoint: str | None = None
    qa_dir: str = "./data/qa_output/"
    tuples_dir: str = "./data/output_with_events/"
    video_dir: str = "./data/video/"
    output_dir: str = "./results_vanilla/"
    with_context: bool = False
    with_subs: bool = False
    all_subs: bool = False
    models: list[str] | None = None
    runs: int = 3
    max_new_tokens: int = 128


def resolve_model_paths(requested_models: list[str]) -> dict[str, str]:
    """Resolve requested model identifiers to filesystem paths.

    Args:
        requested_models: List of model keys or key=path overrides.

    Returns:
        Mapping from model key to resolved path.

    Raises:
        ValueError: If an unknown model key is provided.
    """
    resolved: dict[str, str] = {}
    for name in requested_models:
        if name in DEFAULT_MODEL_PATHS:
            resolved[name] = DEFAULT_MODEL_PATHS[name]
        elif "=" in name:
            key, path = name.split("=", 1)
            resolved[key.strip()] = path.strip()
        else:
            raise ValueError(
                f"Unknown model key '{name}'. Known: {', '.join(DEFAULT_MODEL_PATHS)}"
            )
    return resolved


def run_inference_for_model(
    model_name: str, model_path: str, run_ids: list[int], settings: InferenceSettings
) -> None:
    """Load a model and execute batch inference for specified run ids.

    Args:
        model_name: Key of the model to run.
        model_path: Filesystem path for the model weights.
        run_ids: List of run identifiers to execute.
        settings: Inference settings bundle.

    Raises:
        ValueError: If the model key has no registered class.
    """
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    model_class = MODEL_CLASSES.get(model_name)
    if model_class is None:
        raise ValueError(f"No model class found for {model_name}")

    for run_id in run_ids:
        print(f"\n{'='*80}")
        print(
            f"Running Inference for model: {model_name}  -- {run_datetime} -- Run ID: {run_id}"
        )
        print(f"Model path: {model_path}")
        print(f"{'='*80}\n")

        gc.collect()
        torch.cuda.empty_cache()
        print("Torch CUDA Cache Emptied")

        print(f"Inference: Initializing Model {model_path}")
        model = model_class(model_path=model_path)
        print(f"Inference: Model Loaded from {model_path}")

        print("Inference: Begin Run")
        inference = EpisodicInference(
            model=model,
            episodic_tuples_dir=settings.tuples_dir,
            qa_pairs_dir=settings.qa_dir,
            video_dir=settings.video_dir,
            output_path=settings.output_dir,
            with_subs=settings.with_subs,
            all_subs=settings.all_subs,
            with_context=settings.with_context,
            run_id=str(run_id),
            run_datetime=run_datetime,
            checkpoint=settings.checkpoint,
        )
        inference.generate_batch_run(max_new_tokens=settings.max_new_tokens)


def vanilla_pipeline(
    checkpoint: str | None = None,
    qa_dir: str = "./data/qa_output/",
    tuples_dir: str = "./data/output_with_events/",
    video_dir: str = "./data/video/",
    output_dir: str = "./results_vanilla/",
    with_context: bool = False,
    with_subs: bool = False,
    all_subs: bool = False,
    models: list[str] | None = None,
    runs: int = 3,
    max_new_tokens: int = 128,
) -> None:
    """Entry point to run episodic inference across configured models.

    Args:
        checkpoint: Optional QA index to resume processing from.
        qa_dir: Directory of QA datasets.
        tuples_dir: Directory of episode tuples with events.
        video_dir: Directory of video clips.
        output_dir: Destination directory for results.
        with_context: Include show context in prompts.
        with_subs: Include subtitles for related events.
        all_subs: Include all subtitles rather than only related scenes.
        models: Optional list of model keys or key=path overrides.
        runs: Number of runs per model.
        max_new_tokens: Maximum tokens to generate per inference call.
    """
    settings = InferenceSettings(
        checkpoint=checkpoint,
        qa_dir=qa_dir,
        tuples_dir=tuples_dir,
        video_dir=video_dir,
        output_dir=output_dir,
        with_context=with_context,
        with_subs=with_subs,
        all_subs=all_subs,
        models=models,
        runs=runs,
        max_new_tokens=max_new_tokens,
    )

    requested_models = models or list(DEFAULT_MODEL_PATHS.keys())
    model_paths = resolve_model_paths(requested_models)
    run_ids = list(range(1, runs + 1))

    for model_name, model_path in model_paths.items():
        run_inference_for_model(model_name, model_path, run_ids, settings)
