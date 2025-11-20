import torch
import numpy as np
import av 
import time
import tomllib

from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModel,
    AutoImageProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, 
    VideoLlavaForConditionalGeneration, 
    VideoLlavaProcessor
)
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from google import genai
from google.genai import types

# from videollama2 import model_init, mm_infer
# from videollama2.utils import disable_torch_init


### ======================= Qwen2.5-VL-7B-Instruct =======================
class QwenVL:
    def __init__ (self, model_path: str, config_path: str = "./models/config.toml"):
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        config = self.load_config()
        cfg = config.get("qwen-vl-25", {})

        # Assign configuration values with defaults
        self.fps = float(cfg.get("fps", 1.0))
        # self.max_channels = int(cfg.get("max_channels", 64))
        # self.min_channels = int(cfg.get("min_channels", 64))
        # self.width = int(cfg.get("width", 64))
        # self.height = int(cfg.get("height", 64))

        # Derived parameters
        self.max_pixels = 8*28*28 #1280*28*28
        self.min_pixels = 8*28*28

        # Configure processor parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()


    def load_config(self):
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

        
    def load_model(self):
        if "qwen2.5".lower() in str(self.model_path).lower():
            print("Loading Qwen2.5-VL from pretrained")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                # attn_implementation="flash_attention_2",
                # device_map="auto"
            ).to(self.device)
        else:
            print("Loading Qwen2-VL from pretrained")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                # device_map="auto"
            ).to(self.device)
        return model


    def load_processor(self):
        processor = AutoProcessor.from_pretrained(
            self.model_path, 
            # min_pixels = self.min_pixels,
            max_pixels = self.max_pixels,
            # do_rescale = False
        )
        return processor
        

    def generate_prompt_message(self, text_prompt, video_paths):
        content = []
        for vpath in video_paths:
            content.append(
                {
                    "type": "video",
                    "video": vpath,
                    "fps": self.fps,
                    # "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": text_prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": video_paths[0],
        #                 # "max_pixels": 360 * 420,
        #                 "fps": 1.0,
        #             },
        #             {"type": "text", "text": text_prompt},
        #         ],
        #     }
        # ]
        return messages
    

    ### ==========================================
    ### SINGLE AND MULTI-CLIP INFERENCE
    ### ==========================================
    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens):
        # clips_data = [self.sample_clip_frames(vp) for vp in video_paths]
        # clips_data = [self.sample_clip_frames(vp) for vp in video_paths]
        # combined_clip = np.concatenate(clips_data, axis=0) if len(clips_data) > 1 else clips_data[0]

        print("Begin Inference...")
        ### Qwen-VL2.5 does not need to handle clips separately
        messages = self.generate_prompt_message(
            text_prompt=text_prompt,
            video_paths=video_paths
        )
        
        ### Fetch Inference Inputs
        # Preparation for inference
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
            **video_kwargs
        )
        inputs = inputs.to(self.device)

        ### Run Generation of Inputs
        start = time.perf_counter()
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        print(f"Model response: {output_text}")
        end = time.perf_counter()

        # Free memory manually
        del inputs, video_inputs, image_inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"
        


### ======================= LanguageBind/Video-LLaVA-7B-hf =======================
class VideoLlava:
    def __init__ (self, model_path: str, config_path: str = "./models/config.toml"):
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        config = self.load_config()
        cfg = config.get("video-llava", {})

        # Configure parameters
        self.fps = 8
        self.num_frames = config.get("video-llava").get("num_frames") # used for sampling the number of frames
        self.max_pixels = ""
        self.min_pixels = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()
    

    def load_config(self):
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def load_model(self):
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            # device_map="auto",
            # attn_implementation="flash_attention_2"
        ).to(self.device)
        return model

    def load_processor(self):
        processor = VideoLlavaProcessor.from_pretrained(
            self.model_path,
            do_rescale=False
        )
        return processor


    ### ==========================================
    ### VIDEO CLIP PROCESSING
    ### ==========================================
    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`list[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
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
        """
        Open a video, sample frames, and return them as np.ndarray.
        Args:
            video_path
        Returns:
            result (np.ndarray): np array of clip frames
        """
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = getattr(stream, "frames", None)
        print(f"Total Frames for Video: {total_frames}")
        if not total_frames or total_frames <= 0:
            # fallback: decode all frames and sample
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
        messages = f"USER: <video>\n{text_prompt}\nASSISTANT:"
        return messages
    
    
    ### ==========================================
    ### SINGLE AND MULTI-CLIP INFERENCE
    ### ==========================================
    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens):
        if len(video_paths) > 1:
            clips_data = [self.sample_clip_frames(vp) for vp in video_paths]
            print("Warning: model not explicitly trained for multi-video per prompt; results may be unpredictable.")
            clip = np.concatenate(clips_data, axis=0)
        else:
            container = av.open(video_paths[0])
            # sample uniformly frames from the video
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
            print(f"Total Frames for Video: {total_frames} --- {len(indices)}")
            clip = self.read_video_pyav(container, indices)

        ### Generate Text Prompt
        messages = self.generate_prompt_message(
            text_prompt=text_prompt
        )
        
        ### Fetch Inference Inputs
        # Preparation for inference
        inputs = self.processor(
            text=messages,
            videos=clip,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        ### Run Generation of Inputs
        print("Inference Start")
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        print(f"Model response: {output_text}")
        end = time.perf_counter()
        print("Inference End")

        # Free memory manually
        del inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"
    

### ======================= DAMO-NLP-SG/VideoLLaMA3-7B =======================
class VideoLLama3:
    def __init__ (self, model_path: str, config_path: str = "./models/config.toml"):
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        config = self.load_config()
        cfg = config.get("video-llama-3", {})

        # Assign configuration values with defaults
        # https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/videollama3/mm_utils.py
        self.fps = float(cfg.get("fps", 1.0))
        self.max_frames = int(cfg.get("max_frames", 150))
        self.target_short_side = int(cfg.get("target_short_side", 64))

        # Configure processor parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()

    
    def load_config(self):
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

            
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        ).to(self.device)
        return model
    
    
    def load_processor(self):
        processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        return processor
    

    def generate_prompt_message(self, text_prompt, video_paths) -> str:
        content = []
        for vpath in video_paths:
            content.append(
                {
                    "type": "video",
                    "video": {
                        "video_path": vpath,
                        "fps": self.fps,
                        "max_frames": self.max_frames,
                        "size": self.target_short_side
                    }
                }
            )
        content.append({"type": "text", "text": text_prompt})
        messages = [
            {"role": "system", "content": "You are a video answering assistant."},
            {
                "role": "user",
                "content": content,
            }
        ]
        return messages


    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens: int=128):
        ### Prepare for Inference
        messages = self.generate_prompt_message(text_prompt, video_paths)
        inputs = self.processor(conversation=messages, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        ### Run Generation of Inputs
        print("Inference Start")
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"Model response: {output_text}")
        end = time.perf_counter()
        print("Inference End")
        # Free memory manually
        del inputs
        torch.cuda.empty_cache()
        elapsed_time = end - start
        return output_text, f"{elapsed_time:.4f}"


### =======================OpenGVLab/InternVL3-8B =======================
class InternVL3:
    def __init__ (self, model_path: str, config_path: str = "./models/config.toml"):
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        config = self.load_config()
        cfg = config.get("intern-vl-3", {})

        # Hyperparameters
        self.num_segments = cfg.get("num_segments", 16) # Total num. of frames

        # Configure processor parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.pixel_values = None
        self.video_prefix = None

    
    def load_config(self):
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

            
    def load_model(self):
        model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        ).eval()
        return model


    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), 
            trust_remote_code=True, 
            use_fast=False
        )
        return tokenizer

    @staticmethod
    def build_transform(input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
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
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
    
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVL3.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
        
    # video multi-round conversation (视频多轮对话)
    @staticmethod
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices


    @staticmethod
    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = InternVL3.build_transform(input_size=input_size)
        frame_indices = InternVL3.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = InternVL3.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    

    @staticmethod
    def process_video(video_path: str, num_segments: int):
        pixel_values, num_patches_list = InternVL3.load_video(
            str(video_path), 
            num_segments=num_segments, 
            max_num=1
        )
        # Build pixel values and video_prefix which are the list of frames from load_video()
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        return pixel_values, num_patches_list, video_prefix
                                                                        
    
    def generate_prompt_message(self, text_prompt: str, video_prefix: list) -> str:
        return video_prefix + text_prompt
        
    
    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens:int =128):
        assert len(video_paths) == 1
        pixel_values, num_patches_list, video_prefix = InternVL3.process_video(video_paths[0], self.num_segments)
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
                return_history=True
            )
        end_time = time.perf_counter()
        print(f'Model Response: {response}')
        print("Inference End")
        elapsed_time = end_time - start_time
        print(f"Script execution time: {elapsed_time:.4f} seconds")
        # Free memory manually
        del history
        torch.cuda.empty_cache()
        return response, elapsed_time
