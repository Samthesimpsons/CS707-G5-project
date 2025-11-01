import torch
import numpy as np
import av 
import time

from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModel,
    AutoImageProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    VideoLlavaForConditionalGeneration, 
    VideoLlavaProcessor
)
from qwen_vl_utils import process_vision_info

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


### ======================= Qwen2.5-VL-7B-Instruct =======================
class QwenVL:
    def __init__ (self, model_path: str):
        self.model_path = model_path
        self.fps = 2.0
        self.max_pixels = 128 * 28 * 28
        self.min_pixels = 64 * 28 * 28
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()


    def load_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            # device_map="auto"
        ).to(self.device)
        return model


    def load_processor(self):
        processor = AutoProcessor.from_pretrained(
            self.model_path, 
            max_pixels = self.min_pixels,
            do_rescale = False
        )
        return processor
        

    def generate_prompt_message(self, text_prompt, video_paths) -> str:
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
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
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
            # fps=FPS,
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
    def __init__ (self, model_path: str):
        self.model_path = model_path
        self.fps = 8
        self.num_frames = 8 # used for sampling the number of frames
        self.max_pixels = ""
        self.min_pixels = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()
    

    def load_model(self):
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            # device_map="auto"
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
            # sample uniformly 8 frames from the video
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
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


### ======================= DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base =======================
class VideoLLama2:
    def __init__ (self, model_path: str):
        self.model_path = model_path
        self.fps = 8
        self.num_frames = 8 # used for sampling the number of frames
        self.max_pixels = ""
        self.min_pixels = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.tokenizer  = self.init_model()

    def init_model(self):
        model, processor, tokenizer = model_init(self.model_path)
        return model, processor, tokenizer
        

    def generate_prompt_message(self, text_prompt, video_paths) -> str:
        content = []
        pass


    ### Video Inference Only -- No Audio Inference
    def run_inference(self, text_prompt: str, video_paths: list[str], max_new_tokens: int = 128, modality: str = "video"):
        
        if video_paths.isinstance(list):
            assert len(video_paths) == 1
            video_path = video_paths[0]
        
        ### Prepare for Inference
        # preprocess = self.processor[modality]
        # inputs = preprocess(video_path)
        inputs = self.processor[modality](self.modal_path)

        ### Run Generation of Inputs
        print("Inference Start")
        start = time.perf_counter()
        
        output_text = mm_infer(
            inputs,
            text_prompt, 
            model=self.model, 
            tokenizer=self.tokenizer, 
            do_sample=False, 
            modal=modality
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
    def __init__ (self, model_path: str):
        self.model_path = model_path
        self.fps = 1.0
        self.num_frames = 1.0 # used for sampling the number of frames
        self.max_frames = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.processor = self.load_processor()


    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
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
                        "max_frames": self.max_frames
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
