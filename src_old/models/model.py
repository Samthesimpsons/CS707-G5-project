import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVL:
    def __init__ (self, model_path: str, video_input_path: str, text_prompt: str):
        self.model_path = model_path
        self.video_input_path = video_input_path
        self.text_prompt = text_prompt

        self.model, self.processor = self.load_model()
        self.params = self.load_parameters()

    
    def load_parameters(self) -> dict:
        params = {}
        return params


    def load_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            dtype=torch.bfloat16, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor


    def generate_prompt_message(self) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": self.video_input_path,
                        "max_pixels": self.params.get("max_pixels"),
                        "fps": self.params.get("fps"),
                    },
                    {
                        "type": "text", 
                        "text": self.text_prompt
                    },
                ],
            }
        ]
        return messages
    

    def fetch_inference_inputs(self):
        messages = self.generate_prompt_message()
        # Preparation for inference
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        # Pass inputs to processor
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        return inputs
    

    def run_inference(self):
        inputs = self.fetch_inference_inputs()
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_text
        

    
