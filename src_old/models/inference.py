import torch
import numpy as np
import av

from model import QwenVL


print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU only")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


class EpisodicInference:
    def __init__ (
            self, model, processor, qa_pairs_dir, device: str
    ):
        self.qa_pairs_dir = qa_pairs_dir
        self.qa_pairs = self.load_question_answer_pairs(qa_pairs_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.processor = processor


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
    

    def load_question_answer_pairs(self, directory: str) -> dict:
        pass


    def generate(self, max_new_tokens: int=2048*2):
        generated_answers = []
        for idx, question in enumerate(self.qa_pairs):
            video_path = self.qa_pairs.get("video_path", "")
            container = av.open(video_path)

            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = self.read_video_pyav(container, indices)

            inputs = self.processor(text=question, videos=clip, return_tensors="pt")
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            generated_answers.append(
                {
                    "idx": idx,
                    "question": question,
                    "response": output_text
                }
            )
    
