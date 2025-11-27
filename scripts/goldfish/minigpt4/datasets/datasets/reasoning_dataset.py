import os
import json

from PIL import Image
from torch.utils.data import Dataset




class ReasoningDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data = json.load(open(ann_path))

        # self.data = self.create_data(ann_path)

    # def create_data(self, ann_path):
    #     # processed_data = []
    #     with open(ann_path, 'r') as f:
    #         data = json.load(f)

    #     return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_id = sample["image_id"]+".jpg"
        question = sample["question"]
        answer =  sample["answer"]


        image = Image.open(os.path.join(self.vis_root, image_id)).convert("RGB")
        image = self.vis_processor(image)

        instruction = '<Img><ImageHere></Img> {} '.format(question)
    
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer
        }


