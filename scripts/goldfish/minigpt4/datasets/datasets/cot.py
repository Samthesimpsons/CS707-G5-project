import json

from torch.utils.data import Dataset



class CoTDataset(Dataset):
    def __init__(self, text_processor, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        input = info["inputs"]
        target = info["targets"]
        return {
            "instruction_input": input,
            "answer": target,
        }
