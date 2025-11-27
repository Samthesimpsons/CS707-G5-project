import json

from torch.utils.data import Dataset



class UnnaturalDataset(Dataset):
    def __init__(self, text_processor, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        # with open(ann_path, 'r') as f:
        #     for data in f.readlines():
        #         data = json.loads(data)
        #         self.ann.append(data)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]["instances"][0]
        instruction = info["instruction_with_input"]
        constraints = info["constraints"]
        answer = info["output"]
        if constraints != None:
            instruction = instruction+" "+constraints

        return {
            # "image":None,
            "instruction_input": instruction,
            "answer": answer,
        }
