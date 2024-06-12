# ASAP++ essay dataset용 코드
# argument mining label 사용시 am_flag를 사용

import torch
from torch.utils.data import Dataset

class AsapDataset(Dataset):

    def __init__(self, data, am_flag):
        self.data = data
        self.am_flag = am_flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        essay_id = example["essay_id"]
        prompt_id = example["prompt_id"]
        content_text = example["content_text"]
        scores = example["scores"]

        if self.am_flag:
            amlabel = example["amlabel"]
            amsent = example['amsent']
            return {
                "index": index,
                "essay_id": essay_id,
                "prompt_id": prompt_id,
                "content_text": content_text,
                "amlabel": amlabel,   # 추가
                "scores": scores,
                'amsent':amsent
            }
        else:
            return {
                "index": index,
                "essay_id": essay_id,
                "prompt_id": prompt_id,
                "content_text": content_text,
                "scores": scores
            }

class AsapDatasetCollator(object):

    def __init__(self, tokenizer, max_length, am_flag):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.am_flag = am_flag

    def __call__(self, batch):

        # 에세이 정보
        index = torch.tensor([ex["index"] for ex in batch])
        essay_id = torch.tensor([int(ex["essay_id"]) for ex in batch])
        prompt_id = torch.tensor([int(ex["prompt_id"]) for ex in batch])

        # 에세이 점수
        scores = [ex["scores"] for ex in batch]

        # 에세이 점수 마스킹 TODO(jin): 깔끔하게 줄일 것
        score_masking = []
        for score in scores:
            masking = []
            for s in score:
                masking.append(0 if s == 100 else 1)
            score_masking.append(masking)

        scores = torch.tensor(scores)
        score_masking = torch.tensor(score_masking)



        # 에세이 텍스트 인코딩
        content_text = []
        attention_mask = []

        for i, ex in enumerate(batch):
            tmp_content = [i for i in ex['content_text']][:self.max_length]
            tmp_mask = [1] * len(tmp_content)
            padding = [1] * (self.max_length - len(tmp_content))      # padding: 100 추가
            mask_pad = [0] * (self.max_length - len(tmp_content))
            content = tmp_content + padding
            mask = tmp_mask + mask_pad
            content_text.append(content)
            attention_mask.append(mask)
        
        input_ids = torch.tensor(content_text)
        input_masks = torch.tensor(attention_mask).bool()

        if self.am_flag:
            amlabels = []
            for i, ex in enumerate(batch):
                tmp_amlabel = [i+1 for i in ex['amlabel']][:self.max_length]
                padding = [0] * (len(input_ids[i, :]) - len(tmp_amlabel))      # padding: 100 추가
                amlabel = tmp_amlabel + padding
                amlabels.append(amlabel)
            amlabels = torch.tensor(amlabels, dtype=torch.long)

            amsents = []
            for i, ex in enumerate(batch):
                tmp_amsent = [i+1 for i in ex['amsent']][:30]
                padding = [0] * (30 - len(tmp_amsent))      # padding: 100 추가
                amsent = tmp_amsent + padding
                amsents.append(amsent)
            amsents = torch.tensor(amsents, dtype=torch.long)

            return (index, essay_id, prompt_id, input_ids, input_masks, amlabels, scores, score_masking, amsents)
        else:
            # (인덱스, 에세이 인덱스, 프롬프트 인덱스, 인풋, 마스크, 점수, trait 마스킹)
            return (index, essay_id, prompt_id, input_ids, input_masks, scores, score_masking)
