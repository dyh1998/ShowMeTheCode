import logging
import os

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# train and valid的dataset  evaluate的时候需要重新写
class SummarySquadDataset(Dataset):
    def __init__(self, args, file_name="cached-train-cnn-features-512"):
        features_dir = args.sum_features_data_dir
        features_path = os.path.join(features_dir, file_name)
        logger.info(f"Loading features from cached dir -{features_path}")
        self.pad_token_id = args.pad_token_id
        self.features = torch.load(features_path)

    def __getitem__(self, item):
        feature = self.features[item]
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        target_ids = feature.target
        segment_ids = feature.segment_ids
        start_positions = feature.start_positions
        end_positions = feature.end_positions
        sentence_start_positions = feature.sentence_start_positions
        sentence_end_positions = feature.sentence_end_positions

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "segment_ids": segment_ids,
            "target_ids": target_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "sentence_start_positions": sentence_start_positions,
            "sentence_end_positions": sentence_end_positions
        }
        return input_dict

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch):
        """将batch中的list进行stack 得到batch tensor形式"""
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        segment_ids = torch.stack([torch.tensor(x["segment_ids"]) for x in batch])
        target_ids = torch.stack([torch.tensor(x["target_ids"]for x in batch)])
        start_positions = torch.stack([torch.tensor(x["start_positions"]) for x in batch])
        end_positions = torch.stack([torch.tensor(x["end_positions"]) for x in batch])

        sentence_start_positions = torch.stack([torch.tensor(x["sentence_start_positions"] for x in batch)])
        sentence_end_positions = torch.stack([torch.tensor(x["sentence_end_positions"] for x in batch)])
        input_ids, attention_mask, segment_ids, start_positions, end_positions = self.trim_batch(
            input_ids,
            attention_mask,
            segment_ids,
            start_positions,
            end_positions
        )
        target_ids = self.trim_target_batch(target_ids)

        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "segment_ids": segment_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "sentence_start_positions": sentence_start_positions,
            "sentence_end_positions": sentence_end_positions
        }

        return batch_inputs   # 返回一个训练batch

    def trim_batch(self, input_ids, attention_mask, segment_ids, start_positions, end_positions):
        """去除掉input_ids的batch中全为0的列"""
        keep_column_mask = input_ids.ne(self.pad_token_id).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        segment_ids = segment_ids[:, keep_column_mask]
        start_positions = start_positions[:,  keep_column_mask]
        end_positions = end_positions[:,  keep_column_mask]

        return input_ids, attention_mask, segment_ids, start_positions, end_positions

    def trim_target_batch(self, target_ids):
        keep_column_mask = target_ids.ne(self.pad_token_id).any(dim=0)
        target_ids = target_ids[:, keep_column_mask]
        return target_ids


def get_SummarySquad_dataloader(args, file_name, type_data):

    dataset = SummarySquadDataset(args, file_name=file_name)
    if type_data == "train":
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=dataset.collate_fn
        )
        return train_dataloader
    else:
        valid_sampler = SequentialSampler(dataset)
        valid_dataloader = DataLoader(
            dataset,
            sampler=valid_sampler,
            batch_size=args.eval_batch_size_squad,
            collate_fn=dataset.collate_fn
        )
        return valid_dataloader