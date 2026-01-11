import torch

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": padded_inputs,
        "labels": labels,
        "lengths": lengths,
    }
