from torch.nn.functional import cross_entropy
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        logits = model(input_ids, lengths)
        loss = cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
