import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
from tqdm import tqdm
import numpy as np
from dataset import ClaimsMaskedDataset

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
model = RobertaForMaskedLM.from_pretrained('distilroberta-base')
special_tokens = {'additional_special_tokens': ['<tab>']}
tokenizer.add_special_tokens(special_tokens)
print(f"Added <tab> special token (ID: {tokenizer.convert_tokens_to_ids('<tab>')})")
model.resize_token_embeddings(len(tokenizer))


with open('claims_final.tsv', 'r') as f:
    texts = f.readlines()

train_dataset = ClaimsMaskedDataset(texts, tokenizer)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

item = next(iter(dataloader))

assert item['labels'][(item['input_ids'] != tokenizer.mask_token_id)].float().mean().item() == -100, "Check the masking for loss in labels"
assert torch.all(item['input_ids'][(item['input_ids'] != tokenizer.mask_token_id)] >= 0), "Unmasked positions should not contain -ve"

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.train()

device

num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

np.exp(avg_loss)

import numpy as np
import torch.nn.functional as F

def sample_masked_token(input, logits, tokenizer, candidates=[]):
    logits = logits.detach().cpu()
    probs = F.softmax(logits, dim=-1)
    probs = probs.numpy()
    all_tokens = []
    if candidates:
        candidate_tokens = tokenizer(candidates, add_special_tokens=False)['input_ids']
        candidate_scores = []
    for i in range(len(input)):
        mask_ixs = [ix for ix, token in enumerate(input[i]) if token == 50264]
        if len(mask_ixs) == 0:
            continue
        prob = probs[i, mask_ixs, :]
        if candidates:
            cand_token_ids = candidate_tokens[i]
            if len(cand_token_ids) != len(mask_ixs):
                candidate_scores.append(0)
            else:
                p = 0
                for ic, c_t in enumerate(cand_token_ids):
                    p += probs[:, ic, c_t].squeeze()
                p /= len(cand_token_ids)
                candidate_scores.append(p)
        tokens = np.argmax(prob, axis=-1)
        all_tokens.append(tokens)
    if candidates:
        return tokenizer.batch_decode(all_tokens), candidate_scores
    return tokenizer.batch_decode(all_tokens)

model =model.to(device)

sample = texts[15]
samples, labels = train_dataset.mask_row(sample.replace('\t', '<tab>').strip('\n'))
with torch.no_grad():
    model.eval()
    logits = model(**tokenizer(samples, return_tensors='pt', padding='max_length').to(device))

sample

samples

sample_masked_token(tokenizer(samples)['input_ids'], logits.logits, tokenizer)

'<mask>'*len(tokenizer.tokenize('Hydrochlorothiazide 25 MG Oral Tablet'))

test_candidate = "active\tpharmacy\tnormal\t1\t<mask><mask><mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse\n"
test_candidate = test_candidate.replace("\t", "<tab>").strip("\n")
candidate = ['lisinopril 10 MG Oral Tablet']

logits = model(**tokenizer(test_candidate, return_tensors='pt', padding='max_length').to(device))

sample_masked_token(tokenizer([test_candidate])['input_ids'], logits.logits, tokenizer, candidates=candidate)

test_candidate = "active\tpharmacy\tnormal\t1\t<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse\n"
test_candidate = test_candidate.replace("\t", "<tab>").strip("\n")

candidate = ['Hydrochlorothiazide 25 MG Oral Tablet']

logits = model(**tokenizer(test_candidate, return_tensors='pt', padding='max_length').to(device))
sample_masked_token(tokenizer([test_candidate])['input_ids'], logits.logits, tokenizer, candidates=candidate)

test_candidate = "active\tprofessional\tnormal\t1\t<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse\n"
test_candidate = test_candidate.replace("\t", "<tab>").strip("\n")

candidate = ['Hydrochlorothiazide 25 MG Oral Tablet']

logits = model(**tokenizer(test_candidate, return_tensors='pt', padding='max_length').to(device))
sample_masked_token(tokenizer([test_candidate])['input_ids'], logits.logits, tokenizer, candidates=candidate)

