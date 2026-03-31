import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim import AdamW

print("Loading dataset...")

df = pd.read_csv("train.csv")
df = df.dropna()


def convert_label(row):

    if row["identity_hate"] == 1 or row["threat"] == 1:
        return 2
    
    elif row["severe_toxic"] == 1:
        return 2
    

    elif row[["toxic","obscene","insult"]].max() == 1:
        return 1
    
    else:
        return 0

df["label"] = df.apply(convert_label, axis=1)

df = df[["comment_text","label"]]
df.rename(columns={"comment_text":"text"}, inplace=True)


df["text"] = df["text"].str.lower()


df = df.sample(frac=1).reset_index(drop=True)

print("Dataset size:", df.shape)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.1, stratify=df["label"]
)


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

class HateDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = HateDataset(train_texts, train_labels)
val_dataset = HateDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3
)

model.to(device)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=2e-5)

print("Training started...")

epochs = 3
best_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)


    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    acc = accuracy_score(actuals, preds)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Accuracy: {acc:.4f}")


    if acc > best_acc:
        best_acc = acc
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")
        print("🔥 Best model saved")

print(" Training Complete")