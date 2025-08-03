import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output


class SiameseRoBERTa(nn.Module):
    def __init__(self, model_name, num_classes, num_heads=8):
        super(SiameseRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.cross_attention = CrossAttention(self.hidden_size, num_heads)
        self.classifier = nn.Linear(3 * self.hidden_size, num_classes)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.roberta(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.roberta(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state

        output1 = self.cross_attention(output1, output2, output2)
        output2 = self.cross_attention(output2, output1, output1)

        u = torch.mean(output1, dim=1)
        v = torch.mean(output2, dim=1)
        combined = torch.cat((u, v, torch.abs(u - v)), dim=1)
        logits = self.classifier(combined)
        return logits


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        Results_AC1 = item.get('Results_AC1', [])
        Results_AC2 = item.get('Results_AC2', [])
        ac1_text = item.get('AC1', "")
        ac2_text = item.get('AC2', "")

        prompt_1_ac1 = Results_AC1[0].get('prompt1', []) if Results_AC1 else []
        prompt_2_ac1 = Results_AC1[0].get('prompt2', []) if Results_AC1 else []
        prompt_1_ac2 = Results_AC2[0].get('prompt1', []) if Results_AC2 else []
        prompt_2_ac2 = Results_AC2[0].get('prompt2', []) if Results_AC2 else []

        # Join lists to strings
        prompt_1_ac1 = " ".join(prompt_1_ac1).strip()
        prompt_2_ac1 = " ".join(prompt_2_ac1).strip()
        prompt_1_ac2 = " ".join(prompt_1_ac2).strip()
        prompt_2_ac2 = " ".join(prompt_2_ac2).strip()

        # Fallback to raw AC text if prompts are missing
        if not prompt_1_ac1:
            prompt_1_ac1 = ac1_text
        if not prompt_2_ac1:
            prompt_2_ac1 = ac1_text
        if not prompt_1_ac2:
            prompt_1_ac2 = ac2_text
        if not prompt_2_ac2:
            prompt_2_ac2 = ac2_text

        Input1 = f"[CLS]{prompt_1_ac1} [SEP] {prompt_1_ac2}"
        Input2 = f"[CLS]{prompt_2_ac1} [SEP] {prompt_2_ac2}"
        label = item['Relation']

        tokenized_input1 = self.tokenizer(Input1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        tokenized_input2 = self.tokenizer(Input2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids1 = tokenized_input1["input_ids"].squeeze()
        attention_mask1 = tokenized_input1["attention_mask"].squeeze()
        input_ids2 = tokenized_input2["input_ids"].squeeze()
        attention_mask2 = tokenized_input2["attention_mask"].squeeze()

        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.tensor(label_encoded)

        return input_ids1, attention_mask1, input_ids2, attention_mask2, label_tensor


def train_model(model, train_loader, num_epochs=4, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids1, attention_mask1, input_ids2, attention_mask2, labels in train_loader:
            input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)
            input_ids2, attention_mask2 = input_ids2.to(device), attention_mask2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(train_loader):.4f}")
    return model


def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)['argument_data']


def run_final_training():
    base_dir = './data_src'
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    label_encoder = LabelEncoder()

    zillow_data = load_dataset(os.path.join(base_dir, 'Zillow_train.json'))  + \
                  load_dataset(os.path.join(base_dir, 'Zillow_dev.json')) + \
                  load_dataset(os.path.join(base_dir, 'Zillow_test.json'))


    qt30_data = load_dataset(os.path.join(base_dir, 'QT30_train.json')) + \
                load_dataset(os.path.join(base_dir, 'QT30_dev.json'))+ \
                load_dataset(os.path.join(base_dir, 'QT30_test.json'))

    us2016_data = load_dataset(os.path.join(base_dir, 'US2016_train.json')) + \
                  load_dataset(os.path.join(base_dir, 'US2016_dev.json')) + \
                  load_dataset(os.path.join(base_dir, 'US2016_test.json'))


    combined_data = zillow_data+qt30_data+ us2016_data
    labels_all = [item['Relation'] for item in combined_data]
    label_encoder.fit(labels_all)

    full_dataset = CustomDataset(combined_data, tokenizer, label_encoder)
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)

    model = SiameseRoBERTa("roberta-base", len(label_encoder.classes_))
    model = train_model(model, full_loader)

    torch.save(model.state_dict(), "siamese_roberta_model.pt")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("model and label encoder saved.")
    return model, tokenizer, label_encoder


if __name__ == "__main__":
    run_final_training()
