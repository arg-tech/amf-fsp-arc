# import os
# import json
# import pickle
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import RobertaTokenizer, RobertaModel
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#
# # =================== Model and Dataset ===================
#
# class CrossAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super(CrossAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
#
#     def forward(self, query, key, value, attn_mask=None):
#         attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
#         return attn_output
#
# class SiameseRoBERTa(nn.Module):
#     def __init__(self, model_name, num_classes, num_heads=8):
#         super(SiameseRoBERTa, self).__init__()
#         self.roberta = RobertaModel.from_pretrained(model_name)
#         self.hidden_size = self.roberta.config.hidden_size
#         self.cross_attention = CrossAttention(self.hidden_size, num_heads)
#         self.classifier = nn.Linear(3 * self.hidden_size, num_classes)
#
#     def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
#         output1 = self.roberta(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
#         output2 = self.roberta(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state
#         output1 = self.cross_attention(output1, output2, output2)
#         output2 = self.cross_attention(output2, output1, output1)
#         u = torch.mean(output1, dim=1)
#         v = torch.mean(output2, dim=1)
#         combined = torch.cat((u, v, torch.abs(u - v)), dim=1)
#         logits = self.classifier(combined)
#         return logits
#
# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, label_encoder, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.label_encoder = label_encoder
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         Results_AC1 = item.get('Results_AC1', [])
#         Results_AC2 = item.get('Results_AC2', [])
#         prompt_1_ac1 = Results_AC1[0].get('prompt1', "") if Results_AC1 and Results_AC1[0].get('prompt1', "") else ""
#         prompt_2_ac1 = Results_AC1[0].get('prompt2', "") if Results_AC1 and Results_AC1[0].get('prompt2', "") else ""
#         prompt_1_ac2 = Results_AC2[0].get('prompt1', "") if Results_AC2 and Results_AC2[0].get('prompt1', "") else ""
#         prompt_2_ac2 = Results_AC2[0].get('prompt2', "") if Results_AC2 and Results_AC2[0].get('prompt2', "") else ""
#         Input1 = f"{prompt_1_ac1} [SEP] {prompt_1_ac2}"
#         Input2 = f"{prompt_2_ac1} [SEP] {prompt_2_ac2}"
#         label = item['Relation']
#         tokenized_input1 = self.tokenizer(Input1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
#         tokenized_input2 = self.tokenizer(Input2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
#         input_ids1 = tokenized_input1["input_ids"].squeeze()
#         attention_mask1 = tokenized_input1["attention_mask"].squeeze()
#         input_ids2 = tokenized_input2["input_ids"].squeeze()
#         attention_mask2 = tokenized_input2["attention_mask"].squeeze()
#         label_encoded = self.label_encoder.transform([label])[0]
#         label_tensor = torch.tensor(label_encoded)
#         return input_ids1, attention_mask1, input_ids2, attention_mask2, label_tensor
#
# # =================== Utility Functions ===================
#
# def load_model_and_tokenizer(model_path, label_encoder_path):
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     with open(label_encoder_path, "rb") as f:
#         label_encoder = pickle.load(f)
#     model = SiameseRoBERTa("roberta-base", len(label_encoder.classes_))
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model, tokenizer, label_encoder
#
# def evaluate(model, data_loader, label_encoder):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     all_preds, all_labels = [], []
#
#     with torch.no_grad():
#         for input_ids1, attention_mask1, input_ids2, attention_mask2, labels in data_loader:
#             input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)
#             input_ids2, attention_mask2 = input_ids2.to(device), attention_mask2.to(device)
#             labels = labels.to(device)
#
#             logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
#             preds = torch.argmax(logits, dim=1)
#
#             pred_labels = label_encoder.inverse_transform(preds.cpu().numpy())
#             true_labels = label_encoder.inverse_transform(labels.cpu().numpy())
#
#             for pred, true in zip(pred_labels, true_labels):
#                 print(f"Predicted: {pred: <15}  |  Actual: {true}")
#
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
#     return accuracy, f1
#
# # =================== Main Script ===================
#
# if __name__ == "__main__":
#     test_path = "./Zillow_test.json"
#     model_path = "siamese_roberta_model.pt"
#     label_encoder_path = "label_encoder.pkl"
#
#     with open(test_path, 'r') as f:
#         test_data = json.load(f)['argument_data']
#
#     model, tokenizer, label_encoder = load_model_and_tokenizer(model_path, label_encoder_path)
#     test_dataset = CustomDataset(test_data, tokenizer, label_encoder)
#     test_loader = DataLoader(test_dataset, batch_size=16)
#
#     accuracy, f1 = evaluate(model, test_loader, label_encoder)
#     print(f"\n✅ Evaluation on Zillow Test Set:\nAccuracy = {accuracy:.4f}, Macro F1 = {f1:.4f}")
#
#
# import requests
# import json
# from google.cloud import storage
# from time import sleep
#
# # Set up GCS bucket and path
# BUCKET_NAME = "basalam"
# DEST_BLOB_PATH = "test/products_30000_50000.json"
#
# # Initialize GCS client
# client = storage.Client()
# bucket = client.bucket(BUCKET_NAME)
#
# # Set API headers
# headers = {
#     "Accept": "application/json",
#     "prefer": "",
#     "User-Agent": "curl/7.68.0",
#     "Connection": "close"
# }
#
# # Dictionary to hold all products
# all_products = {}
#
# # Loop from 30000 to 50000
# for product_id in range(50000, 100000):
#     url = f"https://core.basalam.com/v4/products/{product_id}"
#     try:
#         response = requests.get(url, headers=headers, timeout=5)
#         if response.status_code == 200:
#             all_products[str(product_id)] = response.json()
#             print(f"✅ {product_id}")
#         elif response.status_code == 404:
#             print(f"❌ {product_id}: Not Found")
#         else:
#             print(f"⚠️ {product_id}: Status {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"⚠️ {product_id}: Error {e}")
#         sleep(1)  # avoid hammering the server
#
# # Save locally
# with open("products_50000_100000.json", "w", encoding="utf-8") as f:
#     json.dump(all_products, f, ensure_ascii=False, indent=2)
#
# # Upload to GCS
# blob = bucket.blob(DEST_BLOB_PATH)
# blob.upload_from_filename("products_50000_100000.json", content_type="application/json")
#
# print("\n🎉 All products saved and uploaded to GCS: basalam/test/products_50000_100000.json")
#
#
#
#
#




