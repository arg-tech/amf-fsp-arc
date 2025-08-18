from frame_semantic_transformer import FrameSemanticTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import requests
import tempfile
import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from copy import deepcopy

frame_transformer = FrameSemanticTransformer()
import nltk
nltk.data.path.append("/root/nltk_data")
model= "https://huggingface.co/Somaye/FSP-ARC/resolve/main/siamese_roberta_model.pt"
label_encoder = "https://huggingface.co/Somaye/FSP-ARC/resolve/main/label_encoder.pkl"

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


# ========= Dataset Class =========

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

        tokenized_input1 = self.tokenizer(Input1, padding='max_length', truncation=True, max_length=self.max_length,
                                          return_tensors='pt')
        tokenized_input2 = self.tokenizer(Input2, padding='max_length', truncation=True, max_length=self.max_length,
                                          return_tensors='pt')

        input_ids1 = tokenized_input1["input_ids"].squeeze()
        attention_mask1 = tokenized_input1["attention_mask"].squeeze()
        input_ids2 = tokenized_input2["input_ids"].squeeze()
        attention_mask2 = tokenized_input2["attention_mask"].squeeze()

        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.tensor(label_encoded)

        return input_ids1, attention_mask1, input_ids2, attention_mask2, label_tensor




def clean_output(original, arc_modified):
    cleaned = deepcopy(original)
    pred_map = {n['nodeID']: n['type'] for n in arc_modified['AIF']['nodes']}

    for node in cleaned['AIF']['nodes']:
        if node['nodeID'] in pred_map:
            predicted_type = pred_map[node['nodeID']]
            node['type'] = predicted_type

            # Update text based on predicted type
            if predicted_type == "RA":
                node['text'] = "Default Inference"
            elif predicted_type == "CA":
                node['text'] = "Default Conflict"


        # Clean auxiliary keys from I-nodes
        for key in ['frame_label', 'relevant_elements', 'relevant_trigger_word', 'prompt1', 'prompt2']:
            node.pop(key, None)

    return cleaned


def download_file_from_hf(url):
    response = requests.get(url)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(response.content)
    tmp.close()
    return tmp.name

def ARC(data, model=None, label_encoder=None):
    if 'AIF' not in data:
        return data

    model_path = download_file_from_hf(model) if model and model.startswith("http") else model
    label_encoder_path = download_file_from_hf(label_encoder) if label_encoder and label_encoder.startswith(
        "http") else label_encoder

    nodes = data['AIF']['nodes']
    edges = data['AIF']['edges']
    prompt1_map, prompt2_map = {}, {}

    # Collect prompt1 and prompt2 for each I-node
    for node in nodes:
        if node.get("type") == "I":
            node_id = node.get("nodeID")
            prompt1_raw = node.get("prompt1", [])
            prompt2_raw = node.get("prompt2", [])
            prompt1 = " ".join(prompt1_raw) if isinstance(prompt1_raw, list) else prompt1_raw or ""
            prompt2 = " ".join(prompt2_raw) if isinstance(prompt2_raw, list) else prompt2_raw or ""
            prompt1_map[node_id] = prompt1.strip()
            prompt2_map[node_id] = prompt2.strip()

    texts1, texts2, fromIDs, toIDs, types, relationIDs = [], [], [], [], [], []

    # Include all RA/CA edges, with fallback to raw AC text if prompts are missing
    i_node_map = {node["nodeID"]: node for node in nodes if node.get("type") == "I"}

    for node in nodes:
        if node.get("type") in ["RA", "CA"]:
            node_id = node["nodeID"]
            fromID = toID = None
            from_prompt1 = to_prompt1 = ""
            from_prompt2 = to_prompt2 = ""

            for edge in edges:
                if edge["toID"] == node_id and edge["fromID"] in i_node_map:
                    fromID = edge["fromID"]
                    from_prompt1 = prompt1_map.get(fromID, "") or i_node_map[fromID].get("text", "")
                    from_prompt2 = prompt2_map.get(fromID, "") or from_prompt1
                if edge["fromID"] == node_id and edge["toID"] in i_node_map:
                    toID = edge["toID"]
                    to_prompt1 = prompt1_map.get(toID, "") or i_node_map[toID].get("text", "")
                    to_prompt2 = prompt2_map.get(toID, "") or to_prompt1

            # Include if both IDs are found
            if fromID is not None and toID is not None:
                combined1 = f"{from_prompt1} [SEP] {to_prompt1}"
                combined2 = f"{from_prompt2} [SEP] {to_prompt2}"
                texts1.append(combined1)
                texts2.append(combined2)
                fromIDs.append(fromID)
                toIDs.append(toID)
                types.append(node["type"])
                relationIDs.append(node_id)

    df = pd.DataFrame({
        'prompt1': texts1,
        'prompt2': texts2,
        'Relation': types,
        'AC1ID': fromIDs,
        'AC2ID': toIDs,
        'RelationID': relationIDs
    })

    df.replace('', np.nan, inplace=True)
    df.dropna(how='any', inplace=True)

    # Prepare formatted inputs
    data_json = df.to_dict(orient='records')
    formatted_data = []
    for row in data_json:
        ac1_prompt1, ac2_prompt1 = row["prompt1"].split("[SEP]")
        ac1_prompt2, ac2_prompt2 = row["prompt2"].split("[SEP]")
        formatted_data.append({
            "AC1": ac1_prompt1.strip(),
            "AC2": ac2_prompt1.strip(),
            "Results_AC1": [{"prompt1": ac1_prompt1.strip(), "prompt2": ac1_prompt2.strip()}],
            "Results_AC2": [{"prompt1": ac2_prompt1.strip(), "prompt2": ac2_prompt2.strip()}],
            "Relation": row["Relation"],
            "RelationID": row["RelationID"],
            "AC1ID": row["AC1ID"],
            "AC2ID": row["AC2ID"]
        })

    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    model = SiameseRoBERTa("roberta-base", len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = CustomDataset(formatted_data, tokenizer, label_encoder)
    dataloader = DataLoader(dataset, batch_size=8)

    predictions = {}
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (input_ids1, attention_mask1, input_ids2, attention_mask2, labels) in enumerate(dataloader):
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)

            logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            preds = torch.argmax(logits, dim=1)
            pred_labels = label_encoder.inverse_transform(preds.cpu().numpy())

            batch_ids = [formatted_data[i * 8 + j]["RelationID"] for j in range(len(pred_labels)) if i * 8 + j < len(formatted_data)]
            for r_id, pred_label in zip(batch_ids, pred_labels):
                predictions[r_id] = pred_label

    # Update RA/CA node types in original data
    for node in nodes:
        if node.get("type") in ["RA", "CA"]:
            r_id = node["nodeID"]
            pred = predictions.get(r_id)
            if pred:
                node["type"] = pred

    # Final output: enriched argument data
    argument_data = []
    for entry in formatted_data:
        ac1_id = entry["AC1ID"]
        ac2_id = entry["AC2ID"]
        relation_id = entry["RelationID"]
        ac1_info = i_node_map.get(ac1_id, {})
        ac2_info = i_node_map.get(ac2_id, {})

        result = {
            "AC1": ac1_info.get("text", entry["AC1"]),
            "AC2": ac2_info.get("text", entry["AC2"]),
            "Relation": predictions.get(relation_id, entry["Relation"]),
            "AC1ID": ac1_id,
            "AC2ID": ac2_id,
            "RelationID": relation_id,
            "Results_AC1": [{
                "frame_label": ac1_info.get("frame_label"),
                "relevant_elements": ac1_info.get("relevant_elements", {}),
                "relevant_trigger_word": ac1_info.get("relevant_trigger_word"),
                "prompt1": ac1_info.get("prompt1", []),
                "prompt2": ac1_info.get("prompt2", [])
            }] if ac1_info.get("prompt1") else [],
            "Results_AC2": [{
                "frame_label": ac2_info.get("frame_label"),
                "relevant_elements": ac2_info.get("relevant_elements", {}),
                "relevant_trigger_word": ac2_info.get("relevant_trigger_word"),
                "prompt1": ac2_info.get("prompt1", []),
                "prompt2": ac2_info.get("prompt2", [])
            }] if ac2_info.get("prompt1") else []
        }

        argument_data.append(result)

    return data, argument_data





def fill_mask(input_text):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0,
                                                         forced_eos_token_id=2)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    batch = tokenizer(input_text, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=200)
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_output


def create_prompts(frame, trigger, elements, data):

    if 'AIF' not in data:
        return None

    aif_json = data['AIF']
    i_nodes = [node for node in aif_json['nodes'] if node.get('type') == 'I']


    for node in i_nodes:
        text = node.get('text', '')
        frame_elements = [value for key, value in elements.items() if value != "None"]
        list_of_words = [trigger] + frame_elements

        positions = {word.lower(): text.lower().find(word.lower()) for word in list_of_words}
        sorted_list1 = sorted(list_of_words, key=lambda word: positions[word.lower()])

        prompt1 = " <mask> ".join(sorted_list1)

        word_to_role = {value: key for key, value in elements.items() if value != "None"}
        word_to_role[trigger] = frame

        sorted_list2 = [word_to_role[word] for word in sorted_list1]
        prompt2 = " <mask> ".join(sorted_list2)

        return prompt1, prompt2


def frame_net(text):

    result = frame_transformer.detect_frames(text)

    if result.frames:
        frames_data = []
        encountered_frames = set()

        for frame in result.frames:
            if frame.name not in encountered_frames:
                trigger_location = frame.trigger_location
                trigger_word = text[trigger_location:].split()[0]
                frame_info = {
                    "frame": frame.name,
                    "trigger_word": trigger_word,
                    "elements": {}
                }
                for element in frame.frame_elements:
                    frame_info["elements"][element.name] = element.text
                frames_data.append(frame_info)
                encountered_frames.add(frame.name)

        return {"frames": frames_data}
    else:
        return None


def frame_setting(data):
    if 'AIF' not in data:
        return None

    aif_json = data['AIF']
    i_nodes = [node for node in aif_json['nodes'] if node.get('type') == 'I']
    all_frames = []

    for node in i_nodes:
        text = node.get('text', '')
        frames_data = frame_net(text)

        if frames_data:
            frames = frames_data.get("frames", [])

            if frames:
                frame_entry = {
                    "I-node-ID": node.get("nodeID"),
                    "I-node-text": text,
                    "frames": frames
                }
                all_frames.append(frame_entry)

    return {"frames": all_frames} if all_frames else None


def find_most_relevant_frame(all_frames, data):
    if 'AIF' not in data or not all_frames:
        return None

    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    results = []

    for entry in all_frames:
        i_text = entry['I-node-text']
        frame_candidates = entry['frames']
        sentence_embedding = sbert_model.encode(i_text, convert_to_tensor=True)
        frame_labels = [frame['frame'] for frame in frame_candidates]
        frame_embeddings = sbert_model.encode(frame_labels, convert_to_tensor=True)

        cosine_similarities = util.pytorch_cos_sim(sentence_embedding, frame_embeddings)
        max_index = torch.argmax(cosine_similarities).item()
        most_relevant = frame_candidates[max_index]

        results.append({
            "I-node-ID": entry["I-node-ID"],
            "I-node-text": i_text,
            "most_relevant_frame": most_relevant
        })

    return results






# Debugging

def is_valid_json(my_json):
    try:
        json.loads(my_json)
    except ValueError:
        return False
    return True

if __name__ == "__main__":
        true = True
        null = None
        false = False

        file={"AIF": {"nodes": [{"nodeID": 3, "text": "We should go eat", "type": "L"}, {"nodeID": 4, "text": "We should go eat", "type": "I"}, {"nodeID": 5, "text": "Default Illocuting", "type": "YA"}, {"nodeID": 6, "text": "|Wilma: Why", "type": "L"}, {"nodeID": 7, "text": "|Wilma: Why", "type": "I"}, {"nodeID": 8, "text": "Default Illocuting", "type": "YA"}, {"nodeID": 9, "text": "Bob: Because I'm hungry Wilma: Yeah me too", "type": "L"}, {"nodeID": 10, "text": "Bob: Because I'm hungry Wilma: Yeah me too", "type": "I"}, {"nodeID": 11, "text": "Default Illocuting", "type": "YA"}, {"nodeID": 12, "text": "Bob: So let's eat", "type": "L"}, {"nodeID": 13, "text": "Bob: So let's eat", "type": "I"}, {"nodeID": 14, "text": "Default Illocuting", "type": "YA"}, {"text": "Default Inference", "type": "RA", "nodeID": 15}, {"text": "Default Inference", "type": "RA", "nodeID": 16}, {"text": "Default Conflict", "type": "CA", "nodeID": 17}, {"text": "Default Inference", "type": "RA", "nodeID": 18}], "edges": [{"edgeID": 2, "fromID": 3, "toID": 5}, {"edgeID": 3, "fromID": 5, "toID": 4}, {"edgeID": 4, "fromID": 6, "toID": 8}, {"edgeID": 5, "fromID": 8, "toID": 7}, {"edgeID": 6, "fromID": 9, "toID": 11}, {"edgeID": 7, "fromID": 11, "toID": 10}, {"edgeID": 8, "fromID": 12, "toID": 14}, {"edgeID": 9, "fromID": 14, "toID": 13}, {"fromID": 4, "toID": 15, "edgeID": 10}, {"fromID": 15, "toID": 10, "edgeID": 11}, {"fromID": 4, "toID": 16, "edgeID": 12}, {"fromID": 16, "toID": 13, "edgeID": 13}, {"fromID": 7, "toID": 17, "edgeID": 14}, {"fromID": 17, "toID": 13, "edgeID": 15}, {"fromID": 10, "toID": 18, "edgeID": 16}, {"fromID": 18, "toID": 13, "edgeID": 17}], "locutions": [{"nodeID": 3, "personID": 0}, {"nodeID": 6, "personID": 0}, {"nodeID": 9, "personID": 0}, {"nodeID": 12, "personID": 0}], "schemefulfillments": null, "descriptorfulfillments": null, "participants": [{"firstname": "Bob", "participantID": 0, "surname": "None"}, {"firstname": "Bob", "participantID": 0, "surname": "None"}, {"firstname": "Bob", "participantID": 0, "surname": "None"}, {"firstname": "Bob", "participantID": 0, "surname": "None"}, {"firstname": "Bob", "participantID": 0, "surname": "None"}]}, "OVA": [], "dialog": true, "text": {"txt": " Bob None <span class=\"highlighted\" id=\"0\">We should go eat. |Wilma: Why? Bob: Because I'm hungry Wilma: Yeah me too. Bob: So let's eat.</span>.<br><br>"}}

        # file='../28038.json'
        if isinstance(file, (str, bytes, bytearray)):
            if os.path.isfile(file):
                file = open(file, 'r')
                data = json.load(file)
                # You can still apply same filtering here

                frame_result = frame_setting(data)
                if not frame_result or "frames" not in frame_result:
                    print("No frames detected")
                    exit()

                frames = frame_result["frames"]
                most_relevant_frame = find_most_relevant_frame(frames, data)

                # 2. Inject prompts into I-nodes
                frame_index = 0
                original_nodes = data["AIF"]["nodes"]
                for node in original_nodes:
                    if node.get("type") == "I":
                        if frame_index < len(most_relevant_frame):
                            item = most_relevant_frame[frame_index]
                            frame_info = item.get("most_relevant_frame")
                            if frame_info:
                                frame_label = frame_info["frame"]
                                relevant_elements = frame_info["elements"]
                                relevant_trigger_word = frame_info["trigger_word"]

                                prompt1, prompt2 = create_prompts(frame_label, relevant_trigger_word, relevant_elements,
                                                                  data)
                                mask_output1 = fill_mask(prompt1) if prompt1 else []
                                mask_output2 = fill_mask(prompt2) if prompt2 else []

                                node["frame_label"] = frame_label
                                node["relevant_elements"] = relevant_elements
                                node["relevant_trigger_word"] = relevant_trigger_word
                                node["prompt1"] = mask_output1
                                node["prompt2"] = mask_output2
                        else:
                            node["frame_label"] = None
                            node["relevant_elements"] = {}
                            node["relevant_trigger_word"] = None
                            node["prompt1"] = []
                            node["prompt2"] = []
                        frame_index += 1

                # 3. Run ARC to predict relation types
                ARC_result, argument_data = ARC(data, model=model, label_encoder=label_encoder)

                # 4. Clean and print result
                final_result = clean_output(data, ARC_result)
                final_result["FSP"] = argument_data
                print(json.dumps(final_result, ensure_ascii=False))
                # print(final_result)
                # print({"FSP": argument_data})


            else:
                print("invalid file")
        elif is_valid_json(json.dumps(file)):

            frame_result = frame_setting(file)
            if not frame_result or "frames" not in frame_result:
                print("No frames detected")
                exit()

            frames = frame_result["frames"]
            most_relevant_frame = find_most_relevant_frame(frames, file)

            # 2. Inject prompts into I-nodes
            frame_index = 0
            original_nodes = file["AIF"]["nodes"]
            for node in original_nodes:
                if node.get("type") == "I":
                    if frame_index < len(most_relevant_frame):
                        item = most_relevant_frame[frame_index]
                        frame_info = item.get("most_relevant_frame")
                        if frame_info:
                            frame_label = frame_info["frame"]
                            relevant_elements = frame_info["elements"]
                            relevant_trigger_word = frame_info["trigger_word"]

                            prompt1, prompt2 = create_prompts(frame_label, relevant_trigger_word, relevant_elements,
                                                              file)
                            mask_output1 = fill_mask(prompt1) if prompt1 else []
                            mask_output2 = fill_mask(prompt2) if prompt2 else []

                            node["frame_label"] = frame_label
                            node["relevant_elements"] = relevant_elements
                            node["relevant_trigger_word"] = relevant_trigger_word
                            node["prompt1"] = mask_output1
                            node["prompt2"] = mask_output2
                    else:
                        node["frame_label"] = None
                        node["relevant_elements"] = {}
                        node["relevant_trigger_word"] = None
                        node["prompt1"] = []
                        node["prompt2"] = []
                    frame_index += 1

            # 3. Run ARC to predict relation types
            ARC_result, argument_data = ARC(file, model=model, label_encoder=label_encoder)

            # 4. Clean and print result
            final_result = clean_output(file, ARC_result)

            final_result["FSP"] = argument_data
            print(json.dumps(final_result, ensure_ascii=False))

            # print(final_result)
            # print({"FSP": argument_data})

        else:
            print("invalid file")





