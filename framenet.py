# !pip install frame-semantic-transformer

from frame_semantic_transformer import FrameSemanticTransformer
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import json
import os
from sentence_transformers import SentenceTransformer, util


def fill_mask(input_text):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0,
                                                         forced_eos_token_id=2)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    batch = tokenizer(input_text, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=200)
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_output


def create_prompts(frame, trigger, elements, text):
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
    frame_transformer = FrameSemanticTransformer()
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


def frame_setting(text):
    frames_data = frame_net(text)

    if frames_data:

        frames = frames_data.get("frames", [])

        if frames:

            labels = []
            elements_list = []
            triggers = []

            for frame_info in frames:
                label = frame_info['frame']
                elements = frame_info['elements']
                trigger = frame_info['trigger_word']

                labels.append(label)
                elements_list.append(elements)
                triggers.append(trigger)

            return {"frames": frames}
        else:
            return None
    else:
        return None



def find_most_relevant_frame(frames, ac_text):

    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    if not frames:
        return None


    sentence_embedding = sbert_model.encode(ac_text, convert_to_tensor=True)
    frame_labels = [frame['frame'] for frame in frames]
    frame_embeddings = sbert_model.encode(frame_labels, convert_to_tensor=True)


    cosine_similarities = util.pytorch_cos_sim(sentence_embedding, frame_embeddings)


    max_index = torch.argmax(cosine_similarities).item()
    most_relevant_frame = frames[max_index]

    return most_relevant_frame


def data_handling(text):
    frames_data = frame_setting(text)

    if frames_data:
        frames = frames_data.get("frames", [])


        most_relevant_frame = find_most_relevant_frame(frames, text)

        if most_relevant_frame:
            frame_label = most_relevant_frame['frame']
            relevant_elements = most_relevant_frame['elements']
            relevant_trigger_word = most_relevant_frame['trigger_word']

            prompt1, prompt2 = create_prompts(frame_label, relevant_trigger_word, relevant_elements, text)

            mask_output1 = fill_mask(prompt1)
            mask_output2 = fill_mask(prompt2)
            mask_output1 = mask_output1[0] if mask_output1 else text
            mask_output2 = mask_output2[0] if mask_output2 else text

            return [{
                'frame_label': frame_label,
                'relevant_elements': relevant_elements,
                'relevant_trigger_word': relevant_trigger_word,
                'prompt1': mask_output1,
                'prompt2': mask_output2,
            }]

    return None

def process_row(row):
    ac1, ac2 = row['AC1'], row['AC2']

    results_ac1 = data_handling(ac1)
    results_ac2 = data_handling(ac2)


    flattened_results_ac1 = []
    flattened_results_ac2 = []

    if results_ac1:
        for result in results_ac1:
            flattened_results_ac1.append({
                'frame_label': result['frame_label'],
                'relevant_elements': result['relevant_elements'],
                'relevant_trigger_word': result['relevant_trigger_word'],
                'prompt1': result['prompt1'],
                'prompt2': result['prompt2']

            })


    if results_ac2:
        for result in results_ac2:
            flattened_results_ac2.append({
                'frame_label': result['frame_label'],
                'relevant_elements': result['relevant_elements'],
                'relevant_trigger_word': result['relevant_trigger_word'],
                'prompt1': result['prompt1'],
                'prompt2': result['prompt2']

            })

    return flattened_results_ac1, flattened_results_ac2

def data_calling():

    base_dir = './data_src'

    for filename in os.listdir(base_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(base_dir, filename)

            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                argument_data = data.get('argument_data', [])
                for row in argument_data:
                    results_ac1, results_ac2 = process_row(row)

                    row['Results_AC1'] = results_ac1
                    row['Results_AC2'] = results_ac2

            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

    return



# text1 = "advertising always makes the products so tempting, regardless of whether they might bring negative effects on the human body."
# text2 = "advertising is not entirely decisive on personal eating habit."

# text2="Advertising makes products tempting, even if they have negative effects on the human body."
#
# text2="no"
#
# print(data_handling(text2))
#
# print(fill_mask("advertising always <mask> makes <mask> the products <mask> so tempting"))

