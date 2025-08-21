from flask import Flask, request, render_template, jsonify
import json
import os
import logging
from app.frames import frame_setting
from app.frames import find_most_relevant_frame
from app.frames import create_prompts
from app.frames import fill_mask
from app.frames import ARC
from copy import deepcopy
from flask_compress import Compress


app = Flask(__name__)
Compress(app)

model= "https://huggingface.co/Somaye/FSP-ARC/resolve/main/siamese_roberta_model.pt"
label_encoder = "https://huggingface.co/Somaye/FSP-ARC/resolve/main/label_encoder.pkl"

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


def remove_duplicate_acids(fsp_list):
    seen_acids = set()
    unique_fsp = []

    for pair in fsp_list:
        filtered_pair = []
        for entry in pair:
            ac_id = entry.get("ACID")
            if ac_id not in seen_acids:
                seen_acids.add(ac_id)
                filtered_pair.append(entry)
        if filtered_pair:
            unique_fsp.append(filtered_pair)
    return unique_fsp


def is_valid_json(my_json):
    try:
        json.loads(my_json)
    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        return False
    return True

@app.route('/', methods=['GET', 'POST'])
def get_data_file():
    if request.method == 'POST':
        true = True
        null = None
        false = False
        logging.debug("Received POST request")

        if request.files:
            f = request.files['file']

            f.save(f.filename)
            logging.debug(f"Saved file 1 as: {f.filename}")
            with open(f.filename, 'r') as ff:
                data = json.load(ff)

            frame = frame_setting(data)

            if frame:
                    frames = frame.get("frames", [])

                    original_nodes = data["AIF"]["nodes"]
                    i_nodes = [node for node in original_nodes if node.get("type") == "I"]

                    # Step 1: Run frame setting
                    frame_data_result = frame_setting(data)
                    if not frame_data_result or "frames" not in frame_data_result:
                        return jsonify(data)

                    frames = frame_data_result["frames"]
                    most_relevant_frame = find_most_relevant_frame(frames, data)

                    frame_index = 0

                    for node in original_nodes:
                        if node.get("type") == "I":
                            if frame_index < len(most_relevant_frame):
                                item = most_relevant_frame[frame_index]
                                frame_info = item.get("most_relevant_frame")

                                if frame_info:
                                    frame_label = frame_info["frame"]
                                    relevant_elements = frame_info["elements"]
                                    relevant_trigger_word = frame_info["trigger_word"]

                                    prompt1, prompt2 = create_prompts(frame_label, relevant_trigger_word,
                                                                      relevant_elements, data)
                                    mask_output1 = fill_mask(prompt1) if prompt1 else []
                                    mask_output2 = fill_mask(prompt2) if prompt2 else []

                                    # Inject directly into the I node
                                    node["frame_label"] = frame_label
                                    node["relevant_elements"] = relevant_elements
                                    node["relevant_trigger_word"] = relevant_trigger_word
                                    node["prompt1"] = mask_output1
                                    node["prompt2"] = mask_output2

                            else:
                                # No frame found for this I-node; insert empty data
                                node["frame_label"] = None
                                node["relevant_elements"] = {}
                                node["relevant_trigger_word"] = None
                                node["prompt1"] = []
                                node["prompt2"] = []
                            frame_index += 1


                    os.remove(f.filename)

                    # ARC_result,argument_data = ARC(data, model_path="siamese_roberta_model.pt", label_encoder_path="label_encoder.pkl")
                    ARC_result, argument_data = ARC(data, model=model, label_encoder=label_encoder)
                    ARC_result = clean_output(data, ARC_result)
                    argument_data = remove_duplicate_acids(argument_data)

                    ARC_result["FSP"] = argument_data
                    return jsonify(ARC_result)

                    # return jsonify({
                    #     "ARC_result": ARC_result,
                    #     "FSP": argument_data
                    # })

        if request.values:

            data = request.values.get('json')



            # print("Received data1:", data1)
            #
            # print("Received data2:", data2)

            if is_valid_json(json.dumps(data)) :

                # print("Both data1 and data2 are valid JSON")

                data = json.loads(data)



                # print("Parsed data1:", data1)
                #
                # print("Parsed data2:", data2)

                frame = frame_setting(data)

                if frame:
                    frames = frame.get("frames", [])

                    original_nodes = data["AIF"]["nodes"]

                    i_nodes = [node for node in original_nodes if node.get("type") == "I"]

                    # Step 1: Run frame setting
                    frame_data_result = frame_setting(data)
                    if not frame_data_result or "frames" not in frame_data_result:
                        return jsonify(data)

                    frames = frame_data_result["frames"]
                    most_relevant_frame = find_most_relevant_frame(frames, data)

                    frame_index = 0

                    for node in original_nodes:
                        if node.get("type") == "I":
                            if frame_index < len(most_relevant_frame):
                                item = most_relevant_frame[frame_index]
                                frame_info = item.get("most_relevant_frame")

                                if frame_info:
                                    frame_label = frame_info["frame"]
                                    relevant_elements = frame_info["elements"]
                                    relevant_trigger_word = frame_info["trigger_word"]

                                    prompt1, prompt2 = create_prompts(frame_label, relevant_trigger_word,
                                                                      relevant_elements, data)
                                    mask_output1 = fill_mask(prompt1) if prompt1 else []
                                    mask_output2 = fill_mask(prompt2) if prompt2 else []

                                    # Inject directly into the I node
                                    node["frame_label"] = frame_label
                                    node["relevant_elements"] = relevant_elements
                                    node["relevant_trigger_word"] = relevant_trigger_word
                                    node["prompt1"] = mask_output1
                                    node["prompt2"] = mask_output2

                            else:
                                # No frame found for this I-node; insert empty data
                                node["frame_label"] = None
                                node["relevant_elements"] = {}
                                node["relevant_trigger_word"] = None
                                node["prompt1"] = []
                                node["prompt2"] = []
                            frame_index += 1

                    # ARC_result,argument_data = ARC(data, model_path="siamese_roberta_model.pt", label_encoder_path="label_encoder.pkl")
                    ARC_result, argument_data = ARC(data, model=model, label_encoder=label_encoder)

                    ARC_result = clean_output(data, ARC_result)
                    argument_data = remove_duplicate_acids(argument_data)

                    ARC_result["FSP"] = argument_data
                    return jsonify(ARC_result)

                    # return jsonify({
                    #     "ARC_result": ARC_result,
                    #     "FSP": argument_data
                    # })





            else:
                # logging.error("Invalid JSON data provided")
                return jsonify({"error": "Invalid JSON data"}), 400
        else:
            # logging.error("No files or JSON data provided")
            return jsonify({"error": "No files or JSON data provided"}), 400

    elif request.method == 'GET':
        # logging.debug("Received GET request")
        return render_template('docs.html')

if __name__ == '__main__':
    app.run(debug=True)
