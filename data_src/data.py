import pandas as pd
import numpy as np
import os
import json


def AIF(path_to_files):

    combined_texts1 = []
    combined_texts2 = []
    combined_argument_types = []
    combined_fromIDs = []
    combined_toIDs = []
    combined_typeID = []


    for file_name in os.listdir(path_to_files):
        if file_name.endswith(".json"):
            json_file_path = os.path.join(path_to_files, file_name)
            node_ids_texts = {}
            node_ids_ids = {}

            ids = []
            texts = []

            with open(json_file_path) as jsonFilenodes:
                dtnodes = json.load(jsonFilenodes)
                if isinstance(dtnodes, dict):
                    dtnodes = [dtnodes]

                for dr_entry in dtnodes:
                    nodes_list = dr_entry["nodes"]
                    for nodes in nodes_list:
                        n_id, text, type = nodes['nodeID'], nodes['text'], nodes['type']
                        if type == 'I':
                            node_ids_texts[n_id] = text
                            node_ids_ids[n_id] = n_id
                            ids.append(n_id)
                            texts.append(text)

            texts1, texts2, argument_types, fromIDs, toIDs, typeID = [], [], [], [], [], []

            with open(json_file_path) as jsonFile:
                dt = json.load(jsonFile)
                if isinstance(dt, dict):
                    dt = [dt]

                for dr_entry in dt:
                    nodes = dr_entry["nodes"]
                    for idx, line in enumerate(nodes):
                        if idx > 0:
                            n_id, type = line['nodeID'], line['type']
                            text1, text2 = "", ""
                            fromid, toid = [], []
                            if type == 'RA' or type == "CA":
                                edges = dr_entry["edges"]
                                for edge_entry in edges:
                                    fromID, toID, edgeID = edge_entry['fromID'], edge_entry['toID'], edge_entry['edgeID']
                                    if n_id == toID and fromID in node_ids_texts:
                                        text1 = node_ids_texts[fromID]
                                        fromid = node_ids_ids[fromID]

                                    if n_id == fromID and toID in node_ids_texts:
                                        text2 = node_ids_texts[toID]
                                        toid = node_ids_ids[toID]
                                texts1.append(text1)
                                texts2.append(text2)
                                argument_types.append(type)
                                fromIDs.append(fromid)
                                toIDs.append(toid)
                                typeID.append(n_id)

            combined_texts1.extend(texts1)
            combined_texts2.extend(texts2)
            combined_argument_types.extend(argument_types)
            combined_fromIDs.extend(fromIDs)
            combined_toIDs.extend(toIDs)
            combined_typeID.extend(typeID)

    df = pd.DataFrame(data={
            'AC1': combined_texts1,
            'AC2': combined_texts2,
            'Relation': combined_argument_types,
            'AC1ID': combined_fromIDs,
            'AC2ID': combined_toIDs,
            'RelationID': combined_typeID,
        })
    df.replace('', np.nan, inplace=True)
    df.dropna(how='any', inplace=True)



    return df






def data_handling():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # df = AMPERE(os.path.join(base_dir, '../data/AMPERE++/ampere_train.jsonl'))
    # output_csv_file_path = os.path.join(base_dir, '../data_src/AMPERE++_train.csv')
    # df.to_csv(output_csv_file_path, index=False)

    def save_df_to_json(df, output_json_file_path, top_level_key):

        data_dict = df.to_dict(orient='records')

        final_dict = {top_level_key: data_dict}

        with open(output_json_file_path, 'w') as json_file:
            json.dump(final_dict, json_file, indent=4)


    df=AIF(os.path.join(base_dir,'../data/US2016/train'))
    output_json_file_path = os.path.join(base_dir,'../data_src/US2016_train.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/US2016/test'))
    output_json_file_path = os.path.join(base_dir,'../data_src/US2016_test.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/US2016/dev'))
    output_json_file_path = os.path.join(base_dir,'../data_src/US2016_dev.json')
    save_df_to_json(df, output_json_file_path, "argument_data")


    df = AIF(os.path.join(base_dir,'../data/QT30/train'))
    output_json_file_path = os.path.join(base_dir,'../data_src/QT30_train.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/QT30/test'))
    output_json_file_path = os.path.join(base_dir,'../data_src/QT30_test.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/QT30/dev'))
    output_json_file_path = os.path.join(base_dir,'../data_src/QT30_dev.json')
    save_df_to_json(df, output_json_file_path, "argument_data")


    df =AIF(os.path.join(base_dir,'../data/Zillow/train'))
    output_json_file_path = os.path.join(base_dir,'../data_src/Zillow_train.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/Zillow/test'))
    output_json_file_path = os.path.join(base_dir,'../data_src/Zillow_test.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    df = AIF(os.path.join(base_dir,'../data/Zillow/dev'))
    output_json_file_path = os.path.join(base_dir,'../data_src/Zillow_dev.json')
    save_df_to_json(df, output_json_file_path, "argument_data")

    return

if __name__ == "__main__":
    data_handling()


