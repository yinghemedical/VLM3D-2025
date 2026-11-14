import json
import os
from tqdm import tqdm

error_list = [
    "train_1268/train_1268b/train_1268_b_2.npy"
]

image_dir = {
    "train": "/path/to/jiawei_database/ctrate_raw/train",
    "valid": "/path/to/jiawei_database/ctrate_raw/valid",
}

mode_list = ["train", "valid"]
outputs_path = "/data/jiawei_workspace/final_data/final_finetune_vqa.json"
inputs_path_list = [
    "/data/jiawei_workspace/report_generate_docker/datasets/cls_prompt_adapt_resolution_rg/train_cls_head_modified_new.json",
    "/data/jiawei_workspace/train_vqa.json",
]
inputs2_path_list = [
    "/data/jiawei_database/add_cls_vqa/cls_prompt_adapt_rg/valid_cls_head_modified_new.json",
    "/data/jiawei_workspace/valid_choice.json",
    "/data/jiawei_workspace/valid_long.json",
    "/data/jiawei_workspace/valid_rg.json",
    "/data/jiawei_workspace/valid_short.json",
]

inputs_list = {"train": [], "valid": []}
# with open(inputs_path, "r") as f:
#     inputs_list["train"] = json.load(f)
for inputs_path in inputs_path_list:
    with open(inputs_path, "r") as f:
        data = json.load(f)
        inputs_list["train"].extend(data)

for inputs2_path in inputs2_path_list:
    with open(inputs2_path, "r") as f:
        data = json.load(f)
        inputs_list["valid"].extend(data)

outputs_list = []
for mode in mode_list:
    print("Number of inputs in", mode, ":", len(inputs_list[mode]))
    for inputs in tqdm(inputs_list[mode]):
        if "answer" in inputs:
            inputs["conversations"].append({
                "from": "gpt",
                "value": inputs["answer"]
            })
        if "image3d" not in inputs:
            outputs_list.append(inputs)
            continue

        patient_id = inputs["image3d"][0].split("/")[-1].split("_")[1]
        scan_id = inputs["image3d"][0].split("/")[-1].split("_")[2]

        file_path = os.path.join(
            f"{mode}_{patient_id}",
            f"{mode}_{patient_id}{scan_id}",
            inputs["image3d"][0].split("/")[-1],
        )
        if os.path.exists(os.path.join(image_dir[mode], file_path)) and file_path not in error_list:
            inputs["image3d"] = [os.path.join(image_dir[mode], file_path)]
            outputs_list.append(inputs)
    print("Number of outputs in", mode, ":", len(outputs_list))

with open(outputs_path, "w") as f:
    json.dump(outputs_list, f, indent=4)
