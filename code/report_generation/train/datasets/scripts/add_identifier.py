import re
import json

def add_medical_material_tag(input_file_path: str, output_file_path: str) -> int:
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"错误：无法解析文件中的JSON内容 {input_file_path}")
        return 0

    modified_count = 0

    # 捕获 “may have” 后面到行尾的疾病列表
    pattern = re.compile(r"This patient may have the following diseases:([^\n]*)")

    for item in data:
        if isinstance(item, dict) and item.get("conversations"):
            conv = item["conversations"][0]
            if "value" in conv:
                value = conv["value"]

                match = pattern.search(value)
                if not match:
                    continue

                # 拆出疾病列表并标准化空格
                diseases = [d.strip() for d in match.group(1).split(",")]

                # 仅当包含 Medical material 且尚未插入标签时才处理
                if "Medical material" in diseases and "<medical_material>" not in value:
                    prefix = "<image3d>\n"
                    if prefix in value:
                        # **在 <image3d>\n 后插入标签（仅替换第一次出现）**
                        conv["value"] = value.replace(
                            prefix, prefix + "<medical_material> ", 1
                        )
                        modified_count += 1
                    else:
                        # 万一缺失前缀，保持之前的行为：直接在最前面加
                        conv["value"] = "<medical_material> " + value
                        modified_count += 1

    # 输出结果
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return modified_count


if __name__ == "__main__":
    input_filename = (
        "/data/jiawei_workspace/report_generate_docker/datasets/plus_report_vqa/train_vqa_modified.json"
    )
    output_filename = (
        "/data/jiawei_workspace/report_generate_docker/datasets/plus_identifier/only_input/train_vqa_modified.json"
    )

    count = add_medical_material_tag(input_filename, output_filename)

    print("处理完成！")
    print(f"共为 {count} 条记录插入 '<medical_material>' 标签。")
    print(f"结果已保存到: {output_filename}")
