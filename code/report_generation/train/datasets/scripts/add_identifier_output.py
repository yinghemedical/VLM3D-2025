#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给符合条件的条目插入 <medical_material> 标签：
  · conversations 中 human 消息的 <image3d>\n 之后
  · conversations 中 gpt 消息（全部）开头
  · item['answer'] 开头（若存在）
"""

import json
import re
import sys
from pathlib import Path

# 默认文件，可用命令行参数覆盖
DEFAULT_IN = "/data/jiawei_workspace/report_generate_docker/datasets/plus_report_vqa/train_vqa_modified.json"
DEFAULT_OUT = "/data/jiawei_workspace/report_generate_docker/datasets/plus_identifier/input_and_output/train_vqa_modified.json"


def insert_mm_to_human(value: str) -> str:
    """在 <image3d>\n 后插入标签，若无该前缀则放最前面。"""
    if "<medical_material>" in value:
        return value
    img_prefix = "<image3d>\n"
    return (
        value.replace(img_prefix, img_prefix + "<medical_material> ", 1)
        if img_prefix in value
        else "<medical_material> " + value
    )


def add_medical_material_tag(data: list) -> tuple[int, int, int]:
    """
    修改列表中的 JSON 条目。

    返回三元组：(human_count, gpt_count, answer_count)
    """
    # 匹配 may-have 列表
    pat = re.compile(r"This patient may have the following diseases:([^\n]*)")

    h_cnt = g_cnt = a_cnt = 0

    for item in data:
        convs = item.get("conversations", [])
        if not convs:
            continue

        # ── 找到 human 消息（假设第一条就是；否则按 from == human）
        human_msg = (
            convs[0]
            if convs[0].get("from") == "human"
            else next((c for c in convs if c.get("from") == "human"), None)
        )
        if human_msg is None:
            continue

        # 判断是否包含 Medical material
        m = pat.search(human_msg.get("value", ""))
        if not m:
            continue
        diseases = [d.strip() for d in m.group(1).split(",")]
        if "Medical material" not in diseases:
            continue

        # ── 1) 处理 human value
        new_val = insert_mm_to_human(human_msg["value"])
        if new_val != human_msg["value"]:
            human_msg["value"] = new_val
            h_cnt += 1

        # ── 2) 处理 gpt 消息
        for c in convs:
            if c.get("from") == "gpt" and "<medical_material>" not in c.get(
                "value", ""
            ):
                c["value"] = "<medical_material> " + c["value"]
                g_cnt += 1

        # ── 3) 处理 answer 字段
        if "answer" in item and "<medical_material>" not in item["answer"]:
            item["answer"] = "<medical_material> " + item["answer"]
            a_cnt += 1

    return h_cnt, g_cnt, a_cnt


def main(in_path: str, out_path: str) -> None:
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ 找不到输入文件: {in_path}")
    except json.JSONDecodeError:
        sys.exit(f"❌ 无法解析 JSON: {in_path}")

    h, g, a = add_medical_material_tag(data)

    Path(out_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✅ 完成！")
    print(f"  · human  已插入: {h}")
    print(f"  · gpt    已插入: {g}")
    print(f"  · answer 已插入: {a}")
    print(f"结果保存至: {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    IN_FILE = argv[0] if len(argv) >= 1 else DEFAULT_IN
    OUT_FILE = argv[1] if len(argv) >= 2 else DEFAULT_OUT
    main(IN_FILE, OUT_FILE)
