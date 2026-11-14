import json
import re
import sys
from copy import deepcopy

APPEND_TEXT = (
    "Medical material may appear as  follows: "
    "port catheter, port chamber, central venous catheter, Endotracheal tube, Nasogastric tube, pacemaker"
)

# 仅截取 "This patient may have ..." 到 "This patient does not have ..."（或文本末尾）
MAY_HAVE_RE = re.compile(
    r"this patient may have(?: the following diseases)?\s*:\s*(.*?)"
    r"(?:\n\s*this patient does not have|$)",
    flags=re.IGNORECASE | re.DOTALL
)

def may_have_contains_medical_material(text: str) -> bool:
    """检查 'may have' 段是否包含 Medical material（大小写不敏感）"""
    if not text:
        return False
    m = MAY_HAVE_RE.search(text)
    if not m:
        return False
    may_have_segment = m.group(1)
    return re.search(r"\bmedical material\b", may_have_segment, flags=re.IGNORECASE) is not None

def already_appended(text: str) -> bool:
    """避免重复追加"""
    return "Medical material may appear as the following keywords:" in (text or "")

def process_full_keep_all(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = deepcopy(data)
    items_hit = 0
    convs_modified = 0

    for item in out_data:
        hit_item = False
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                val = conv.get("value", "")
                if may_have_contains_medical_material(val):
                    hit_item = True
                    if not already_appended(val):
                        # 若原文本末尾没有换行，加一个换行再拼接 APPEND_TEXT
                        sep = "" if val.endswith(("\n", "\r")) else "\n"
                        conv["value"] = val + sep + APPEND_TEXT
                        convs_modified += 1
        if hit_item:
            items_hit += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Items hit: {items_hit}")
    print(f"Conversations modified: {convs_modified}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_json_path> <output_json_path>")
        sys.exit(1)
    process_full_keep_all(sys.argv[1], sys.argv[2])
