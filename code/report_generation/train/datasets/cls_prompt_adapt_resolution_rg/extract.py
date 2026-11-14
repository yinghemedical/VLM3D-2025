import json
import re
import sys
from copy import deepcopy

APPEND_TEXT = (
    "Medical material may appear as the following keywords: "
    "port catheter, port chamber, central venous catheter, Endotracheal tube, Nasogastric tube, pacemaker"
)

# 匹配 "This patient may have ..." 段，仅截取到 "This patient does not have ..." 或文本末尾
MAY_HAVE_RE = re.compile(
    r"this patient may have(?: the following diseases)?\s*:\s*(.*?)"
    r"(?:\n\s*this patient does not have|$)",
    flags=re.IGNORECASE | re.DOTALL
)

def may_have_contains_medical_material(text: str) -> bool:
    """仅检查 'may have' 段里是否包含 Medical material（大小写不敏感）"""
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

def process(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_items = []
    matched_count = 0

    for item in data:
        convs = item.get("conversations", [])
        # 深拷贝，避免原数据被修改
        new_item = deepcopy(item)

        hit_this_item = False
        for conv in new_item.get("conversations", []):
            if conv.get("from") == "human":
                val = conv.get("value", "")
                if may_have_contains_medical_material(val):
                    hit_this_item = True
                    if not already_appended(val):
                        conv["value"] = (val.rstrip() + "\n" + APPEND_TEXT)

        if hit_this_item:
            out_items.append(new_item)
            matched_count += 1

    # 保存新的 JSON，仅包含命中的条目（且 human value 已追加提示）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(matched_count)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_json_path> <output_json_path>")
        sys.exit(1)

    process(sys.argv[1], sys.argv[2])
