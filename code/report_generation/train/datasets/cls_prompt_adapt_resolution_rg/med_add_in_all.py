import json
import sys
from copy import deepcopy

APPEND_TEXT = (
    "Medical material may appear as follows: "
    "enous ports/port chambers (port catheter/Port-a-Cath/Mediport), central venous catheters including PICC/Hickman/Permcath (often terminating in the SVC, RA, or SVC-RA junction), gastrointestinal tubes (nasogastric/orogastric/feeding tubes, e.g., Dobhoff), airway devices (tracheostomy, endotracheal tube, tracheal/bronchial stents), cardiac implants (pacemaker/ICD/CRT with leads—commonly to the RV apex), prosthetic heart valves (AVR/TAVR, MVR), vascular and coronary stents, pleural drainage devices (chest tube, pleural catheter/PleurX), and postoperative metallic materials (sternal wires, surgical clips) or vascular grafts."
)

def already_appended(text: str) -> bool:
    """避免重复追加"""
    return "Medical material may appear as the following keywords:" in (text or "")

def process_all_add(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = deepcopy(data)
    items_touched = 0
    convs_modified = 0

    for item in out_data:
        touched_this_item = False
        for conv in item.get("conversations", []):
            if str(conv.get("from", "")).lower() == "human":
                val = conv.get("value", "")
                if not already_appended(val):
                    sep = "" if val.endswith(("\n", "\r")) else "\n"
                    conv["value"] = val + sep + APPEND_TEXT
                    convs_modified += 1
                    touched_this_item = True
        if touched_this_item:
            items_touched += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Items touched: {items_touched}")
    print(f"Human conversations modified: {convs_modified}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_json_path> <output_json_path>")
        sys.exit(1)
    process_all_add(sys.argv[1], sys.argv[2])
