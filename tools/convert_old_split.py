"""Convert old YOLO-format leaky dataset to ImageFolder for DJI 5 classes."""
import os, shutil, re

src = r"E:\Drone_dataset\RFUAV\pics_exp1_alldrones\1.classification_dataset"
dst = r"E:\Drone_dataset\RFUAV-v2\split_old_dji5"

# DJI 5 class name mapping (old class names -> new taxonomy_ids)
# Old names from the YOLO dataset. We need to figure out which old names map to our 5 classes.
# Let's first scan what classes exist
from collections import Counter
cls_counter = Counter()
for split_name in ["train", "valid"]:
    sp = os.path.join(src, split_name)
    if not os.path.isdir(sp):
        continue
    img_dir = os.path.join(sp, "images")
    if not os.path.isdir(img_dir):
        # maybe flat: train/class/img.jpg
        for entry in os.listdir(sp):
            ep = os.path.join(sp, entry)
            if os.path.isdir(ep):
                for f in os.listdir(ep):
                    if f.lower().endswith(".jpg"):
                        cls_counter[entry] += 1
        continue
    # YOLO format: train/images/*.jpg, train/labels/*.txt
    for f in os.listdir(img_dir):
        if not f.lower().endswith(".jpg"):
            continue
        # Extract class from filename: "DJI AVATA2 (frame).jpg" -> "DJI AVATA2"
        # or "AVATA2_pack1 (frame).jpg" -> "AVATA2"
        base = os.path.splitext(f)[0]
        # try to get class from filename
        parts = base.split("_")
        if len(parts) >= 1:
            cls_counter[parts[0]] += 1

print("Classes found in old dataset:")
for cls, cnt in cls_counter.most_common(20):
    print(f"  {cls}: {cnt}")

# DJI 5 class mapping (old names from YOLO dataset)
dji5_map = {
    "DJI AVATA2": "uav.dji.avata2",
    "AVATA2": "uav.dji.avata2",
    "DJIAVATA2": "uav.dji.avata2",
    "DJFPVCOMBO": "uav.dji.fpv_combo",
    "DJI FPV": "uav.dji.fpv_combo",
    "DJMAVIC3PRO": "uav.dji.mavic3_pro",
    "DJI MAVIC3": "uav.dji.mavic3_pro",
    "DJMINI3": "uav.dji.mini3",
    "DJI MINI3": "uav.dji.mini3",
    "DJMINI4PRO": "uav.dji.mini4_pro",
    "DJI MINI4": "uav.dji.mini4_pro",
}

# Convert
for split_name in ["train", "valid"]:
    sp = os.path.join(src, split_name)
    if not os.path.isdir(sp):
        continue
    img_dir = os.path.join(sp, "images")
    if not os.path.isdir(img_dir):
        continue
    for f in os.listdir(img_dir):
        if not f.lower().endswith(".jpg"):
            continue
        base = os.path.splitext(f)[0]
        # Try to match class
        cls = None
        for old_name, new_name in dji5_map.items():
            if old_name.lower() in base.lower() or old_name.replace(" ", "").lower() in base.lower():
                cls = new_name
                break
        if cls is None:
            continue
        # Copy to ImageFolder format
        # Use "val" instead of "valid" to match training config
        out_split = "val" if split_name == "valid" else split_name
        out_dir = os.path.join(dst, out_split, cls)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(os.path.join(img_dir, f), os.path.join(out_dir, f))

# Count
for s in ["train", "val"]:
    sp = os.path.join(dst, s)
    if not os.path.isdir(sp):
        continue
    n = 0
    for c in os.listdir(sp):
        n += len(os.listdir(os.path.join(sp, c)))
    print(f"{s}: {n} images")
