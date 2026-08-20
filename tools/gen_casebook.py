"""Generate canonical_casebook.md: one collapsible <details> block per class.

Summary line: class | status | sources | recordings | slices | SNR range.
Expanded: per-recording table (split, slices, images, SNR, duration, origin)
plus label-hierarchy / TF-signature notes for the DroneRFb classes.
"""
import os, csv, json, time
from collections import defaultdict

ROOT = r"E:\Drone_dataset\RFUAV-v2"
CANON = os.path.join(ROOT, "rawdata", "DroneRFb")  # only used for DroneRFb notes
MANI = os.path.join(ROOT, "manifests")
OUT = os.path.join(MANI, "canonical_casebook.md")
NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi", "bg.dronerf"}

recs = list(csv.DictReader(open(os.path.join(MANI, "inventory_recordings.csv"),
                                newline="", encoding="utf-8")))
by_class = defaultdict(list)
for r in recs:
    by_class[r["taxonomy_id"]].append(r)


def status_of(tax, v):
    sp = set(x["split"] for x in v)
    if tax in NON_TRANSMITTER:
        return "非发射机"
    if "train" in sp and "test" in sp:
        return "可评"
    if "train" in sp:
        return "仅train"
    return "excluded"


def drfb_note(tax):
    """label hierarchy + tf signature from the first recording's metadata."""
    tdir = os.path.join(CANON, tax)
    for rec in sorted(os.listdir(tdir)):
        mp = os.path.join(tdir, rec, "metadata.json")
        if os.path.exists(mp):
            md = json.load(open(mp, encoding="utf-8"))
            lh = md.get("label_hierarchy")
            tf = md.get("tf_signature_observed")
            lines = []
            if lh and lh.get("model_group"):
                lines.append("- **型号组**: `%s`  个体号: %s（匿名型号×3实体，"
                             "种类识别折叠到型号组，个体识别用完整类名）"
                             % (lh["model_group"], lh.get("individual_index")))
            if tf:
                lines.append("- **时频签名（观察推断）**: %s" % tf["model_specific"])
            return lines
    return []


L = ["# RFUAV-v2 Canonical Casebook（逐类可展开明细）", "",
     "生成时间: %s · 数据: inventory_recordings.csv (%d recordings)"
     % (time.strftime("%Y-%m-%d %H:%M"), len(recs)), "",
     "点击每个类展开逐录制明细。摘要行: 状态 | 来源 | 录制数(train/test) | 切片 | SNR 范围。",
     ""]

GROUPS = [("DJI 无人机", lambda t: t.startswith("uav.dji")),
          ("其他无人机", lambda t: t.startswith("uav.") and not t.startswith("uav.dji")),
          ("遥控器/RC 发射机", lambda t: t.startswith("rc.")),
          ("DroneRFb 匿名孪生个体", lambda t: t.startswith("ext.dronerfb") and "background" not in t),
          ("背景/隔离", lambda t: t in NON_TRANSMITTER)]

seen = set()
for gname, pred in GROUPS:
    taxes = [t for t in sorted(by_class) if pred(t) and t not in seen]
    seen.update(taxes)
    if not taxes:
        continue
    L.append("## %s（%d 类）" % (gname, len(taxes)))
    L.append("")
    for tax in taxes:
        v = sorted(by_class[tax], key=lambda x: x["recording"])
        snrs = [float(x["native_snr_db"]) for x in v if x["native_snr_db"]]
        srng = "%.1f..%.1f dB" % (min(snrs), max(snrs)) if snrs else "-"
        tr = sum(1 for x in v if x["split"] == "train")
        te = sum(1 for x in v if x["split"] == "test")
        sl = sum(int(x["n_slices"]) for x in v)
        srcs = "+".join(sorted(set(x["source_dataset"].replace("DroneRFb-DIR", "RFb")
                                   .replace("DroneRFa", "RFa") for x in v)))
        gb = sum(int(x["slice_bytes"]) for x in v) / 1e9
        L.append("<details>")
        L.append("<summary><b>%s</b> — %s | %s | %d 录制 (%d/%d) | %d 切片 | "
                 "%.1f GB | SNR %s</summary>" % (
                     tax, status_of(tax, v), srcs, len(v), tr, te, sl, gb, srng))
        L.append("")
        notes = drfb_note(tax) if tax.startswith("ext.dronerfb") and tax not in NON_TRANSMITTER else []
        L.extend(notes)
        if notes:
            L.append("")
        L.append("| recording | source | split | slices | imgs | SNR(dB) | dur(s) | origin |")
        L.append("|---|---|---|---|---|---|---|---|")
        for x in v:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                x["recording"], x["source_dataset"].replace("DroneRFb-DIR", "RFb")
                .replace("DroneRFa", "RFa"), x["split"], x["n_slices"],
                x["n_images"], x["native_snr_db"] or "-",
                x["delivered_seconds"] or "-",
                (x["source_ref"] or "-").replace("\\", "/")[:60]))
        L.append("")
        L.append("</details>")
        L.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("casebook written: %s (%d classes, %d recordings)" % (OUT, len(seen), len(recs)))
