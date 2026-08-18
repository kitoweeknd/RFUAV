"""Enrich DroneRFb canonical metadata with the two-level label hierarchy
(model_group / individual) and the TF-signature observations from the
2026-08-17 visual inspection. All inference fields are explicitly marked
inference_only — class names and existing fields are untouched."""
import os, json, glob

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"

TF_SIG = {
    "a": "dense wide burst groups in upper sub-band; several 15-20 MHz OFDM "
         "video blocks mid/low band",
    "c": "sparse weak narrow bursts; minimal video activity; narrowband "
         "carrier at band edge",
    "d": "dense wide bursts with interleaved narrow pulses; scattered OFDM "
         "blocks",
    "e": "sparse regular narrow bursts; strong continuous mid-band OFDM "
         "video strip",
    "f": "dense wide bursts; strong full-width bottom OFDM band",
    "g": "dense medium bursts; OFDM blocks left-mid plus bottom speckle",
}
FAMILY = ("consumer-drone proprietary OFDM digital link (DJI "
          "OcuSync/Lightbridge-like): quasi-periodic control bursts in a "
          "fixed upper sub-band + 10-20 MHz OFDM video bursts below")

updated = 0
for tax in sorted(os.listdir(CANON)):
    if not tax.startswith("ext.dronerfb."):
        continue
    code = tax.split(".")[-1]          # a1..g3 or background
    tdir = os.path.join(CANON, tax)
    for mpath in glob.glob(os.path.join(tdir, "*", "metadata.json")):
        md = json.load(open(mpath, encoding="utf-8"))
        if code == "background":
            md["label_hierarchy"] = {"model_group": "background",
                                     "individual_index": None}
            md["anonymous_model"] = False
        else:
            letter, idx = code[0], int(code[1])
            md["label_hierarchy"] = {
                "model_group": "ext.dronerfb.model_%s" % letter,
                "individual_index": idx,
                "note": "letter = anonymous model code as published by the "
                        "upstream twin_droneRF release (ScienceDB); digit = "
                        "physical unit of that model. Collapse to "
                        "model_group for type recognition (6 classes); use "
                        "full class for individual recognition (18 classes).",
            }
            md["anonymous_model"] = True
            md["tf_signature_observed"] = {
                "family": FAMILY,
                "model_specific": TF_SIG[letter],
                "observed_by": "TF visual inspection of IN_S0 slice 0, "
                               "2026-08-17",
                "inference_only": True,
            }
        json.dump(md, open(mpath, "w", encoding="utf-8"), indent=2)
        updated += 1

print("metadata enriched: %d recordings" % updated)
