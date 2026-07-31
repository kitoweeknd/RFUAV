# -*- coding: utf-8 -*-
"""Generate ``taxonomy/labels.json`` -- the authoritative class registry.

WHY A GENERATOR AND NOT A HAND-WRITTEN JSON
-------------------------------------------
Three files already hold pieces of the truth, and hand-copying between them is
how a taxonomy rots:

  dialect.py::_V         raw directory/filename spelling -> (vendor, model, role)
  link_protocols.json    the controlled vocabulary of radio-link families
  draft_recordings.csv   what was actually MEASURED per class (sessions, seconds)

So labels.json is derived: the vocabulary and the per-class evidence are read
from those files, and the only thing authored here is ``_PROTOCOL`` -- the
device -> protocol assignment, which exists nowhere else and cannot be measured
from the bytes.  Every assignment carries its basis in the same tuple, because
per fact F2 an unsourced protocol claim is indistinguishable from a guess.

THE ASSERTIONS ARE THE POINT
----------------------------
Building fails loudly rather than emitting a plausible-looking file when:

  * a protocol id is not in link_protocols.json      (typo / stale id)
  * a taxonomy_id in dialect._V has no assignment    (new device added silently)
  * an assignment names a class that does not exist  (dead entry after a rename)

That is what makes labels.json safe to cite in the datasheet: it cannot drift
from dialect.py without turning the build red.

WHAT THE ASSERTIONS DO NOT CATCH: A COLUMN THAT LIES QUIETLY
------------------------------------------------------------
The guards above catch drift between files.  They cannot catch a column that is
internally consistent and still misleads, and this file shipped two of those:

  * ``unique_seconds`` summed every row, including the 531 re-cut windows of a
    parent in the same session -- the double-count named in E-015 item 1.
  * ``eval_tier`` and ``eval_tier_including_foreign`` shipped with no basis
    column, so a tier derived from a filename pattern was indistinguishable from
    one derived from capture events.  ``grouped_cv`` held for 42 classes under
    the second name with nothing anywhere in the file to qualify it (E-016).

Both are now disclosed rather than fixed-and-forgotten, because both are facts
about the library, not bugs in the code: the tiering rule is right, the evidence
under it is thin.  ``tier_basis`` says which, per scope, and the scopes are never
mixed -- ``tier_basis()`` takes a triple whose singleton test is
``n_sessions == n_recordings``, so an own-scope session count paired with an
all-roots recording count returns a plausible answer to a different question.
That is why ``evidence()`` carries ``_own`` counters for all three inputs.

READ-ONLY over the library.  Writes exactly one file, under taxonomy/.

    python3 tools/rfuav_v2/taxonomy/build_labels.py \
        --recordings /tmp/gt/draft_recordings.csv
"""

from __future__ import annotations

import os
import io
import sys
import csv
import json
import argparse
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import dialect                                       # noqa: E402
import split_grouped                                 # noqa: E402

REGISTRY = os.path.join(_HERE, "link_protocols.json")
OUT = os.path.join(_HERE, "labels.json")

SCHEMA = "rfuav.taxonomy.labels/0.1"
VERSION = "0.1.0"
SPEC_REF = "memoria/planes/g2-canonical-spec-v0.1.md sec.2 (D-spec approved 2026-07-31)"

# ---------------------------------------------------------------------------
# E-007 / F4.  The v1 paper called PARROT "FHSS, 18.36 MHz occupied"; the
# capture is forensically a Parrot ANAFI on Wi-Fi 802.11a/n.  wifi.80211an is
# the ONLY row in the registry at evidence level `verified`, and this is the
# class it was verified on.  dialect.py still emits the directory spelling, so
# the rename is declared here and listed as a migration, not applied silently.
PARROT_CANONICAL = "uav.parrot.anafi"
PARROT_OBSERVED = "uav.parrot.parrot_unspecified"

# ---------------------------------------------------------------------------
# Device -> link-protocol assignment.
#
#   (protocol_id, basis)
#
# basis is prose naming WHY, and is copied verbatim into the JSON.  Per the
# registry's global_caveat (F2), every row below except the Parrot one is a
# PRIOR from vendor documentation for that model -- not an on-air measurement --
# because no capture logged the link setting.  `unknown` is used wherever the
# model ships more than one family and nothing disambiguates it; that is a
# result, not a gap to be filled by guessing.
_PROTOCOL = {
    # --- UAVs: DJI, by transmission-system generation --------------------
    "uav.dji.avata2":           ("dji.o4", "DJI Avata 2 ships O4."),
    "uav.dji.avata":            ("dji.o3", "DJI Avata ships O3."),
    "uav.dji.fpv_combo":        ("dji.o3", "DJI FPV (2021) ships OcuSync 3.0."),
    "uav.dji.mavic3_pro":       ("dji.o3plus", "Mavic 3 Pro ships O3+."),
    "uav.dji.mavic3":           ("dji.o3", "Mavic 3 ships O3."),
    "uav.dji.mavic_pro":        ("dji.o1", "Mavic Pro ships OcuSync 1."),
    "uav.dji.mini2":            ("dji.o2", "Mini 2 ships O2."),
    "uav.dji.mini3":            ("dji.o2", "Mini 3 ships O2."),
    "uav.dji.mini3_pro":        ("dji.o3", "Mini 3 Pro ships O3."),
    "uav.dji.mini4_pro":        ("dji.o4", "Mini 4 Pro ships O4."),
    "uav.dji.air2s":            ("dji.o3", "Air 2S ships O3."),
    "uav.dji.inspire2":         ("dji.lightbridge2", "Inspire 2 ships Lightbridge 2."),
    "uav.dji.matrice200":       ("dji.lightbridge2", "M200 series ships Lightbridge 2."),
    "uav.dji.matrice600_pro":   ("dji.lightbridge2", "M600 Pro ships Lightbridge 2."),
    "uav.dji.matrice300":       ("dji.o2", "M300 RTK ships OcuSync 2.0 Enterprise."),
    "uav.dji.matrice30t":       ("dji.o3", "M30T ships O3 Enterprise."),
    "uav.dji.phantom4_pro":     ("dji.lightbridge2", "P4 Pro ships Lightbridge HD."),
    "uav.dji.phantom4_pro_rtk": ("dji.o2", "P4 RTK ships OcuSync."),
    "uav.dji.matrice100": ("unknown",
        "M100 shipped with more than one link option across production runs and "
        "nothing in the capture names which; not guessed."),
    "uav.dji.phantom3": ("unknown",
        "Phantom 3 variants split across Wi-Fi (Standard/4K) and Lightbridge "
        "(Professional/Advanced); the directory does not name the variant."),
    "uav.dji.mavic2_air2": ("unknown",
        "Class name conflates two different aircraft (Mavic 2 / Air 2), so no "
        "single protocol can be attributed -- see variant_unresolved."),
    "uav.dji": ("unknown",
        "ZJU bare-'DJi' bucket: model itself unknown, so protocol is too."),
    "uav.dji.air2r": ("unknown",
        "Registered spelling only; never observed in the library."),

    # --- UAVs: non-DJI ---------------------------------------------------
    "uav.autel.evo_nano": ("autel.skylink", "Autel EVO Nano ships SkyLink."),
    PARROT_CANONICAL: ("wifi.80211an",
        "MEASURED, not a prior: E-007/F4 forensics identify a Parrot ANAFI on "
        "Wi-Fi 802.11a/n, refuting the v1 paper's 'FHSS, 18.36 MHz'.  This is "
        "the sole `verified` row in link_protocols.json."),

    # --- handheld RC / ground link ---------------------------------------
    "rc.devention.devo_10":     ("walkera.devo", "Walkera/Devention DEVO series link."),
    "rc.flysky.el18":           ("flysky.afhds2a", "FlySky EL18 ships AFHDS 2A."),
    "rc.flysky.fs_i6x":         ("flysky.afhds2a", "FlySky FS-i6X ships AFHDS 2A."),
    "rc.flysky.nv14":           ("flysky.afhds2a", "FlySky NV14 ships AFHDS 2A."),
    "rc.frsky.taranis_plus":    ("frsky.accst_d16", "Taranis X9D Plus ships ACCST D16."),
    "rc.frsky.x9dp2019":        ("frsky.accst_d16",
        "X9D Plus 2019 ships ACCST D16; it is field-reflashable to ACCESS and "
        "the capture did not record which firmware was on it (F2)."),
    "rc.frsky.x14":             ("frsky.access", "FrSky X14 ships ACCESS."),
    "rc.frsky.x20":             ("frsky.access", "FrSky X20 ships ACCESS."),
    "rc.frsky.x20r":            ("frsky.access", "FrSky X20R ships ACCESS."),
    "rc.futaba.t10j":           ("futaba.tfhss", "Futaba T10J ships T-FHSS."),
    "rc.futaba.t14sg":          ("futaba.fasstest",
        "T14SG default is FASSTest; the radio also offers S-FHSS from the menu."),
    "rc.futaba.t16iz":          ("futaba.fasstest",
        "T16IZ default is FASSTest; also offers T-FHSS/S-FHSS from the menu."),
    "rc.futaba.t18sz":          ("futaba.fasstest",
        "T18SZ default is FASSTest; also offers T-FHSS/S-FHSS from the menu."),
    "rc.futaba.t61z":           ("unknown",
        "'T61Z' is not a Futaba product; suspected transposition of T16IZ.  The "
        "protocol is left unknown until the spelling is resolved (E-002)."),
    "rc.herelink.hx4":          ("microhard.pmddl2450",
        "Herelink HX4 ground link is built on a Microhard pMDDL2450 module."),
    "rc.jrpropo.xg14":          ("jrpropo.dmss", "JR Propo XG14 ships DMSS."),
    "rc.jrpropo.xg7":           ("jrpropo.dmss", "JR Propo XG7 ships DMSS."),
    "rc.radiolink.at10_ii":     ("radiolink.dsss_fhss", "RadioLink AT10II link."),
    "rc.radiolink.at9s":        ("radiolink.dsss_fhss", "RadioLink AT9S link."),
    "rc.radiolink.at9s_pro":    ("radiolink.dsss_fhss", "RadioLink AT9S Pro link."),
    "rc.siyi.ft24":             ("siyi.fhss", "SIYI FT24 is an FHSS RC link."),
    "rc.siyi.mk15":             ("siyi.datalink", "SIYI MK15 uses the SIYI datalink."),
    "rc.siyi.mk32":             ("siyi.datalink", "SIYI MK32 uses the SIYI datalink."),
    "rc.skydroid.h12":          ("skydroid.link", "Skydroid H12 integrated link."),
    "rc.skydroid.t10":          ("skydroid.link", "Skydroid T10 integrated link."),
    "rc.wfly.et10":             ("wfly.wfly24", "WFLY ET10 ships the WFLY 2.4G link."),
    "rc.wfly.et16s":            ("wfly.wfly24", "WFLY ET16S ships the WFLY 2.4G link."),
    "rc.wfly.wft09sii":         ("wfly.wfly24", "WFLY WFT09SII ships the WFLY 2.4G link."),

    # EdgeTX / OpenTX radios: the transmitter is a HOST, and the protocol is
    # whatever module was fitted.  Per F7 no capture log records the module, so
    # these stay unknown -- this is why elrs.24 and multiprotocol.4in1 are
    # registered in link_protocols.json but assigned to nobody.
    "rc.jumper.t14": ("unknown",
        "EdgeTX host radio; internal/external module not recorded (F7).  "
        "Plausible: elrs.24 or multiprotocol.4in1 -- neither is asserted."),
    "rc.jumper.tpro_v2": ("unknown",
        "EdgeTX host radio; module not recorded (F7).  Plausible: elrs.24 or "
        "multiprotocol.4in1 -- neither is asserted."),
    "rc.radiomaster.boxer": ("unknown",
        "EdgeTX host radio; module not recorded (F7).  Plausible: elrs.24 or "
        "multiprotocol.4in1 -- neither is asserted."),
    "rc.radiomaster.tx16s": ("unknown",
        "EdgeTX host radio; module not recorded (F7).  Plausible: elrs.24 or "
        "multiprotocol.4in1 -- neither is asserted."),
    "rc.vbar.vbar_control": ("unknown",
        "VBar Control link family not established, and the class spelling is "
        "itself unresolved (VBARFLY vs VBAR)."),

    # 云卓 / Yunzhuo.  NOT assigned skydroid.link: see _VENDOR_FAMILY below --
    # assigning the Skydroid protocol here would smuggle in the very vendor
    # merge that fact F1 forbids.
    "rc.yunzhuo.h12": ("unknown",
        "Yunzhuo link family not independently established; skydroid.link is "
        "NOT assumed (F1: Yunzhuo H12 and Skydroid H12 are distinct devices)."),
    "rc.yunzhuo.h16": ("unknown", "Yunzhuo link family not established (F1)."),
    "rc.yunzhuo.h30": ("unknown", "Yunzhuo link family not established (F1)."),
    "rc.yunzhuo.t12": ("unknown", "Yunzhuo link family not established (F1)."),

    # --- not devices ------------------------------------------------------
    "bg.kong": (None,
        "'KONG' (空) is the empty-channel background class: there is no "
        "transmitter, so link_protocol is not applicable."),
    "unk.weizhi": (None,
        "F3: a ~940 MHz probe, not a drone link, and slated for quarantine.  "
        "Protocol not applicable."),
}

# ---------------------------------------------------------------------------
# The decision dialect.py explicitly delegated here.
#
# dialect.py: "spec s2 also merges SKYDROID == 云卓(YUNZHUO) into
# vendor=skydroid.  We do NOT apply that here because fact F1 established
# SKYDROID H12 and YUNZHUO H12 are two distinct physical devices; merging the
# vendor namespace would collide their taxonomy_id.  Exposed for the taxonomy
# step to decide:"
#
# DECISION: do not merge the namespace.  rc.skydroid.h12 and rc.yunzhuo.h12
# stay distinct taxonomy_ids, because they are two distinct physical radios
# that were both captured (F1) and a merge would silently fuse two classes into
# one -- an irreversible loss dressed up as normalisation.  Spec sec.2's merge is
# honoured as a NON-KEY grouping field instead: anyone who wants the
# vendor-family view groups on `vendor_family`, and gets it without the ids
# colliding.  This deviates from the literal wording of spec sec.2, so it is
# logged as an open decision for ratification.
_VENDOR_FAMILY = {"skydroid": "skydroid_yunzhuo", "yunzhuo": "skydroid_yunzhuo"}

MERGE_DECISION = {
    "question": "spec sec.2 merges vendor SKYDROID == 云卓(YUNZHUO); dialect.py "
                "refused and delegated the call to this file.",
    "decision": "NOT merged in the taxonomy_id namespace.",
    "rationale": "F1: Skydroid H12 and Yunzhuo H12 are two distinct physical "
                 "radios, both captured.  Merging vendor would collide "
                 "rc.skydroid.h12 with rc.yunzhuo.h12 and fuse two real "
                 "classes into one -- data loss disguised as normalisation.",
    "how_spec_s2_is_honoured": "as the non-key field `vendor_family` = "
                               "'skydroid_yunzhuo' on all six affected classes, "
                               "so the merged view is a GROUP BY, not a rename.",
    "status": "deviates from the literal text of spec sec.2 -- OPEN, awaiting "
              "ratification.",
}

FOREIGN_ROOTS = split_grouped.NON_OWN_ROOTS


def vocabulary():
    """taxonomy_id -> {vendor, model, role, variant_unresolved, raw_keys}."""
    out = collections.OrderedDict()
    for raw, (vend, mod, role, unres) in dialect._V.items():
        tid = ".".join(x for x in (role, vend, mod) if x) or "unk"
        e = out.setdefault(tid, {"vendor": vend, "model": mod, "device_role": role,
                                 "variant_unresolved": False, "raw_keys": []})
        e["raw_keys"].append(raw)
        e["variant_unresolved"] = e["variant_unresolved"] or bool(unres)
    return out


def evidence(path):
    """taxonomy_id -> measured counts, from the draft manifest.

    Four duration figures come out of here, not one, and none of them may be
    dropped in favour of another:

      nominal_seconds   -- every recording row's seconds, summed.  Overstates
                           the library, because 531 of 700 own-capture rows are
                           re-cut windows of a parent in the same session.
      unique_seconds    -- uncut rows only (E-015 item 1).  The honest figure.
      n_uncut           -- how many rows the unique sum is over, so the reader
                           can see the ratio rather than trust a single number.
      duration_fallbacks-- rows whose seconds were inferred from size rather
                           than measured, so an inferred column cannot pass as
                           a measured one.

    An earlier version of this function summed EVERY row and published the
    result under the name ``unique_seconds``.  It could not have done otherwise:
    the column that distinguishes the two, ``is_uncut``, did not exist in
    recordings.csv until build_manifests.py began exporting it.  That is why
    the guard below is fatal rather than a fallback -- the only fallback
    available is the bug.
    """
    # The `_own` counters exist because tier_basis() takes a TRIPLE and its
    # singleton test is literally `n_sessions == n_recordings`.  Handing it an
    # own-scope session count against an all-roots recording count does not
    # fail -- it silently answers a different question.  Every scope that gets a
    # tier published must therefore carry its own matching recording count and
    # its own unique-second sum.
    out = collections.defaultdict(
        lambda: {"n_recordings": 0, "n_uncut": 0, "sessions": set(),
                 "sessions_own": set(), "roots": collections.Counter(),
                 "nominal_seconds": 0.0, "unique_seconds": 0.0,
                 "n_recordings_own": 0, "n_uncut_own": 0,
                 "unique_seconds_own": 0.0,
                 "duration_fallbacks": 0,
                 "center_freq_mhz": set(), "is_misfiled": 0})
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        split_grouped.require_is_uncut(rdr.fieldnames, path)
        for r in rdr:
            if (r.get("artifact_class") or "recording") != "recording":
                continue
            e = out[r["taxonomy_id"]]
            e["n_recordings"] += 1
            e["sessions"].add(r["session_id"])
            root = r.get("source_root") or ""
            e["roots"][root] += 1
            is_own = root not in FOREIGN_ROOTS
            if is_own:
                e["sessions_own"].add(r["session_id"])
                e["n_recordings_own"] += 1
            # row_seconds, not float(delivered_duration_s): a duration-less row
            # contributed 0.0 here and size/8/100e6 in build_manifests.py, so
            # the same class had two different second counts depending on which
            # file printed it.  One reader, one rate assumption, one fallback
            # counter.
            secs, inferred = split_grouped.row_seconds(r)
            e["nominal_seconds"] += secs
            if inferred:
                e["duration_fallbacks"] += 1
            if split_grouped.row_is_uncut(r):
                e["n_uncut"] += 1
                e["unique_seconds"] += secs
                if is_own:
                    e["n_uncut_own"] += 1
                    e["unique_seconds_own"] += secs
            if r.get("center_freq_mhz"):
                try:
                    e["center_freq_mhz"].add(int(round(float(r["center_freq_mhz"]))))
                except ValueError:
                    pass
            if r.get("is_misfiled") in ("1", "True", "true"):
                e["is_misfiled"] += 1
    return out


def band_of(freqs):
    b = set()
    for f in freqs:
        if f < 1000:
            b.add("sub_1g")
        elif f < 2000:
            b.add("1_2g")
        elif 2300 <= f < 2600:
            b.add("2_4g")
        elif 5000 <= f < 6000:
            b.add("5_8g")
        else:
            b.add("other")
    return sorted(b)


def build(recordings):
    with io.open(REGISTRY, "r", encoding="utf-8") as fh:
        reg = json.load(fh)
    valid = {p["id"] for p in reg["protocols"]}

    vocab = vocabulary()
    ev = evidence(recordings)

    # Apply the E-007 rename: the canonical id replaces the observed one, and
    # the observed id survives as `supersedes` so the manifest still joins.
    if PARROT_OBSERVED in vocab:
        entry = vocab.pop(PARROT_OBSERVED)
        entry["supersedes"] = [PARROT_OBSERVED]
        entry["model"] = "anafi"
        vocab[PARROT_CANONICAL] = entry
        if PARROT_OBSERVED in ev:
            ev[PARROT_CANONICAL] = ev.pop(PARROT_OBSERVED)

    # --- the assertions -------------------------------------------------
    errs = []
    for tid, (pid, _why) in sorted(_PROTOCOL.items()):
        if pid is not None and pid not in valid:
            errs.append("protocol id %r (on %s) is not in link_protocols.json" % (pid, tid))
        if tid not in vocab:
            errs.append("assignment for %r but no such class in dialect._V" % tid)
    for tid in sorted(vocab):
        if tid not in _PROTOCOL:
            errs.append("class %r has no link_protocol assignment "
                        "(add an explicit 'unknown' if that is the answer)" % tid)
    if errs:
        for e in errs:
            sys.stderr.write("FAIL: %s\n" % e)
        raise SystemExit("labels.json NOT written: %d consistency error(s)" % len(errs))

    labels = []
    for tid in sorted(vocab):
        v = vocab[tid]
        e = ev.get(tid)
        pid, why = _PROTOCOL[tid]
        n_own = len(e["sessions_own"]) if e else 0
        n_all = len(e["sessions"]) if e else 0
        row = collections.OrderedDict()
        row["taxonomy_id"] = tid
        row["vendor"] = v["vendor"]
        row["model"] = v["model"]
        row["device_role"] = v["device_role"]
        if v["vendor"] in _VENDOR_FAMILY:
            row["vendor_family"] = _VENDOR_FAMILY[v["vendor"]]
        row["raw_keys"] = sorted(v["raw_keys"])
        if v.get("supersedes"):
            row["supersedes"] = v["supersedes"]
            row["rename_errata"] = "E-007"
            row["migration_required"] = (
                "dialect.py still emits %s; build_manifests.py must map it to "
                "%s in the same commit that rebuilds the manifest, or the two "
                "files disagree." % (PARROT_OBSERVED, PARROT_CANONICAL))
        row["link_protocol"] = pid
        row["link_protocol_basis"] = why
        row["link_protocol_evidence"] = (
            "not_applicable" if pid is None else
            "verified" if tid == PARROT_CANONICAL else
            "unknown" if pid == "unknown" else "factory_default_prior")
        row["variant_unresolved"] = v["variant_unresolved"]
        row["observed_in_library"] = e is not None
        row["n_recordings"] = e["n_recordings"] if e else 0
        row["n_uncut_recordings"] = e["n_uncut"] if e else 0
        row["n_recordings_own_capture"] = e["n_recordings_own"] if e else 0
        row["n_uncut_recordings_own_capture"] = e["n_uncut_own"] if e else 0
        row["n_sessions_total"] = n_all
        row["n_sessions_own_capture"] = n_own
        # Both figures, always, side by side.  Publishing only the unique sum
        # would hide how much of the class is re-cut; publishing only the
        # nominal sum is the double-count E-015 item 1 exists to name.  Dropping
        # either one just relocates the dishonesty (E-015 item 6).
        row["nominal_seconds"] = round(e["nominal_seconds"], 3) if e else 0.0
        row["unique_seconds"] = round(e["unique_seconds"], 3) if e else 0.0
        row["unique_seconds_own_capture"] = (
            round(e["unique_seconds_own"], 3) if e else 0.0)
        row["duration_inferred_rows"] = e["duration_fallbacks"] if e else 0
        row["bands"] = band_of(e["center_freq_mhz"]) if e else []
        row["source_roots"] = sorted(e["roots"]) if e else []
        row["own_capture"] = n_own > 0
        row["reportable_in_class_count"] = n_own > 0
        row["eval_tier"] = split_grouped.tier_of(n_own)
        row["eval_tier_including_foreign"] = split_grouped.tier_of(n_all)
        # E-016, applied to the file the datasheet cites.  A tier is a CLAIM
        # about what split exists; the basis says whether the claim is BACKED.
        # `grouped_cv` on a class whose every session holds one recording is a
        # session count that a filename pattern produced, not a capture event --
        # and `eval_tier_including_foreign == grouped_cv` holds for 42 classes
        # here with no qualifier anywhere else in this file.  Both scopes get a
        # basis, computed from a triple matched to that scope, never mixed.
        # The rule itself is never re-implemented: one definition, in the
        # splitter, or the two files drift into disagreeing about the same word.
        if n_own > 0:
            row["tier_basis"] = split_grouped.tier_basis(
                n_own, e["n_recordings_own"], e["unique_seconds_own"])
            row["eval_tier_supported"] = split_grouped.tier_is_supported(
                n_own, e["n_recordings_own"], e["unique_seconds_own"])
        else:
            # No own sessions -> no own tier to back.  Emitting `thin_signal`
            # for 0.0 s of own signal would be arithmetically true and read as
            # a measurement of a class that was never captured here.
            row["tier_basis"] = None
            row["eval_tier_supported"] = None
        if e is not None:
            row["tier_basis_including_foreign"] = split_grouped.tier_basis(
                n_all, e["n_recordings"], e["unique_seconds"])
            row["eval_tier_including_foreign_supported"] = (
                split_grouped.tier_is_supported(
                    n_all, e["n_recordings"], e["unique_seconds"]))
        else:
            row["tier_basis_including_foreign"] = None
            row["eval_tier_including_foreign_supported"] = None
        if e and e["is_misfiled"]:
            row["misfiled_recordings"] = e["is_misfiled"]
            row["misfiled_errata"] = "E-004"
        labels.append(row)

    own = [r for r in labels if r["own_capture"]]
    foreign_only = [r for r in labels if r["observed_in_library"] and not r["own_capture"]]
    unobserved = [r for r in labels if not r["observed_in_library"]]
    assigned = {r["link_protocol"] for r in labels if r["link_protocol"]}

    doc = collections.OrderedDict()
    doc["schema"] = SCHEMA
    doc["version"] = VERSION
    doc["spec_ref"] = SPEC_REF
    doc["generated_by"] = "tools/rfuav_v2/taxonomy/build_labels.py"
    doc["purpose"] = (
        "Authoritative class registry for RFUAV v2.  One entry per taxonomy_id. "
        "Vocabulary and per-class evidence are DERIVED (from dialect.py::_V and "
        "the recordings manifest); the only authored content is the device -> "
        "link_protocol assignment, each with its basis.  Protocol prose lives "
        "in link_protocols.json and is never inlined here.")
    doc["headline_counts"] = collections.OrderedDict([
        ("taxonomy_ids_total", len(labels)),
        ("observed_in_library", len(labels) - len(unobserved)),
        ("reportable_own_capture", len(own)),
        ("foreign_root_only", len(foreign_only)),
        ("registered_but_never_observed", len(unobserved)),
        ("honesty_note",
         "The class count defensible as RFUAV's own capture is %d, NOT %d.  The "
         "other %d classes exist only inside the ZJU and .dat_pic roots, which "
         "E-005/E-006 measured as int16-converted with cross-class duplicate "
         "content and E-014.3 forbids counting.  Any '>=60 classes' claim built "
         "on %d is a claim about someone else's data."
         % (len(own), len(labels) - len(unobserved), len(foreign_only),
            len(labels) - len(unobserved))),
    ])
    # Over the REPORTABLE classes only.  Counting all 64 here would fold the 23
    # foreign-only classes (n_sessions_own == 0) into `unsplittable` and make the
    # own-capture library look far less splittable than it is.
    doc["eval_tier_histogram_own_capture"] = dict(
        collections.Counter(r["eval_tier"] for r in own))
    doc["eval_tier_note"] = (
        "Histogram is over the %d own-capture classes.  `unsplittable` here "
        "means the class has exactly ONE own session, so no leakage-clean split "
        "of it exists at any ratio -- it can be trained on or held out whole, "
        "never both.  The %d foreign-only classes are excluded (they have zero "
        "own sessions by definition)." % (len(own), len(foreign_only)))
    # E-016 disclosure, aggregated.  Two totals because they answer two
    # different questions and the gap between them is itself the finding:
    # `n_tier_unsupported` counts classes whose TIER asserts a split it cannot
    # back; `n_thin_signal` counts classes under the 10 s floor regardless of
    # tier.  The difference is exactly the `unsplittable` rows -- they are thin
    # AND exempt, because a one-session class claims no split to begin with, so
    # there is nothing for thin signal to undermine.  Reporting only the first
    # hides 2 thin classes; only the second overstates the split problem by 2.
    observed = [r for r in labels if r["observed_in_library"]]

    def _disclosure(rows, basis_key, supported_key):
        return collections.OrderedDict([
            ("n_classes", len(rows)),
            ("n_tier_unsupported",
             sum(1 for r in rows if r[supported_key] is False)),
            ("n_thin_signal",
             sum(1 for r in rows if r[basis_key]
                 and split_grouped.BASIS_THIN in r[basis_key])),
            ("n_singleton_sessions",
             sum(1 for r in rows if r[basis_key]
                 and split_grouped.BASIS_SINGLETON in r[basis_key])),
        ])

    doc["eval_tier_disclosure"] = collections.OrderedDict([
        ("own_capture", _disclosure(own, "tier_basis", "eval_tier_supported")),
        ("all_roots_observed",
         _disclosure(observed, "tier_basis_including_foreign",
                     "eval_tier_including_foreign_supported")),
        ("reconciles_with",
         "manifests/library_totals.csv publishes the same four counts over the "
         "same two scopes.  A mismatch between that file and this one means one "
         "of the two aggregations is wrong -- they read the same recordings.csv "
         "and call the same split_grouped.tier_basis()."),
        ("do_not_quote",
         "Do NOT quote eval_tier_including_foreign == grouped_cv as "
         "'cross-validatable classes'.  Check eval_tier_including_foreign_"
         "supported first: %d of the %d observed classes carry a tier their own "
         "evidence does not support."
         % (sum(1 for r in observed
                if r["eval_tier_including_foreign_supported"] is False),
            len(observed))),
    ])
    doc["link_protocol_caveat"] = (
        "See link_protocols.json::global_caveat (F2).  Every assignment below "
        "at evidence level `factory_default_prior` is a PRIOR from vendor "
        "documentation for that model, never an on-air measurement, and must "
        "not be used as a training label or evaluation ground truth.  Exactly "
        "one row is `verified`: %s." % PARROT_CANONICAL)
    doc["protocols_registered_but_unassigned"] = collections.OrderedDict([
        ("ids", sorted(valid - assigned - {"unknown"})),
        ("why", "F7: no capture log records which internal/external module was "
                "fitted to the EdgeTX/OpenTX host radios (Jumper T14, Jumper "
                "TPro V2, RadioMaster Boxer, RadioMaster TX16S).  Registering a "
                "protocol without assigning it is the honest encoding of "
                "'plausible but unverifiable' -- see those classes' basis text."),
    ])
    doc["vendor_merge_decision"] = MERGE_DECISION
    doc["grouping_key_warning"] = (
        "session_id, never class_dir and never taxonomy_id, is the grouping key "
        "for splitting.  See class_aliases.json::_policy -- the same device "
        "appears under several directory spellings, and a per-directory split "
        "puts one device on both sides of the boundary.")
    doc["labels"] = labels
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recordings", required=True,
                    help="recordings manifest CSV (measured per-class evidence)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)

    doc = build(a.recordings)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")

    h = doc["headline_counts"]
    print("wrote %s" % a.out)
    print("  taxonomy_ids            : %d" % h["taxonomy_ids_total"])
    print("  observed in library     : %d" % h["observed_in_library"])
    print("  REPORTABLE (own capture): %d" % h["reportable_own_capture"])
    print("  foreign-root only       : %d" % h["foreign_root_only"])
    print("  never observed          : %d" % h["registered_but_never_observed"])
    print("  eval tiers              : %s" % doc["eval_tier_histogram_own_capture"])
    print("  protocols unassigned    : %s"
          % ", ".join(doc["protocols_registered_but_unassigned"]["ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
