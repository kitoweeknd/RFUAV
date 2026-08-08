# -*- coding: utf-8 -*-
"""RFUAV v1 -> v2.0 filename dialect parser.

Implements the cleaning-list row of ``memoria/planes/g2-canonical-spec-v0.1.md`` s5::

    | 4 种命名方言+typo | 方言解析器(带单测)-> 归一字段;`naming_dialect` 保留原方言标记 | sidecar |

HARD CONSTRAINTS
----------------
* stdlib only (csv/json/re/os/sys/argparse/collections). No numpy/pandas/pyarrow.
* python >= 3.9 compatible; must run unchanged on the Windows data server (3.11.4).
* Pure function of the *relative path*: never touches the filesystem, never reads
  file sizes.  (The size evidence below was gathered offline from the E1 scan and
  is recorded here as documentation, not as runtime input.)
* NEVER invent a value.  A field that is not present in the path is ``None``.
  Any numeric group that cannot be explained lands in ``unparsed_residue``.

=============================================================================
WHAT EVERY NUMERIC GROUP MEANS  (empirically derived from the E1 scan, 954 rows)
=============================================================================

The single hardest question in v1 naming is the *three* ``_Nm`` groups, e.g.::

    DJFPVCOMBO-16db-90db_5760m_100m_10m.iq
                         ^^^^^ ^^^^ ^^^
                           |     |    +--- VTSBW  (MHz)
                           |     +-------- sample rate (MS/s), NOT metres
                           +-------------- centre frequency (MHz), NOT metres

Evidence (all measured, none assumed):

1. group #1 = **centre frequency in MHz**.  Observed value set across the whole
   library is {940, 2410, 2420, 2440, 2450, 2460, 2470, 5735, 5740, 5750, 5760,
   5765, 5770, 5775, 5790, 5800}.  Those are ISM / U-NII channel centres and they
   track the class: every 2.4 GHz handheld RC is 2410-2470, every 5.8 GHz UAV is
   5735-5800.  940 is the WEIZHI 940 MHz probe (fact F3).  A "2440 metre" or
   "5800 metre" stand-off for a handheld transmitter test is physically absurd.

2. group #2 = **sample rate in MS/s**, always 100.  Proof by byte arithmetic on
   complex-float32 (8 B/sample):
       PublicDataset\\...\\pack1_0-1s.iq          = 800,000,000 B for 1 s  -> 100 MS/s
       rawdata\\DJMAVIC3PRO\\...-10m_0-2s.iq      = 1,600,000,000 B for 2 s -> 100 MS/s
   Some files drop the unit letter entirely (``..._2440m_100.iq``), which only
   makes sense for a magnitude prefix ("100 M"), never for "100 metres".

   SCOPE WARNING (measured 2026-07-31 over all 953 signal files with real sizes
   from e1_inventory.csv).  "always 100" is true only for the dialects that
   *have* a group #2, and those are the minority.  ``sample_rate_msps`` is
   populated by this parser for 276/953 files (29 %); for the other 71 % the
   filename carries no rate at all and the library is NOT uniformly 100 MS/s:
       D1/D2/D3/D8 (group #2 present) : 100 MS/s, 458/458, no exceptions
       D4_gain_underscore (no group)  : 100 MS/s x113  **and 200 MS/s x28**
                                        (FUtabaT14SG_2440_daifei_80dB(2)_*,
                                         herelinkHx4_2420_duima_gain60dB_*)
       D9_bare_model      (no group)  : 15 MS/s x5, 100 MS/s x4, 5 MS/s x2
       one file self-labels 75000000SPS in its own name.
   Where this parser does emit a rate it is never contradicted by byte
   arithmetic (0/276 disagreements), so the field is trustworthy -- it is just
   frequently absent.  **Downstream code must DERIVE the rate from
   size / bytes_per_sample / duration and must never default to 100.**

   DTYPE IS PER-SOURCE-ROOT, NOT GLOBAL (measured by reading bytes, 2026-07-31).
   The ``/ 8`` above holds only for cf32_le roots.  Two roots are ci16_le
   (interleaved int16, **4 B/sample**; ``sc16_le`` is the SDR-vendor spelling of
   the same layout and resolves to this token) and were verified by reading their
   payload: read as float32 they yield 91-99 % finite values with rms ~1e37 and
   min ~ -3.4e38 (near FLT_MAX), whereas every genuine cf32_le file reads
   100.00 % finite with rms 7e-4 .. 4e-1 and mean ~1e-6:
       .dat_pic\\Mavic2Air2\\              ci16_le   1 file
       浙大被预处理为IQ的数据\\             ci16_le   81 files  (see ERRATA E-010/A3)
       rawdata / crop_data / PublicDataset / 过大的原始数据   cf32_le  (6/6 probed)
   ``bytes_per_sample`` must therefore be a manifest FIELD keyed on source root,
   never a constant.  Neither ci16_le root carries a ``_A-Bs`` window, so no rate
   figure in this docstring was computed with the wrong divisor.

   Corollary -- truncation detector.  The filename window ``_A-Bs`` is the
   **requested** window, not the delivered one.  When the slicer was pointed at a
   source recording that ended before second B it wrote only the bytes that
   existed and still named the file ``_A-Bs``.  Detection is therefore NOT "is
   the derived rate fractional" -- the rate is a perfectly round 100 or 200 MS/s
   in every case -- but:

       delivered_s = size / bytes_per_sample / rate      # rate from siblings
       truncated   = delivered_s < (B - A) - epsilon

   Proven, not inferred: for 25/27 candidates a source file exists whose size
   satisfies ``src_size == A * rate * 8 + slice_size`` exactly (delta 0 B), and
   for 4/4 spot-checked pairs the slice is byte-identical to that source at byte
   offset ``A * rate * 8`` across head/mid/tail 64 KB windows.  So the payload is
   NOT corrupt -- it is a valid short tail with a filename that overstates its
   duration.  See ERRATA E-008.  Do not conflate this with a legitimate rate
   variant: an earlier detector flagged the exp3 5 MS/s files as "33 %
   truncated" purely because their siblings ran at 100 MS/s.

3. group #3 = **VTSBW in MHz** (the tuned/sweep bandwidth).  Proof by directory
   cross-check: ``PublicDataset\\DJI FPV COMBO\\VTSBW=10|20|40`` mirrors exactly the
   ``_10m|_20m|_40m`` suffixes seen for the same model in ``rawdata\\DJFPVCOMBO``;
   AVATA2 additionally has VTSBW=60 in both places.  Value set {10,20,40,60}.

**Therefore ``distance_m`` is null for every file in the library.**  No RFUAV v1
filename encodes a distance.  This parser emits the field (it is in the agreed
schema) but it is *always* ``None``; see the coverage report.

The two ``-Ndb`` groups::

    DJMINI3--10db-60db_2470m_100m_10m.iq
             ^^^^ ^^^^
               |    +--- rx gain (dB)          : CONSTANT within a class
               +-------- nominal ladder step   : VARIES within a class

Evidence: per-class value sets measured over the 60 three-group files ->
  DJFPVCOMBO   slot1 {16,22,28,33,35,36,41}  slot2 {90}
  DJMAVIC3PRO  slot1 {10,11,14,16}           slot2 {90}
  DJMINI3      slot1 {5,10,28,35,46}         slot2 {60}
  DJMINI4PRO   slot1 {5,10,17}               slot2 {60}
Slot 2 is invariant per class and matches spec appendix B ("MINI3 全阶梯(定增益
60db)").  Slot 1 is the attenuation / intent ladder.  Per verificacion-empirica-04
the ladder is monotonic but NOT dB-true (slopes 0.58-0.77), so slot 1 is emitted
as ``attenuation_label_db`` and NEVER as ``snr_nominal_db``.

``--Xdb`` with a DOUBLE dash (``DJMINI3--10db-...``) is a separator artefact;
X is POSITIVE (spec appendix B, A4 #2).  It is not a negative number.

In the two-group dialect (``PARROT_90db_5750m_100m(2).iq``,
``FUTABA-T10J-daifei-60db_2440m_100m(1).iq``) there is a single ``db`` group and
it sits in the *same positional slot* as slot 2 above (immediately before the
centre frequency) -> it is the **rx gain**.  Confirmed constant at 60 dB for all
2.4 GHz RCs and 90 dB for the 5.8 GHz UAVs.

``SNR{n}dB`` (AVATA2 only) occupies the *ladder* slot, not the gain slot:
``DJI AVTA2-SNR0dB-85db_5765m_100m_10m(3).iq`` -> ladder=SNR0dB, gain=85 dB.
Per ERRATA candidate E-006 the nominal SNR label failed sanity (nominal "SNR2dB"
measured 29.9 dB), so ``snr_nominal_db`` is populated but must be treated as an
untrusted nominal, never as ground truth.

``-SNR{n}-`` (ZJU directory) has no dB unit at all: it is an ordinal 1/2/3
(verificacion-empirica-03: same source segment + independent AWGN per file).
``snr_nominal_db`` stays ``None`` there; ``snr_ordinal`` carries the 1/2/3.

Other tokens: ``(N)`` = take · ``_a-bs`` = slice window in seconds ·
``-lostN`` = lost-sample tag · ``（局部）`` = derived partial fragment ·
``packN`` = sub-capture index · ``daifei``/``dafei``(typo) = 待飞 idle-powered ·
``duima`` = 对码 pairing · ``xuanting`` = 悬停 hover · ``FLY``/``_Fly`` = in flight ·
``KONG`` = 空 background/empty channel · ``ceshi``/``cheshi`` = 测试 test artefact ·
``_high_RF0|_low_RF1`` = DroneRFa-style band/RF-chain index.

=============================================================================
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

__all__ = [
    "parse",
    "parse_all",
    "FIELDS",
    "DIALECTS",
    "DEVICE_ROLES",
    "SIGNAL_STATES",
]

# ---------------------------------------------------------------------------
# Literals written as escapes so the module is byte-safe under any locale on the
# Windows data server (cp936 consoles, PowerShell EncodedCommand, etc.).
# ---------------------------------------------------------------------------
PARTIAL_MARK = "（局部）"          # （局部）
ROOT_ZJU = "浙大被预处理为IQ的数据"  # 浙大被预处理为IQ的数据
ROOT_OVERSIZE = "过大的原始数据"                  # 过大的原始数据

DEVICE_ROLES = ("uav", "rc", "gs", "bg", "unk")
SIGNAL_STATES = ("flight", "pairing", "hover", "idle", "unknown")

DIALECTS = (
    "D1_pack_slice",
    "D2_uav_ladder",
    "D3_hyphen_2group",
    "D4_gain_underscore",
    "D5_zju_snr",
    "D6_sdr_default",
    "D7_dronerfa_rf",
    "D8_crop_slice",
    "D9_bare_model",
    "D10_archive",
    "D11_derived_cache",
)

FIELDS = (
    # provenance (verbatim, never normalised)
    "source_relpath",
    "source_file",
    "source_root",
    "class_dir",
    # identity
    "vendor_raw",
    "model_raw",
    "vendor",
    "model",
    "device_role",
    "signal_state",
    # radio
    "rx_gain_db",
    "snr_nominal_label",
    "snr_nominal_db",
    "snr_ordinal",
    "attenuation_label_raw",
    "attenuation_label_db",
    "center_freq_mhz",
    "sample_rate_msps",
    "vtsbw_mhz",
    "distance_m",
    # structure
    "take",
    "time_start_s",
    "time_end_s",
    "duration_s",
    "pack_index",
    "rf_chain",
    "band_tag",
    # flags
    "is_partial",
    "is_slice",
    "is_misfiled",
    "lost_samples_tag",
    "is_test_artifact",
    "variant_unresolved",
    "artifact_class",
    # bookkeeping
    "naming_dialect",
    "unparsed_residue",
    "vocab_hit",
    "ok",
    "reason",
)

# ---------------------------------------------------------------------------
# Controlled vocabulary.  key = model string uppercased with [ _-.] removed.
# value = (vendor, model, device_role, variant_unresolved)
# Corrections mandated by spec s2: AVTA2->avata2, DIMINI4PRO->mini4pro,
# DAUTEL->autel, dafei->daifei.
# ---------------------------------------------------------------------------
_V = collections.OrderedDict()


def _reg(keys, vendor, model, role, unresolved=False):
    for k in keys:
        _V[k] = (vendor, model, role, unresolved)


# --- UAVs -------------------------------------------------------------------
_reg(["DAUTELEVONANO", "AUTELEVONANO", "DAUTELEVONANO"], "autel", "evo_nano", "uav")
_reg(["DJIAVTA2", "DJIAVATA2", "DJAVTA2"], "dji", "avata2", "uav")
_reg(["AVATA"], "dji", "avata", "uav")
_reg(["DJFPVCOMBO", "DJICOMBO", "DJCOMBO", "DJIFPVCOMBO"], "dji", "fpv_combo", "uav")
_reg(["DJMAVIC3PRO", "DJIMAVIC3PRO"], "dji", "mavic3_pro", "uav")
_reg(["MAVIC3"], "dji", "mavic3", "uav")
_reg(["MAVICPRO"], "dji", "mavic_pro", "uav")
_reg(["DJMINI3", "DJIMINI3", "MINI3"], "dji", "mini3", "uav")
_reg(["MINI3PRO", "DJIMINI3PRO"], "dji", "mini3_pro", "uav")
_reg(["MINI2"], "dji", "mini2", "uav")
_reg(["DIMINI4PRO", "DJMINI4PRO", "DJIMINI4PRO"], "dji", "mini4_pro", "uav")
_reg(["AIR2S", "AIR2S"], "dji", "air2s", "uav", True)   # V03: dir says Air2R
_reg(["AIR2R"], "dji", "air2r", "uav", True)
_reg(["MAVIC2AIR2"], "dji", "mavic2_air2", "uav")
_reg(["INSPIRE2", "INSPIRE"], "dji", "inspire2", "uav")
_reg(["MATRICE100"], "dji", "matrice100", "uav")
_reg(["MATRICE200"], "dji", "matrice200", "uav")
_reg(["MATRICE300"], "dji", "matrice300", "uav")
_reg(["MATRICE30T"], "dji", "matrice30t", "uav")
_reg(["MATRICE600PRO"], "dji", "matrice600_pro", "uav")
_reg(["PHANTOM3"], "dji", "phantom3", "uav")
_reg(["PHANTOM4PRO"], "dji", "phantom4_pro", "uav")
_reg(["PHANTOM4PRORTK"], "dji", "phantom4_pro_rtk", "uav")
_reg(["DJI", "DJI"], "dji", None, "uav", True)          # ZJU 'DJi' bucket, model unknown
_reg(["PARROT"], "parrot", "parrot_unspecified", "uav", True)  # V05/F4: forensically ANAFI

# --- handheld RC / ground link ---------------------------------------------
_reg(["DEVENTIONDEVO10", "DEVENTIONDEVO"], "devention", "devo_10", "rc")
_reg(["FLYSKYEL18"], "flysky", "el18", "rc")
_reg(["FLYSKYFSI6X"], "flysky", "fs_i6x", "rc")
_reg(["FLYSKYNV14"], "flysky", "nv14", "rc")
_reg(["FRSKYX14"], "frsky", "x14", "rc")
_reg(["FRSKYX20R"], "frsky", "x20r", "rc")
_reg(["FRSKYX20"], "frsky", "x20", "rc")
_reg(["FRSKYX9DP2019"], "frsky", "x9dp2019", "rc")
_reg(["FUTABAT10J"], "futaba", "t10j", "rc")
_reg(["FUTABAT14SG", "FUTABAT14SGFLY"], "futaba", "t14sg", "rc")
_reg(["FUTABAT16IZ"], "futaba", "t16iz", "rc")
_reg(["FUTABAT18SZ"], "futaba", "t18sz", "rc")
_reg(["FUTABAT61ZFLY", "FUTABAT61Z"], "futaba", "t61z", "rc", True)  # suspected typo of T16IZ
_reg(["HERELINKHX4"], "herelink", "hx4", "rc")
_reg(["JRPROPOXG14"], "jrpropo", "xg14", "rc")
_reg(["JRPROPOXG7"], "jrpropo", "xg7", "rc")
_reg(["JUMPERT14"], "jumper", "t14", "rc")
_reg(["JUMPERTPROV2"], "jumper", "tpro_v2", "rc")
_reg(["RADIOLINKAT10II"], "radiolink", "at10_ii", "rc")
_reg(["RADIOLINKAT9SPRO"], "radiolink", "at9s_pro", "rc")
_reg(["RADIOLINKAT9SFLY", "RADIOLINKAT9S"], "radiolink", "at9s", "rc", True)
_reg(["RADIOMASTERBOXER"], "radiomaster", "boxer", "rc")
_reg(["RADIOMASTERTX16S"], "radiomaster", "tx16s", "rc")
_reg(["SIYIFT24"], "siyi", "ft24", "rc")
_reg(["SIYIMK15"], "siyi", "mk15", "rc")
_reg(["SIYIMK32"], "siyi", "mk32", "rc")
_reg(["SKYDROIDH12"], "skydroid", "h12", "rc")
_reg(["SKYDROIDT10"], "skydroid", "t10", "rc")
_reg(["TARANISPLUS", "TARAINSPLUSFLY", "TARAINSPLUS", "TARANISPLUSFLY"],
     "frsky", "taranis_plus", "rc")
_reg(["VBARFLY", "VBAR"], "vbar", "vbar_control", "rc", True)
_reg(["WFLYET10"], "wfly", "et10", "rc")
_reg(["WFLYET16S"], "wfly", "et16s", "rc")
_reg(["WFLYWFT09SII"], "wfly", "wft09sii", "rc")
_reg(["YUNZHUOH12"], "yunzhuo", "h12", "rc")
_reg(["YUNZHUOH16"], "yunzhuo", "h16", "rc")
_reg(["YUNZHUOH30"], "yunzhuo", "h30", "rc")
_reg(["YUNZHUOT12FLY", "YUNZHUOT12"], "yunzhuo", "t12", "rc")

# --- background / probe -----------------------------------------------------
_reg(["KONG"], None, "kong", "bg")
_reg(["WEIZHI"], "weizhi", None, "unk", True)   # F3: 940 MHz probe -> quarantine

# spec s2 also merges SKYDROID == 云卓(YUNZHUO) into vendor=skydroid.  We do NOT
# apply that here because fact F1 established SKYDROID H12 and YUNZHUO H12 are two
# distinct physical devices; merging the vendor namespace would collide their
# taxonomy_id.  Exposed for the taxonomy step to decide:
VENDOR_MERGE_SPEC_S2 = {"yunzhuo": "skydroid"}

_STATE_MAP = {
    "daifei": "idle",
    "dafei": "idle",     # typo of daifei (spec s2)
    "duima": "pairing",
    "xuanting": "hover",
    "fly": "flight",
}

_TEST_ROOTS = frozenset([
    "exp3", ".pt asset", "RFUAV_pt",
    "augmentation_exp1_MethodSelect", "augmentation_exp2_allDrone",
    "pics_exp1_alldrones",
])

# ---------------------------------------------------------------------------
# Regexes.  All are applied to the *stem* (basename minus extension).
# ---------------------------------------------------------------------------
RE_PACK = re.compile(r"^pack(?P<pack>\d+)_(?P<t0>\d+)-(?P<t1>\d+)s$")

# D2: {model}[-|--]{ladder}db|SNR{n}dB-{gain}db_{cf}m_{sr}m_{bw}m{tail}
RE_UAV3 = re.compile(
    r"^(?P<model>.+?)"
    r"(?P<dash>-{1,2})"
    r"(?P<ladder>SNR-?\d+dB|\d+db)"
    r"-(?P<gain>\d+)db"
    r"_(?P<cf>\d+)m_(?P<sr>\d+)m_(?P<bw>\d+)m"
    r"(?P<tail>.*)$"
)

# D3: {model}[-|_]{state}?[-|_]{gain}db_{cf}m_{sr}[m]{tail}
RE_RC2 = re.compile(
    r"^(?P<model>.+?)"
    r"(?:[-_](?P<state>daifei|dafei|duima|xuanting))?"
    r"[-_](?P<gain>\d+)db"
    r"_(?P<cf>\d+)m_(?P<sr>\d+)(?P<srm>m?)"
    r"(?P<tail>.*)$"
)

# D4: {model}_{cf}_{state}_[gain]{G}dB{tail}
RE_GAIN = re.compile(
    r"^(?P<model>.+?)_(?P<cf>\d{3,4})_(?P<state>daifei|dafei|duima|xuanting)_"
    r"(?:gain)?(?P<gain>\d+)dB(?P<tail>.*)$"
)

# D5: {model}-SNR{n}-{cf}m
RE_ZJU = re.compile(r"^(?P<model>.+?)-SNR(?P<n>\d+)-(?P<cf>\d+)m$")

# D6: SDR / GNU Radio default sink name
RE_SDR = re.compile(
    r"^data\.bin_(?P<sps>\d+)SPS_(?P<hz>-?\d+)Hz_"
    r"(?P<y>\d{4})_(?P<mo>\d{2})_(?P<d>\d{2})_T(?P<hh>\d{2})-(?P<mm>\d{2})-(?P<ss>\d{2})$"
)

# D7: DroneRFa-style {Model}_{high|low}_RF{n}
RE_RFA = re.compile(r"^(?P<model>.+?)_(?P<band>high|low)_RF(?P<rf>\d+)$", re.IGNORECASE)

# D9: bare model, optional slice window or bare duration
RE_BARE = re.compile(
    r"^(?P<model>[A-Za-z][A-Za-z0-9_. -]*?)"
    r"(?:_(?P<t0>\d+)-(?P<t1>\d+)s|_(?P<dur>\d+(?:\.\d+)?)s)?$"
)

# tail sub-tokens
RE_TAIL_LOST = re.compile(r"-lost(\d+)")
RE_TAIL_TAKE = re.compile(r"\((\d+)\)")
RE_TAIL_SLICE = re.compile(r"_(\d+)-(\d+)s")


def _norm_key(s):
    """Uppercase and strip separators so 'DJI AVTA2' == 'DJIAVTA2'."""
    if s is None:
        return ""
    return re.sub(r"[ _\-.]", "", s).upper()


def _blank():
    d = collections.OrderedDict()
    for f in FIELDS:
        d[f] = None
    d["is_partial"] = False
    d["is_slice"] = False
    d["is_misfiled"] = False
    d["is_test_artifact"] = False
    d["variant_unresolved"] = False
    d["device_role"] = "unk"
    d["signal_state"] = "unknown"
    d["ok"] = False
    return d


def _split_path(relpath):
    """Split a v1 relpath on either separator; returns list of components."""
    return [p for p in re.split(r"[\\/]+", relpath) if p != ""]


def _class_dir_for(parts):
    """Per-root rules (explicit, not heuristic) for locating the class directory."""
    root = parts[0]
    if len(parts) < 3:
        return None
    if root == "PublicDataset":
        return None if parts[1] == "Dataset" else parts[1]
    if root == ROOT_ZJU:
        return parts[2] if (len(parts) > 3 and parts[1] == "iq data") else None
    if root == "exp3":
        return parts[2] if len(parts) > 3 else None
    if root in ("rawdata", ROOT_OVERSIZE, "crop_data", ".dat_pic"):
        return parts[1]
    return None


def _artifact_class_for(relpath, parts, ext):
    root = parts[0]
    if ext in (".rar", ".zip"):
        if root == "PublicDataset":
            if len(parts) > 2 and parts[2] == "RAWDATA":
                return "archive_rawdata"
            return "archive_experiment"
        return "archive_experiment"
    if ext == ".npy":
        return "derived_cache"
    if root in _TEST_ROOTS:
        return "experiment_sandbox"
    if root == "crop_data":
        return "derived_slice"
    return "recording"


def _is_test_artifact(relpath, parts, class_dir):
    root = parts[0]
    if root in _TEST_ROOTS:
        return True
    low = relpath.lower()
    if "ceshi" in low or "cheshi" in low:
        return True
    if class_dir and re.search(r"(ce|che)shi$", class_dir, re.IGNORECASE):
        return True
    if root == "PublicDataset" and len(parts) > 2 and parts[2] in (
        "IMAGESET-exp", "ValidationSet_5Drones"
    ):
        return True
    return False


def _resolve_identity(out, model_raw):
    """Fill vendor/model/device_role from the controlled vocabulary."""
    out["model_raw"] = model_raw
    key = _norm_key(model_raw)
    hit = _V.get(key)
    if hit is None:
        # try trimming a trailing '_FLY' / '-FLY' marker (ZJU flight tag)
        k2 = re.sub(r"FLY$", "", key)
        hit = _V.get(k2)
    if hit is None:
        out["device_role"] = "unk"
        out["variant_unresolved"] = True
        out["vocab_hit"] = False
        return False
    out["vocab_hit"] = True
    vendor, model, role, unresolved = hit
    out["vendor"] = vendor
    out["model"] = model
    out["device_role"] = role
    if unresolved:
        out["variant_unresolved"] = True
    # vendor_raw = the literal prefix of model_raw that produced the vendor,
    # only when it really is a prefix of the raw string (never fabricated).
    if vendor:
        vk = vendor.upper()
        rk = _norm_key(model_raw)
        if rk.startswith(vk):
            n = 0
            seen = 0
            for ch in model_raw:
                n += 1
                if re.match(r"[ _\-.]", ch):
                    continue
                seen += 1
                if seen == len(vk):
                    break
            out["vendor_raw"] = model_raw[:n]
    return True


def _consume_tail(out, tail):
    """Pull （局部） / -lostN / (N) / _a-bs out of *tail*; leftovers -> residue."""
    if not tail:
        out["unparsed_residue"] = None
        return
    rest = tail
    if PARTIAL_MARK in rest:
        out["is_partial"] = True
        rest = rest.replace(PARTIAL_MARK, "")
    m = RE_TAIL_LOST.search(rest)
    if m:
        out["lost_samples_tag"] = "lost" + m.group(1)
        rest = rest[: m.start()] + rest[m.end():]
    m = RE_TAIL_TAKE.search(rest)
    if m:
        out["take"] = int(m.group(1))
        rest = rest[: m.start()] + rest[m.end():]
    m = RE_TAIL_SLICE.search(rest)
    if m:
        out["time_start_s"] = float(m.group(1))
        out["time_end_s"] = float(m.group(2))
        out["duration_s"] = float(m.group(2)) - float(m.group(1))
        out["is_slice"] = True
        rest = rest[: m.start()] + rest[m.end():]
    rest = rest.strip()
    out["unparsed_residue"] = rest if rest else None


def _apply_state(out, tok):
    if tok is None:
        return
    out["signal_state"] = _STATE_MAP.get(tok.lower(), "unknown")


def _vtsbw_from_dirs(parts):
    for p in parts:
        m = re.match(r"^VTSBW=(\d+)$", p)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def parse(relpath):
    """Parse one v1 relative path into normalised fields.

    Returns an ``OrderedDict`` with the keys in :data:`FIELDS`.  ``ok`` is True
    when a dialect matched.  A matched file may still carry a non-empty
    ``unparsed_residue`` -- that means a token was seen but could NOT be
    explained, and it is deliberately surfaced rather than dropped.
    """
    out = _blank()
    out["source_relpath"] = relpath
    parts = _split_path(relpath)
    if not parts:
        out["reason"] = "empty path"
        out["naming_dialect"] = "UNKNOWN"
        return out
    fname = parts[-1]
    out["source_file"] = fname
    out["source_root"] = parts[0]
    stem, ext = os.path.splitext(fname)
    ext = ext.lower()
    class_dir = _class_dir_for(parts)
    out["class_dir"] = class_dir
    out["artifact_class"] = _artifact_class_for(relpath, parts, ext)
    out["is_test_artifact"] = _is_test_artifact(relpath, parts, class_dir)
    out["vtsbw_mhz"] = _vtsbw_from_dirs(parts)

    # ---- D10 / D11: containers and caches --------------------------------
    if ext in (".rar", ".zip"):
        out["naming_dialect"] = "D10_archive"
        out["ok"] = True
        if class_dir:
            _resolve_identity(out, class_dir)
        elif parts[0] == "PublicDataset" and len(parts) > 2 and parts[2] == "RAWDATA":
            _resolve_identity(out, stem)
        elif parts[0] == "PublicDataset" and len(parts) > 2 and \
                parts[2] == "ValidationSet_5Drones":
            _resolve_identity(out, stem)
        out["reason"] = "container archive; not a recording"
        return out
    if ext == ".npy":
        out["naming_dialect"] = "D11_derived_cache"
        out["ok"] = True
        out["reason"] = "derived training cache; not a recording"
        return out

    # ---- D8: crop_data derived slice (.dat) ------------------------------
    if parts[0] == "crop_data" and ext == ".dat":
        m = RE_TAIL_SLICE.search(stem)
        parent_stem = stem
        t0 = t1 = None
        if m and m.end() == len(stem):
            parent_stem = stem[: m.start()]
            t0, t1 = float(m.group(1)), float(m.group(2))
        sub = parse(parts[0] + "\\" + (class_dir or "") + "\\" + parent_stem + ".iq")
        for k in FIELDS:
            if k in ("source_relpath", "source_file", "source_root", "class_dir",
                     "naming_dialect", "artifact_class", "is_test_artifact", "ok",
                     "reason", "time_start_s", "time_end_s", "duration_s",
                     "is_slice", "unparsed_residue"):
                continue
            out[k] = sub[k]
        out["unparsed_residue"] = sub["unparsed_residue"]
        out["time_start_s"], out["time_end_s"] = t0, t1
        if t0 is not None:
            out["duration_s"] = t1 - t0
            out["is_slice"] = True
        out["naming_dialect"] = "D8_crop_slice"
        out["ok"] = sub["ok"]
        if not sub["ok"]:
            out["reason"] = "crop parent stem unparsed: " + str(sub["reason"])
        return out

    # ---- D7: DroneRFa-style .dat -----------------------------------------
    if ext == ".dat":
        m = RE_RFA.match(stem)
        if m:
            out["naming_dialect"] = "D7_dronerfa_rf"
            out["band_tag"] = m.group("band").lower()
            out["rf_chain"] = int(m.group("rf"))
            _resolve_identity(out, m.group("model"))
            if out["model"] is None and class_dir:
                _resolve_identity(out, class_dir)
            out["unparsed_residue"] = None
            out["ok"] = True
            return out
        out["naming_dialect"] = "UNKNOWN"
        out["reason"] = "unrecognised .dat filename shape"
        return out

    # ---- D6: SDR default sink name ---------------------------------------
    m = RE_SDR.match(stem)
    if m:
        out["naming_dialect"] = "D6_sdr_default"
        out["sample_rate_msps"] = float(m.group("sps")) / 1e6
        hz = int(m.group("hz"))
        if hz == -2147483648:
            # INT32_MIN sentinel -> the SDR frontend never had a frequency set.
            out["center_freq_mhz"] = None
            out["unparsed_residue"] = "-2147483648Hz(INT32_MIN sentinel, cf unset)"
        else:
            out["center_freq_mhz"] = hz / 1e6
        if class_dir:
            _resolve_identity(out, class_dir)
        out["ok"] = True
        return out

    # ---- D1: PublicDataset pack slice ------------------------------------
    m = RE_PACK.match(stem)
    if m:
        out["naming_dialect"] = "D1_pack_slice"
        out["pack_index"] = int(m.group("pack"))
        out["time_start_s"] = float(m.group("t0"))
        out["time_end_s"] = float(m.group("t1"))
        out["duration_s"] = out["time_end_s"] - out["time_start_s"]
        out["is_slice"] = True
        if class_dir:
            _resolve_identity(out, class_dir)
        out["unparsed_residue"] = None
        out["ok"] = True
        return out

    # ---- D5: ZJU AWGN ladder ---------------------------------------------
    m = RE_ZJU.match(stem)
    if m:
        out["naming_dialect"] = "D5_zju_snr"
        out["snr_nominal_label"] = "SNR" + m.group("n")
        out["snr_ordinal"] = int(m.group("n"))
        # verificacion-empirica-03: the ladder carries no dB unit -> no dB value.
        out["snr_nominal_db"] = None
        out["center_freq_mhz"] = float(m.group("cf"))
        _resolve_identity(out, m.group("model"))
        if out["model"] is None and class_dir:
            _resolve_identity(out, class_dir)
        if re.search(r"(?:^|[_-])(FLY|Fly)$", m.group("model")):
            out["signal_state"] = "flight"
        out["unparsed_residue"] = None
        out["ok"] = True
        return out

    # ---- D2: three-group UAV ladder --------------------------------------
    m = RE_UAV3.match(stem)
    if m:
        out["naming_dialect"] = "D2_uav_ladder"
        ladder = m.group("ladder")
        out["snr_nominal_label"] = ladder
        ms = re.match(r"^SNR(-?\d+)dB$", ladder)
        if ms:
            # explicit dB claim (AVATA2 only).  E-006: nominal, NOT trustworthy.
            out["snr_nominal_db"] = float(ms.group(1))
        else:
            out["attenuation_label_raw"] = ladder
            out["attenuation_label_db"] = float(re.sub(r"db$", "", ladder))
        out["rx_gain_db"] = float(m.group("gain"))
        out["center_freq_mhz"] = float(m.group("cf"))
        out["sample_rate_msps"] = float(m.group("sr"))
        out["vtsbw_mhz"] = float(m.group("bw"))
        _resolve_identity(out, m.group("model"))
        if class_dir and _norm_key(class_dir) != _norm_key(m.group("model")) \
                and out["device_role"] == "bg":
            out["is_misfiled"] = True     # e.g. rawdata\DJMINI3\KONG-*.iq (E-004)
        _consume_tail(out, m.group("tail"))
        out["ok"] = True
        return out

    # ---- D4: underscore gain dialect --------------------------------------
    m = RE_GAIN.match(stem)
    if m:
        out["naming_dialect"] = "D4_gain_underscore"
        out["center_freq_mhz"] = float(m.group("cf"))
        out["rx_gain_db"] = float(m.group("gain"))
        _apply_state(out, m.group("state"))
        _resolve_identity(out, m.group("model"))
        _consume_tail(out, m.group("tail"))
        out["ok"] = True
        return out

    # ---- D3: two-group hyphen dialect -------------------------------------
    m = RE_RC2.match(stem)
    if m:
        out["naming_dialect"] = "D3_hyphen_2group"
        out["rx_gain_db"] = float(m.group("gain"))
        out["center_freq_mhz"] = float(m.group("cf"))
        out["sample_rate_msps"] = float(m.group("sr"))
        _apply_state(out, m.group("state"))
        _resolve_identity(out, m.group("model"))
        if class_dir and out["device_role"] == "bg" and \
                _norm_key(class_dir) != _norm_key(m.group("model")):
            out["is_misfiled"] = True     # rawdata\DJMINI3\KONG-60db_... (E-004)
        _consume_tail(out, m.group("tail"))
        out["ok"] = True
        return out

    # ---- D9: bare model name (exp3 / *_pt sandboxes) ----------------------
    m = RE_BARE.match(stem)
    if m:
        out["naming_dialect"] = "D9_bare_model"
        if m.group("t0") is not None:
            out["time_start_s"] = float(m.group("t0"))
            out["time_end_s"] = float(m.group("t1"))
            out["duration_s"] = out["time_end_s"] - out["time_start_s"]
            out["is_slice"] = True
        elif m.group("dur") is not None:
            out["duration_s"] = float(m.group("dur"))
            out["is_slice"] = True
        _resolve_identity(out, m.group("model"))
        out["unparsed_residue"] = None
        out["ok"] = True
        return out

    out["naming_dialect"] = "UNKNOWN"
    out["reason"] = "no dialect matched"
    return out


def parse_all(rows):
    """Parse an iterable of relpaths (str) or dicts carrying a 'relpath' key.

    Returns ``(parsed, unparsed)``.  ``parsed`` holds every row where a dialect
    matched; ``unparsed`` holds the rest, each with a ``reason``.
    """
    parsed = []
    unparsed = []
    for row in rows:
        if isinstance(row, dict):
            relpath = row.get("relpath") or row.get("source_relpath") or ""
            extra = row
        else:
            relpath = row
            extra = None
        rec = parse(relpath)
        if extra is not None:
            for k, v in extra.items():
                if k not in rec:
                    rec[k] = v
        if rec["ok"]:
            parsed.append(rec)
        else:
            unparsed.append(rec)
    return parsed, unparsed


# ---------------------------------------------------------------------------
# CLI: coverage report generator
# ---------------------------------------------------------------------------
def _fmt_pct(n, d):
    return "0.00%" if d == 0 else "%.2f%%" % (100.0 * n / d)


def build_report(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    parsed, unparsed = parse_all(rows)
    total = len(rows)
    by_dialect = collections.Counter(r["naming_dialect"] for r in parsed)
    residue = [r for r in parsed if r["unparsed_residue"]]
    # A true vocabulary miss = a model token WAS present in the filename but no
    # controlled-vocabulary entry matched it (vocab_hit is False).  Entries that
    # are registered on purpose with model=None (ZJU 'DJi' bucket, WEIZHI probe)
    # are NOT misses - they are deliberate nulls, counted separately.
    unknown_models = collections.Counter(
        r["model_raw"] or "(none)" for r in parsed if r["vocab_hit"] is False
    )
    deliberate_null = collections.Counter(
        r["model_raw"] for r in parsed
        if r["vocab_hit"] is True and r["model"] is None
    )
    no_model_token = collections.Counter(
        r["naming_dialect"] for r in parsed if r["vocab_hit"] is None
    )
    by_role = collections.Counter(r["device_role"] for r in parsed)
    by_state = collections.Counter(r["signal_state"] for r in parsed)

    L = []
    ap = L.append
    ap("# RFUAV v1 filename dialect parser - coverage report")
    ap("")
    ap("Generated by `tools/rfuav_v2/dialect.py` over **all %d rows** of "
       "`memoria/datos/e1/e1_binfiles.csv`." % total)
    ap("Measured, not estimated: rerun with "
       "`python3 tools/rfuav_v2/dialect.py --csv memoria/datos/e1/e1_binfiles.csv`.")
    ap("")
    ap("## 1. Headline coverage")
    ap("")
    ap("| metric | count | share |")
    ap("|---|---:|---:|")
    ap("| rows in e1_binfiles.csv | %d | 100.00%% |" % total)
    ap("| dialect matched (`ok=True`) | %d | %s |" % (len(parsed), _fmt_pct(len(parsed), total)))
    ap("| of which fully explained (no residue) | %d | %s |"
       % (len(parsed) - len(residue), _fmt_pct(len(parsed) - len(residue), total)))
    ap("| of which carry an unexplained token (`unparsed_residue`) | %d | %s |"
       % (len(residue), _fmt_pct(len(residue), total)))
    ap("| no dialect matched (`ok=False`) | %d | %s |"
       % (len(unparsed), _fmt_pct(len(unparsed), total)))
    ap("")
    ap("## 2. Dialect table")
    ap("")
    ap("| dialect | regex sketch | count | share |")
    ap("|---|---|---:|---:|")
    sketch = {
        "D1_pack_slice": "`^pack(\\d+)_(\\d+)-(\\d+)s$` (+ `VTSBW=N` dir)",
        "D2_uav_ladder": "`^(model)-{1,2}(SNR\\d+dB|\\d+db)-(\\d+)db_(\\d+)m_(\\d+)m_(\\d+)m(tail)$`",
        "D3_hyphen_2group": "`^(model)[-_](state)?[-_](\\d+)db_(\\d+)m_(\\d+)m?(tail)$`",
        "D4_gain_underscore": "`^(model)_(\\d{3,4})_(state)_(gain)?(\\d+)dB(tail)$`",
        "D5_zju_snr": "`^(model)-SNR(\\d)-(\\d+)m$`",
        "D6_sdr_default": "`^data\\.bin_(\\d+)SPS_(-?\\d+)Hz_Y_M_D_THH-MM-SS$`",
        "D7_dronerfa_rf": "`^(model)_(high|low)_RF(\\d)$`",
        "D8_crop_slice": "`crop_data\\CLASS\\PARENT\\PARENT_(\\d+)-(\\d+)s.dat` (parent re-parsed as D2)",
        "D9_bare_model": "`^(model)(_(\\d+)-(\\d+)s|_(\\d+\\.?\\d*)s)?$`",
        "D10_archive": "`*.rar` / `*.zip` container",
        "D11_derived_cache": "`*.npy` training cache",
    }
    for d in DIALECTS:
        n = by_dialect.get(d, 0)
        if n:
            ap("| `%s` | %s | %d | %s |" % (d, sketch.get(d, ""), n, _fmt_pct(n, total)))
    ap("| **total** | | **%d** | **%s** |" % (len(parsed), _fmt_pct(len(parsed), total)))
    ap("")
    ap("## 3. Rows with an unexplained numeric group (`unparsed_residue`)")
    ap("")
    if not residue:
        ap("None.")
    else:
        ap("| relpath | dialect | residue |")
        ap("|---|---|---|")
        for r in sorted(residue, key=lambda x: x["source_relpath"]):
            ap("| `%s` | %s | `%s` |"
               % (r["source_relpath"], r["naming_dialect"], r["unparsed_residue"]))
    ap("")
    ap("## 4. Rows where NO dialect matched")
    ap("")
    if not unparsed:
        ap("None - every row in the scan matched a dialect.")
    else:
        ap("| relpath | reason |")
        ap("|---|---|")
        for r in sorted(unparsed, key=lambda x: x["source_relpath"]):
            ap("| `%s` | %s |" % (r["source_relpath"], r["reason"]))
    ap("")
    ap("## 5. Controlled-vocabulary accounting")
    ap("")
    ap("### 5a. True misses (a model token was present, no vocab entry matched)")
    ap("")
    if not unknown_models:
        ap("None - every model token in the scan resolved against the controlled "
           "vocabulary.")
    else:
        ap("| model_raw | count |")
        ap("|---|---:|")
        for k, v in unknown_models.most_common():
            ap("| `%s` | %d |" % (k, v))
    ap("")
    ap("### 5b. Deliberate nulls (vocab entry exists but `model` is knowingly None)")
    ap("")
    if not deliberate_null:
        ap("None.")
    else:
        ap("| model_raw | count | why |")
        ap("|---|---:|---|")
        _why = {
            "DJi": "ZJU catch-all bucket - V03 says these 9 files cannot be "
                   "attributed to a model; vendor=dji only",
            "Dji": "same ZJU catch-all bucket, .dat_pic spelling",
            "DJI": "same ZJU catch-all bucket",
            "WEIZHI": "F3 940 MHz probe - vendor only, model undetermined, "
                      "destined for `_quarantine\\probe\\`",
            "KONG": "background class, no device",
        }
        for k, v in deliberate_null.most_common():
            ap("| `%s` | %d | %s |" % (k, v, _why.get(k, "see vocabulary table")))
    ap("")
    ap("### 5c. Rows that legitimately carry no model token at all")
    ap("")
    if not no_model_token:
        ap("None.")
    else:
        ap("| naming_dialect | count |")
        ap("|---|---:|")
        for k, v in no_model_token.most_common():
            ap("| %s | %d |" % (k, v))
        ap("")
        ap("These are the experiment archives that sit above any class directory "
           "(`IMAGESET-exp`, `augmentation_exp*`) and the derived `.npy` cache - "
           "there is no device identity to recover, by construction.  Note that "
           "`D1_pack_slice` does NOT appear here: pack filenames carry no model "
           "token either, but their identity is recovered from `class_dir`.")
    ap("")
    ap("## 6. Derived field distributions")
    ap("")
    ap("device_role: " + ", ".join("%s=%d" % kv for kv in sorted(by_role.items())))
    ap("")
    ap("signal_state: " + ", ".join("%s=%d" % kv for kv in sorted(by_state.items())))
    ap("")
    nz = sum(1 for r in parsed if r["distance_m"] is not None)
    ap("`distance_m` populated on %d/%d rows - **no RFUAV v1 filename encodes a "
       "distance**; the three `_Nm` groups are centre frequency / sample rate / "
       "VTSBW (see module docstring for the byte-arithmetic proof)." % (nz, len(parsed)))
    ap("")
    return "\n".join(L) + "\n", parsed, unparsed


def main(argv=None):
    ap = argparse.ArgumentParser(description="RFUAV v1 filename dialect parser")
    ap.add_argument("--csv", default=os.path.join("memoria", "datos", "e1",
                                                  "e1_binfiles.csv"))
    ap.add_argument("--report", default=None, help="write markdown report here")
    ap.add_argument("--jsonl", default=None, help="write parsed rows as JSONL here")
    ap.add_argument("--path", default=None, help="parse a single relpath and exit")
    args = ap.parse_args(argv)

    if args.path:
        json.dump(parse(args.path), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    text, parsed, unparsed = build_report(args.csv)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as fh:
            for r in parsed:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
