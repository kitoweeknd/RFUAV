# -*- coding: utf-8 -*-
"""
build_manifests.py -- E1 stage 3: turn the read-only inventory + hash pass into
the RFUAV v2 manifest layer.

    e1_inventory.csv  (path, size, mtime, ...)          <- E1 stage 1
    e1_hashes.csv     (path, size, sha512)  HEADERLESS   <- E1 stage 2
        |
        v
    recordings.csv        one row per signal file, fully typed
    sessions.csv          the leakage grouping key (session_id)
    dedup_wholefile.csv   byte-identical groups found by SHA-512
    dedup_substring.csv   slices that are byte SUBSTRINGS of another file
    leakage_groups.csv    the split-safe grouping any CV must respect
    rate_evidence.csv     how each sample_rate was decided, per file
    anomalies.csv         every errata-relevant finding, machine-readable

WRITE SCOPE: only under manifests_dir (default E:\\Drone_dataset\\RFUAV-v2\\manifests).
Reads originals only when --verify-bytes is passed, and then read-only, 64 KB at
a time.  Never writes, moves, renames or deletes anything under the source roots.
Rollback = delete the output directory.

------------------------------------------------------------------------------
WHY THIS FILE IS NOT A THIN CSV JOIN
------------------------------------------------------------------------------
Four measured properties of this library (2026-07-31, by reading bytes, see
spec section 6 ERRATA) make the naive join wrong:

E-009b  ``bytes_per_sample`` is NOT a constant.  cf32_le (8 B/sample) for
        rawdata / crop_data / PublicDataset / 过大的原始数据 (6/6 probed);
        ci16_le (4 B/sample) for .dat_pic\\Mavic2Air2\\ and
        浙大被预处理为IQ的数据\\ (81 files).  Hardcoding /8 mislabels 82 files.

E-009   ``sample_rate`` is NOT 100 MS/s library-wide.  Measured 200 MS/s (28
        files), 15, 5 and 75 MS/s.  Only 29 % of filenames carry a rate.

E-008   the ``_A-Bs`` filename window is the REQUESTED window, not the
        delivered one.  25/27 short files are byte-proven valid TAILS of an
        identified source; the payload is intact and the rate is a round
        100/200 MS/s.  So duration must be derived from bytes, and the
        filename window kept only as ``requested_window_s``.

E-011   the dangerous duplication is SUBSTRING duplication, invisible to
        whole-file SHA-512: rawdata\\SIYI_MK32\\..._34-36s.iq is byte-identical
        to 过大的原始数据\\SIYI_MK32\\...gain80dB.iq at offset 27_200_000_000.
        A hash-only dedup puts the same bytes in train and test.

So: dtype is a field, rate is inferred with recorded evidence, duration comes
from bytes, and dedup runs at two levels (whole-file hash + structural stem
grouping with arithmetic tail search).
"""

from __future__ import unicode_literals

import argparse
import codecs
import collections
import csv
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dialect  # noqa: E402
import iq_dtype  # noqa: E402

# The tiering rule and the foreign-root list are imported, never copied.  A
# second definition here is exactly the drift `test_labels.py
# ::test_tier_matches_the_splitter_rule` exists to stop: labels.json, the
# splitter and this manifest would each publish their own "eval_tier" and
# nobody would notice they had diverged.  So there is deliberately NO fallback
# definition below -- a missing module dies loudly instead of silently
# producing a manifest whose tiers disagree with the split that consumes it.
try:
    import split_grouped  # noqa: E402
except ImportError as exc:                                  # pragma: no cover
    raise SystemExit(
        "FATAL: cannot import split_grouped (%s).\n"
        "  expected at: %s\n"
        "  n_sessions_in_class / eval_tier / unique_seconds come from that\n"
        "  module's single definition.  Sync the file; do not re-implement it."
        % (exc, os.path.join(_HERE, "split_grouped.py")))

# ---------------------------------------------------------------------------
# Constants measured from the library, not guessed.
# ---------------------------------------------------------------------------

#: dtype per source path prefix.  Determined by reading bytes and measuring the
#: finite fraction when interpreted as float32: genuine cf32_le reads 100.00 %
#: finite with rms 7e-4..4e-1; ci16_le misread as float32 reads 91-99 % finite
#: with rms ~1e37 and min ~ -3.4e38 (near FLT_MAX).  See ERRATA E-009b.
#: SigMF spells interleaved int16 ``ci16_le``; ``sc16_le`` is the SDR-vendor
#: spelling of the same layout.
#:
#: The third field is the CONFIDENCE, and it is not decoration.  Only the
#: prefixes marked "measured" were actually probed.  Unmeasured prefixes inherit
#: a guess AND raise a probe request in anomalies.csv; they must not reach a
#: sidecar unprobed.  This matters because a wrong ``bytes_per_sample`` does not
#: stay local: it propagates into ``num_samples``, ``delivered_duration_s`` and
#: the size/bps/duration rate derivation, which is exactly the cascade that
#: produced the refuted E-010 claim ("the library is complex64" was assumed
#: rather than tested, and a perfectly aligned int16 file got written up as a
#: misaligned float32 one).
#:
#: ``.dat_pic`` is now measured across all 12 subdirectories, not extrapolated
#: from the single Mavic2Air2 probe.  The float32 finite-fraction screen alone
#: only proves NOT-cf32, so the ci16_le call rests on an alignment test, which
#: no amplitude heuristic can fake: an oversampled IQ stream read at the right
#: element width is band-limited and strongly correlated sample-to-sample, and
#: reading it one byte off destroys that.  Measured per file, lag-1 |autocorr| /
#: spectral concentration: ``i16@0`` = 0.43..0.99 / 0.55..1.00, while ``i16@1``
#: collapses to 0.001..0.004 / 0.33 (the noise floor), ``i8`` and ``f32``
#: likewise collapse.  Two files (AVATA, Taranis_Plus) tie between offset 0 and
#: offset 2 to within 0.002 -- expected and NOT a 2-byte header, because a
#: 2-byte shift merely swaps which channel is called I and which Q, leaving the
#: lag-1 magnitude near-degenerate; spectral concentration prefers offset 0 in
#: 11/11.  Peak |v| lands at 11.9-14.8 bits, i.e. a 12/14-bit converter in a
#: 16-bit word, as expected of this front end.
#: Longest matching prefix wins.
DTYPE_RULES = (
    # (path prefix, datatype, confidence)
    ("rawdata",                       "cf32_le", "measured"),
    ("crop_data",                     "cf32_le", "measured"),
    ("PublicDataset",                 "cf32_le", "measured"),
    (dialect.ROOT_OVERSIZE,           "cf32_le", "measured"),
    (dialect.ROOT_ZJU,                "ci16_le", "measured"),
    (".dat_pic",                      "ci16_le", "measured"),
    # Experiment working copies and the packaged .pt assets: probed, 100.000 %
    # finite as float32 with rms 2.0e-3..1.1e-2 and min -3.3e-2, i.e. unit-scale
    # baseband floats with no NaN/inf at all -- unambiguous cf32_le.
    ("exp3",                          "cf32_le", "measured"),
    ("RFUAV_pt",                      "cf32_le", "measured"),
    (".pt asset",                     "cf32_le", "measured"),
)
DTYPE_DEFAULT = ("cf32_le", "ASSUMED_UNPROBED")

#: Re-exported, not redeclared.  This manifest is the PRODUCER of the datatype
#: tokens; iq_dtype.py is the vocabulary every consumer reads.  Two dicts that
#: agree today are two dicts that disagree after the next edit, and that exact
#: divergence -- producer knew ci16_le, no consumer did -- is defect D9.
#: `test_iq_dtype.py::test_producer_and_consumer_share_one_table` asserts
#: identity (`assertIs`), so a copy here fails the suite rather than drifting.
BYTES_PER_SAMPLE = iq_dtype.BYTES_PER_SAMPLE

#: Rates actually observed or plausible for a USRP X310 in this library, MS/s.
#: Order matters only for tie-breaking: prefer the larger explanation last.
STD_RATES_MSPS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 61.44,
                  75.0, 100.0, 120.0, 125.0, 200.0)

#: Relative tolerance when testing "does size/bps/duration land on a std rate".
RATE_REL_TOL = 1e-9

#: A delivered duration shorter than the requested window by more than this
#: many seconds counts as a truncated delivery (E-008).
TRUNC_EPS_S = 1e-6

SIG_EXT = (".iq", ".dat", ".bin", ".mat")
ARCHIVE_EXT = (".rar", ".zip", ".7z", ".tar", ".gz")

#: Directory suffixes that denote a DIFFERENT capture condition.  From
#: taxonomy/class_aliases.json _policy: "A directory alias is NOT automatically
#: a session merge.  Directories carrying a state or intent suffix (*ceshi /
#: *cheshi = test, *_FLY = in-flight, *_daifei) denote a DIFFERENT capture
#: condition and MUST remain distinct session_ids even though they share a
#: taxonomy_id."  Without this, ``DJMINI3ceshi\X.iq`` and ``DJMINI3\X.iq``
#: collapse into one session purely because the stems match.
CONDITION_SUFFIXES = ("ceshi", "cheshi", "_fly", "daifei", "dafei", "duima",
                      "kong", "hover", "test")

_RE_SLICE_SUFFIX = re.compile(r"_(\d+)-(\d+)s$")
_RE_PACK_STEM = re.compile(r"^pack(\d+)$")

#: ERRATA E-016.  The ZJU directory grammar is ``{model}-SNR{n}-{cf}m`` (D5,
#: dialect.RE_ZJU) where ``SNR{n}`` is an ORDINAL 1/2/3 with no dB unit: it is
#: the attenuation STEP of one capture at one frequency, not a separate capture.
#: Measured: 81 ZJU recordings carry ``-SNR{n}-`` and collapse to 27 stems, each
#: currently split across 3 spurious session_ids -- a 3-fold CV then trains on
#: SNR1+SNR2 and tests on SNR3 OF THE SAME RECORDING (leakage).  Stripping the
#: ordinal here fuses the steps into one session.  The lookahead ``(?=-|$)``
#: fires ONLY on the ZJU ordinal (digits followed by ``-`` or end): it must NOT
#: touch the AVATA2 native ladder ``SNR{n}dB`` (dialect.RE_UAV3), which is the
#: calibration asset of goal 3a and whose digits are followed by ``dB``.  A
#: looser ``-SNR\d+(?!dB)`` is WRONG: on ``-SNR10dB`` it backtracks to ``-SNR1``
#: and mangles the stem to ``...20dB``.
_RE_ZJU_SNR = re.compile(r"-SNR\d+(?=-|$)")


def condition_tag(class_dir):
    """Return the capture-condition marker carried by a directory name, or ''.

    Kept separate from taxonomy on purpose: same aircraft, different condition
    = same taxonomy_id, different session_id.
    """
    if not class_dir:
        return ""
    low = class_dir.lower()
    hits = [s for s in CONDITION_SUFFIXES if s.strip("_") in low]
    return "+".join(sorted(hits))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _wsplit(p):
    """Split a Windows path.  os.path cannot be used: this script may run on
    POSIX while the paths are Windows."""
    return [x for x in p.replace("/", "\\").split("\\") if x != ""]


def _wbase(p):
    parts = _wsplit(p)
    return parts[-1] if parts else p


def _wdir(p):
    parts = _wsplit(p)
    return "\\".join(parts[:-1])


def _ext(p):
    b = _wbase(p)
    i = b.rfind(".")
    return b[i:].lower() if i >= 0 else ""


def _stem(p):
    b = _wbase(p)
    i = b.rfind(".")
    return b[:i] if i > 0 else b


def _fmt(x, nd=6):
    if x is None or x == "":
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, float):
        s = ("%%.%df" % nd) % x
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    return "%s" % x


def _log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# input readers
# ---------------------------------------------------------------------------

def read_inventory(path, prefix):
    """Read e1_inventory.csv.  Returns list of dicts with relpath/size/mtime.

    The header is located by name because stage 1 may add columns; if no header
    is recognisable the file is treated as headerless path,size,mtime.
    """
    rows = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        try:
            first = next(rdr)
        except StopIteration:
            return rows
        low = [c.strip().lower() for c in first]
        has_header = any("path" in c for c in low) and not first[0].startswith("E:")
        if has_header:
            def col(*names):
                for n in names:
                    for i, c in enumerate(low):
                        if c == n:
                            return i
                for n in names:
                    for i, c in enumerate(low):
                        if n in c:
                            return i
                return None
            ip = col("path", "fullpath", "relpath")
            isz = col("size", "bytes", "length")
            imt = col("mtime_iso", "mtime", "modified")
            if ip is None:
                ip = 0
        else:
            ip, isz, imt = 0, 1, 2
            rdr = [first] + list(rdr)

        for row in rdr:
            if not row or len(row) <= ip:
                continue
            p = row[ip]
            if not p:
                continue
            sz = None
            if isz is not None and len(row) > isz:
                try:
                    sz = int(row[isz])
                except (ValueError, TypeError):
                    sz = None
            mt = row[imt] if (imt is not None and len(row) > imt) else ""
            rel = p[len(prefix):] if p.startswith(prefix) else p
            rows.append({"abspath": p, "relpath": rel, "size": sz, "mtime": mt})
    return rows


def read_hashes(path):
    """Read e1_hashes.csv.

    MEASURED: this file is HEADERLESS -- its columns are path,size,sha512 and
    the first line is data.  Detecting a header by column name silently drops
    the first row and yields 0 bytes.  A defensive check keeps it working if a
    header is ever added upstream.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            p, s, h = row[0], row[1], row[2]
            if not p:
                continue
            try:
                s = int(s)
            except (ValueError, TypeError):
                # a header line, or a malformed row -- skip it, do not abort
                continue
            out[p] = (s, h.strip())
    return out


# ---------------------------------------------------------------------------
# dtype / rate / duration
# ---------------------------------------------------------------------------

def dtype_for(relpath, source_root):
    """dtype is keyed on source path, never assumed globally (ERRATA E-009b).

    Returns (datatype, evidence).  Longest prefix wins so that a probed
    subdirectory overrides its unprobed parent root.
    """
    norm = relpath.replace("/", "\\")
    best = None
    for prefix, dt, conf in DTYPE_RULES:
        if norm == prefix or norm.startswith(prefix + "\\"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, dt, conf)
    if best is not None:
        return best[1], "%s:%s" % (best[2], best[0])
    if source_root:
        for prefix, dt, conf in DTYPE_RULES:
            if source_root == prefix:
                return dt, "%s:%s" % (conf, prefix)
    return DTYPE_DEFAULT[0], "%s:library_default" % DTYPE_DEFAULT[1]


def _snap_std_rate(r_msps):
    """Return the standard rate r_msps equals, or None."""
    if r_msps is None or r_msps <= 0:
        return None
    for std in STD_RATES_MSPS:
        if abs(r_msps - std) <= std * RATE_REL_TOL:
            return std
    return None


def session_stem(relpath, parsed):
    """The structural session key.

    All slices of one source recording, plus the uncut source itself, share a
    stem -- this is what whole-file SHA-512 cannot see (ERRATA E-011 (2)).
    Built by stripping the ``_A-Bs`` window and the （局部）partial mark from
    the basename.  The take number ``(N)`` is KEPT: ``..._80dB(2)_22-24s.iq``
    belongs with ``..._80dB(2).iq``, not with ``..._80dB.iq``.

    ERRATA E-016: the ZJU ordinal ``-SNR{n}`` is ALSO stripped, so the SNR1/2/3
    attenuation steps of one capture fuse into one session (they are not
    independent recordings).  This is the documented v2 re-cut: measured effect
    is 81 -> 27 sessions, 21 classes degraded off ``grouped_cv``.  The AVATA2
    native ladder ``SNR{n}dB`` is deliberately NOT stripped (see ``_RE_ZJU_SNR``).
    """
    s = _stem(relpath)
    s = s.replace(dialect.PARTIAL_MARK, "")
    s = _RE_SLICE_SUFFIX.sub("", s)
    s = _RE_ZJU_SNR.sub("", s)
    return s.strip()


def infer_rates(recs):
    """Two-pass sample-rate inference with recorded evidence per file.

    Pass 1 -- a slice whose delivered bytes divide exactly into a standard rate
    at its REQUESTED duration is self-evidencing: it was not truncated, so
    ``size / bps / requested_duration`` IS the rate.  Such files vote for their
    stem group, then their class directory, then their source root.

    Pass 2 -- everyone else inherits, in order of decreasing specificity:
    an explicit rate in the filename > stem-group vote > class-dir vote >
    root vote > library default.

    This is the correction forced by E-008: the earlier detector called a
    fractional derived rate a short write, which flagged the legitimate 5 MS/s
    exp3 files as "33 % truncated" merely because their siblings ran at
    100 MS/s.  A fractional rate means the duration is wrong, not the rate.
    """
    vote_stem = collections.defaultdict(collections.Counter)
    vote_dir = collections.defaultdict(collections.Counter)
    vote_root = collections.defaultdict(collections.Counter)

    for r in recs:
        r["rate_msps"] = None
        r["rate_source"] = ""
        declared = r["p"].get("sample_rate_msps")
        if declared:
            try:
                d = float(declared)
            except (TypeError, ValueError):
                d = None
            if d and d > 0:
                r["rate_msps"] = d
                r["rate_source"] = "filename"
                vote_stem[r["stem"]][d] += 3
                vote_dir[r["class_key"]][d] += 3
                vote_root[r["root_key"]][d] += 3
                continue

        req = r["requested_duration_s"]
        if req and r["size"] and r["bps"]:
            cand = (r["size"] / float(r["bps"])) / req / 1e6
            snapped = _snap_std_rate(cand)
            if snapped is not None:
                r["rate_msps"] = snapped
                r["rate_source"] = "exact_window"
                vote_stem[r["stem"]][snapped] += 2
                vote_dir[r["class_key"]][snapped] += 2
                vote_root[r["root_key"]][snapped] += 1

    for r in recs:
        if r["rate_msps"] is not None:
            continue
        for scope, table, key in (("stem", vote_stem, r["stem"]),
                                  ("class_dir", vote_dir, r["class_key"]),
                                  ("source_root", vote_root, r["root_key"])):
            c = table.get(key)
            if c:
                r["rate_msps"] = c.most_common(1)[0][0]
                r["rate_source"] = "vote:%s" % scope
                break
        if r["rate_msps"] is None:
            # Do NOT fall back to 100 MS/s.  The spec forbids it (E-009) and it
            # would be worse than a gap: a fabricated rate silently produces a
            # fabricated duration_s and num_samples that look measured.  81 ZJU
            # files + 2 FUTABA-T10J copies land here -- they carry no window,
            # no rate token and no sibling to vote, so the rate is genuinely
            # unknown until someone probes the capture metadata.  NULL
            # propagates to duration and is flagged for a probe.
            r["rate_source"] = "UNKNOWN_no_evidence"


def fill_durations(recs):
    """Derive delivered duration and num_samples from bytes; compare with the
    requested window (E-008)."""
    for r in recs:
        bps = r["bps"]
        sz = r["size"]
        if not sz or not bps:
            r["num_samples"] = None
            r["delivered_duration_s"] = None
            r["sample_align_ok"] = ""
            r["truncated_delivery"] = ""
            r["delivered_fraction"] = None
            continue
        r["num_samples"] = sz // bps
        r["sample_align_ok"] = (sz % bps == 0)
        if r["rate_msps"] is None:
            # dtype is known, so the sample count is; the rate is not, so the
            # duration must stay empty rather than be invented
            r["delivered_duration_s"] = None
            r["truncated_delivery"] = ""
            r["delivered_fraction"] = None
            continue
        rate_hz = r["rate_msps"] * 1e6
        r["delivered_duration_s"] = (sz / float(bps)) / rate_hz
        req = r["requested_duration_s"]
        if req:
            r["truncated_delivery"] = (r["delivered_duration_s"] < req - TRUNC_EPS_S)
            r["delivered_fraction"] = r["delivered_duration_s"] / req
        else:
            r["truncated_delivery"] = False
            r["delivered_fraction"] = None


# ---------------------------------------------------------------------------
# substring (tail) dedup -- the part whole-file SHA-512 cannot do
# ---------------------------------------------------------------------------

def find_tail_sources(recs):
    """For every slice, search for a file it could be a byte TAIL of.

    Condition, proven on this library to close to 0 bytes for 25/27 candidates:

        src_size == offset_bytes + slice_size,  offset_bytes % (rate*bps) == 0

    i.e. the slice is the final ``slice_size`` bytes of ``src`` and starts on a
    whole-second boundary.  Deriving the offset instead of asserting it from
    the filename is deliberate: PublicDataset window labels are re-invented on
    copy (ERRATA E-012), so ``pack3_15-16s.iq`` actually begins at t=14.000 s.

    THE ARITHMETIC IS ONLY INFORMATIVE FOR NON-ROUND SLICES.  If a slice is a
    whole number of seconds (800 MB at 100 MS/s cf32), the modulo test is
    satisfied by EVERY whole-second file in the taxonomy -- it carries no
    information at all.  Measured on this library: leaving that unguarded made
    five distinct 1-second ``pack1_*`` slices all "match" one 2-second source
    at offset 1, and inflated the E-012 fabricated-window count from 1 to 78.
    So each pair is labelled with a ``match_class``:

      arith_unique_offset_shift  informative, exactly one source, and the byte
                                 offset disagrees with the filename -> this is
                                 what an E-012 fabricated window looks like
      arith_unique_confirms      informative, exactly one source, offset agrees
      arith_ambiguous            informative but several sources fit
      uninformative_round        slice is whole seconds; arithmetic proves
                                 nothing.  The filename offset is recorded as a
                                 HYPOTHESIS for --verify-bytes to accept or
                                 reject, and no claim is made until it does.

    Only the first class is allowed to raise an E-012 anomaly unaided; the
    round ones need bytes.  Candidate sources are restricted to files sharing
    the resolved taxonomy so the search stays O(n*k) with small k, and
    cross-root pairs are allowed because the proven SIYI_MK32 case is
    cross-root.
    """
    by_tax = collections.defaultdict(list)
    for r in recs:
        if r["size"]:
            by_tax[r["tax_key"]].append(r)

    out = []
    for r in recs:
        r["tail_of"] = ""
        r["tail_offset_bytes"] = None
        r["derived_start_s"] = None
        r["start_label_mismatch"] = ""
        r["tail_match_class"] = ""
        r["tail_n_candidates"] = None
        if not r["is_slice_like"] or not r["size"] or r["rate_msps"] is None:
            continue
        step = int(round(r["rate_msps"] * 1e6)) * r["bps"]
        if step <= 0:
            continue

        informative = (r["size"] % step != 0)
        req_start = r["p"].get("time_start_s")
        try:
            req_start = float(req_start) if req_start not in (None, "") else None
        except (TypeError, ValueError):
            req_start = None

        cands = []
        for c in by_tax.get(r["tax_key"], ()):
            if c is r or not c["size"] or c["size"] <= r["size"]:
                continue
            delta = c["size"] - r["size"]
            if delta % step != 0:
                continue
            start_s = delta // step
            if c["delivered_duration_s"] is None:
                continue
            if start_s + r["delivered_duration_s"] > c["delivered_duration_s"] + 1e-6:
                continue
            cands.append((start_s, c))
        if not cands:
            continue

        n = len(cands)
        offsets = set(t[0] for t in cands)
        # Two candidates that are byte-duplicates of each other (the 262 GB
        # rawdata<->过大的原始数据 pairs, E-011) both match at the SAME offset.
        # That is not ambiguity about where the slice came from, only about
        # which copy to name.  Ambiguity is DISAGREEMENT ABOUT THE OFFSET;
        # counting sources instead of distinct offsets was hiding the one
        # E-012 case that had been byte-proven, because its source exists in
        # two roots.
        n_off = len(offsets)
        cands.sort(key=lambda t: (0 if t[1]["is_uncut"] else 1, -t[1]["size"],
                                  t[0], t[1]["relpath"]))
        if informative:
            if n_off == 1:
                start_s, src = cands[0]
                mismatch = (req_start is not None
                            and abs(req_start - start_s) > 1e-6)
                klass = ("arith_unique_offset_shift" if mismatch
                         else "arith_unique_confirms")
            else:
                # candidates genuinely disagree: assert no offset
                start_s, src = cands[0]
                mismatch = ""
                klass = "arith_ambiguous"
        else:
            # round slice: prefer the candidate that AGREES with the filename,
            # because that is the hypothesis worth spending byte reads on
            agree = [t for t in cands
                     if req_start is not None and abs(t[0] - req_start) <= 1e-6]
            pick = agree or cands
            pick = sorted(pick, key=lambda t: (0 if t[1]["is_uncut"] else 1,
                                               -t[1]["size"], t[0],
                                               t[1]["relpath"]))
            start_s, src = pick[0]
            mismatch = ""
            klass = "uninformative_round"

        r["tail_of"] = src["relpath"]
        r["tail_offset_bytes"] = start_s * step
        r["derived_start_s"] = float(start_s)
        r["start_label_mismatch"] = mismatch
        r["tail_match_class"] = klass
        r["tail_n_candidates"] = n
        out.append({
            "match_class": klass,
            "n_candidates": n,
            "arith_informative": informative,
            "slice_relpath": r["relpath"],
            "slice_size": r["size"],
            "source_relpath": src["relpath"],
            "source_size": src["size"],
            "offset_bytes": r["tail_offset_bytes"],
            "rate_msps": r["rate_msps"],
            "bytes_per_sample": r["bps"],
            "requested_start_s": "" if req_start is None else req_start,
            "derived_start_s": start_s,
            "start_label_mismatch": mismatch,
            "cross_root": src["root_key"] != r["root_key"],
            "residual_bytes": src["size"] - r["tail_offset_bytes"] - r["size"],
            "verified_bytes": "",
            "verify_windows": "",
        })
    return out


def verify_tail_pairs(pairs, prefix, limit, window=65536):
    """Read-only byte confirmation of tail candidates.

    Reads three ``window``-sized windows per pair (head / mid / tail) from both
    files -- about 400 KB of I/O per pair -- and compares them.  Arithmetic
    alone leaves a residual chance of coincidence; on the 4 pairs checked by
    hand all 12 windows matched exactly, which is what upgraded E-008 from
    inference to proof.  Nothing is written; files are opened 'rb'.
    """
    import hashlib
    done = 0
    for pr in pairs:
        if limit is not None and done >= limit:
            break
        a = prefix + pr["slice_relpath"]
        b = prefix + pr["source_relpath"]
        off = pr["offset_bytes"]
        ssz = pr["slice_size"]
        offs = [0, max(0, (ssz // 2) & ~7), max(0, ssz - window)]
        ok = 0
        tried = 0
        try:
            with open(a, "rb") as fa, open(b, "rb") as fb:
                for oa in offs:
                    n = min(window, ssz - oa)
                    if n <= 0:
                        continue
                    fa.seek(oa)
                    fb.seek(off + oa)
                    da = fa.read(n)
                    db = fb.read(n)
                    tried += 1
                    if len(da) == len(db) and hashlib.sha256(da).digest() == hashlib.sha256(db).digest():
                        ok += 1
        except (IOError, OSError) as e:
            pr["verified_bytes"] = "ERROR:%s" % (e.__class__.__name__,)
            pr["verify_windows"] = ""
            continue
        pr["verify_windows"] = "%d/%d" % (ok, tried)
        pr["verified_bytes"] = "1" if (tried > 0 and ok == tried) else "0"
        done += 1
    return done


def find_interior_sources(recs, pairs, prefix, screen=4096, window=65536,
                          max_cands=60):
    """Byte-probe slices at their CLAIMED window offset (interior extraction).

    A tail search can only ever find suffixes.  Most PublicDataset packs are
    not suffixes -- they are interior windows cut out of a much longer
    recording -- and for those the tail arithmetic is not merely uninformative,
    it matches the WRONG file.  Measured: all 88 pairs the tail search proposed
    for the round-slice class were refuted by bytes, while probing the claimed
    offset instead found the true source immediately.  Example:
    ``PublicDataset\\DJI FPV COMBO\\VTSBW=10\\pack1_10-11s.iq`` is byte-identical
    to ``rawdata\\DJFPVCOMBO\\...100m_10m.iq`` (23.775 s) at exactly t=10 s --
    a source the tail search never considered, because the slice is not its
    tail.

    So this is not a refinement of the tail search, it is the complementary
    half: tails are found by arithmetic and confirmed by bytes, interiors can
    only be found by bytes.  Both offsets ``A`` and ``A-1`` are probed, because
    the two proven E-012 cases both sit one second earlier than their label.

    Cheap by construction: a 4 KB screen rejects an unrelated candidate on the
    first read, and only a survivor pays for the three 64 KB confirmation
    windows.  READ-ONLY -- every open is 'rb'.
    """
    import hashlib

    def h(b):
        return hashlib.sha256(b).digest()

    # a slice already byte-proven as a tail needs no interior search
    proven = set(p["slice_relpath"] for p in pairs if p["verified_bytes"] == "1")

    by_tax = collections.defaultdict(list)
    for r in recs:
        if r["size"]:
            by_tax[r["tax_key"]].append(r)

    found = 0
    probed = 0
    for r in recs:
        if r["relpath"] in proven or not r["is_slice_like"]:
            continue
        if not r["size"] or r["rate_msps"] is None:
            continue
        req = r["p"].get("time_start_s")
        try:
            req = float(req) if req not in (None, "") else None
        except (TypeError, ValueError):
            req = None
        if req is None:
            continue
        step = int(round(r["rate_msps"] * 1e6)) * r["bps"]
        if step <= 0:
            continue

        # Prefer an uncut, large, same-root source.  Among byte-duplicate
        # copies this keeps the canonical root rather than the 过大 mirror.
        cands = [c for c in by_tax.get(r["tax_key"], ())
                 if c is not r and c["size"] > r["size"]]
        cands.sort(key=lambda c: (0 if c["is_uncut"] else 1,
                                  0 if c["root_key"] == r["root_key"] else 1,
                                  -c["size"], c["relpath"]))
        cands = cands[:max_cands]

        hit = None
        for c in cands:
            for start_s in (req, req - 1.0):
                if start_s < 0:
                    continue
                off = int(round(start_s * step))
                if off + r["size"] > c["size"]:
                    continue
                try:
                    with open(prefix + r["relpath"], "rb") as fa, \
                            open(prefix + c["relpath"], "rb") as fb:
                        probed += 1
                        fa.seek(0)
                        fb.seek(off)
                        if h(fa.read(screen)) != h(fb.read(screen)):
                            continue
                        ok = 0
                        tried = 0
                        for oa in (0, max(0, (r["size"] // 2) & ~7),
                                   max(0, r["size"] - window)):
                            n = min(window, r["size"] - oa)
                            if n <= 0:
                                continue
                            fa.seek(oa)
                            fb.seek(off + oa)
                            tried += 1
                            if h(fa.read(n)) == h(fb.read(n)):
                                ok += 1
                except (IOError, OSError):
                    continue
                if tried and ok == tried:
                    hit = (start_s, c, "%d/%d" % (ok, tried), off)
                    break
            if hit:
                break
        if not hit:
            continue

        start_s, src, wins, off = hit
        shifted = abs(start_s - req) > 1e-6
        found += 1
        r["tail_of"] = src["relpath"]
        r["tail_offset_bytes"] = off
        r["derived_start_s"] = start_s
        r["start_label_mismatch"] = shifted
        r["tail_match_class"] = ("interior_offset_shift" if shifted
                                 else "interior_confirms")
        pairs.append({
            "match_class": r["tail_match_class"],
            "n_candidates": len(cands),
            "arith_informative": False,
            "slice_relpath": r["relpath"],
            "slice_size": r["size"],
            "source_relpath": src["relpath"],
            "source_size": src["size"],
            "offset_bytes": off,
            "rate_msps": r["rate_msps"],
            "bytes_per_sample": r["bps"],
            "requested_start_s": req,
            "derived_start_s": start_s,
            "start_label_mismatch": shifted,
            "cross_root": src["root_key"] != r["root_key"],
            "residual_bytes": src["size"] - off - r["size"],
            "verified_bytes": "1",
            "verify_windows": wins,
        })
    return found, probed


# ---------------------------------------------------------------------------
# sessions and leakage groups
# ---------------------------------------------------------------------------

def build_sessions(recs, tail_pairs, use_tail=True, use_sha=True, assign=True):
    """Assign session_id.

    A session is one capture event.  Members are unioned from two relations:

      1. same structural stem  (slices + their uncut source)
      2. tail-of relation      (covers PublicDataset re-slices, whose stem is
                                only ``pack3`` and carries no source identity)

    Relation 2 is applied for every match class EXCEPT the byte-refuted ones.
    The asymmetry is deliberate: for a leakage grouping a false merge only
    makes a group larger (harmless -- the two recordings simply stay on the
    same side of the split), while a missed merge puts the same bytes in train
    and test.  So an unverified hypothesis is still allowed to merge.  A pair
    that --verify-bytes actively REFUTED is different: it is not an unproven
    guess but a disproven one, and letting it merge would write a lineage into
    the manifest that the bytes say is false.  ``use_tail=False`` recomputes
    without relation 2 so the size of the effect is reportable instead of
    hidden.

    This is the Shulman leakage group: any cross-validation must split on
    session_id, never on file.  With R=1 measured for RFUAV (one continuous
    session per class, single X310, single site) this collapses to roughly one
    session per class -- which is exactly the finding that forces a
    cross-dataset headline rather than an intra-dataset grouped CV.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_rel = {}
    for r in recs:
        by_rel[r["relpath"]] = r
        find(r["relpath"])

    # relation 1: structural stem within a taxonomy AND capture condition.
    # The condition tag must be in the key: class_aliases.json's _policy states
    # that a shared taxonomy across *ceshi / *_FLY / *_daifei directories is
    # NOT a session merge, because those are different capture conditions.
    stem_groups = collections.defaultdict(list)
    for r in recs:
        stem_groups[(r["tax_key"], r["condition"], r["stem"])].append(r["relpath"])
    for _, members in stem_groups.items():
        for m in members[1:]:
            union(members[0], m)

    # relation 2: tail-of / interior-of, minus anything the bytes refuted
    if use_tail:
        for pr in tail_pairs:
            if str(pr.get("verified_bytes")) == "0":
                continue
            if pr["slice_relpath"] in by_rel and pr["source_relpath"] in by_rel:
                union(pr["source_relpath"], pr["slice_relpath"])

    # relation 3: identical whole-file SHA-512 (gap 75).  Two byte-identical
    # files ARE the same emission and must never straddle the split.  Neither
    # relation 1 (stem) nor 2 (tail) catches a duplicate that was renamed out of
    # its stem group AND is not a substring of its twin.  Same asymmetry as
    # above: a false merge only enlarges a group (harmless), a missed one leaks.
    # Narrow by measurement (gap 75: 1/25 hash groups crossed sessions, 0.80 GB)
    # -- reported, not oversold.  Gated so ``use_sha=False`` recomputes without
    # it and the marginal effect is reportable instead of hidden (same treatment
    # as ``use_tail``).
    if use_sha:
        by_sha = collections.defaultdict(list)
        for r in recs:
            if r["sha512"]:
                by_sha[r["sha512"]].append(r["relpath"])
        for members in by_sha.values():
            for m in members[1:]:
                union(members[0], m)

    groups = collections.defaultdict(list)
    for r in recs:
        groups[find(r["relpath"])].append(r)

    if not assign:
        return [{"n": len(v)} for v in groups.values()]

    seq_by = collections.Counter()
    sessions = []
    for root_rel in sorted(groups.keys()):
        members = groups[root_rel]
        members.sort(key=lambda x: (-(x["size"] or 0), x["relpath"]))
        anchor = members[0]
        # capture date must come from the uncut source, never from a slice:
        # slice mtimes are the DATE OF SLICING (2024-11-26 cluster), not capture
        uncut = [m for m in members if m["is_uncut"] and m["mtime"]]
        date_src = (uncut or members)[0]
        d = _date8(date_src["mtime"])
        tax = anchor["tax_short"]
        key = (tax, d)
        seq_by[key] += 1
        sid = "%s-u%02d-r%02d-s%02d-d%s-%02d" % (tax, 1, 1, 1, d, seq_by[key])
        cfg = 0
        for m in members:
            m["session_id"] = sid
            cfg += 1
            m["recording_id"] = "%s-c%02d" % (sid, cfg)
        sessions.append({
            "session_id": sid,
            "taxonomy_short": tax,
            "n_recordings": len(members),
            "total_bytes": sum(m["size"] or 0 for m in members),
            "capture_condition": anchor["condition"],
            "capture_date": d,
            "date_from": date_src["relpath"],
            "date_from_is_uncut": bool(uncut),
            "n_uncut": len(uncut),
            "n_slices": sum(1 for m in members if m["is_slice_like"]),
            "roots": "|".join(sorted(set(m["root_key"] for m in members))),
            "cross_root": len(set(m["root_key"] for m in members)) > 1,
            "anchor": anchor["relpath"],
        })
    return sessions


# ---------------------------------------------------------------------------
# Class statistics (E-014 item 1, E-015 item 1)
# ---------------------------------------------------------------------------
#
# THE INCLUSION RULE.  `build_labels.py::evidence()` counts a row iff
#
#     (artifact_class or "recording") == "recording"
#
# and nothing else -- note the default, which matters because a blank column
# must read as `recording`, not as excluded.  E-014 phrases the same basis as
# "artifact_class=recording 且已挂税目", but `tax_id` is built as
# `".".join(...) or "unk"` and therefore NEVER comes back empty, so the second
# clause excludes nothing and the rule is the single predicate below.  It is
# written once, here, because two copies of a counting rule is how a published
# class count starts disagreeing with the registry that cites it.
#
# VERIFICATION.  On the real library this must reproduce E-014's figures
# exactly: 793 rows / 227 sessions / 63 classes over all roots, and 700 / 134 /
# 40 once the two non-own roots are dropped.  A different number means the rule
# is wrong, not that the library changed -- the HEADLINE prints all six so the
# comparison is one glance rather than a re-derivation.
#
# THE FOUR DIMENSIONS (E-015 item 6).  None of these is reportable alone:
#
#   n_recordings   overstates independence by up to 27x -- one capture re-cut
#                  27 times is 27 rows and one session.
#   bytes          overstates the 200 MS/s material 2x against 100 MS/s for the
#                  same amount of signal.
#   nominal_sec    double-counts: 531 of 700 own-capture rows are re-cut
#                  windows of signal already counted in their parent.
#   unique_sec     uncut rows only.  The only one that counts each second of
#                  captured RF exactly once, and the number the datasheet owes
#                  the reader.
#
# So classes.csv prints all four side by side and library_totals.csv carries
# `unique_seconds_total`.  `recordings.csv` exports `is_uncut` per row rather
# than a per-row column named `unique_seconds`: E-015 defines that name as a
# per-class quantity, and a row-level field wearing it would be a trap.  With
# `is_uncut` published, every class figure here is recomputable -- and so
# falsifiable -- from recordings.csv alone.


def is_recording_row(rec):
    """The one definition of "counts toward a class statistic"."""
    return (rec["p"].get("artifact_class") or "recording") == "recording"


def class_stats(recs, own_only):
    """taxonomy_id -> the four dimensions + session count, one scope.

    ``own_only`` drops the two roots E-005/E-006 measured as int16-converted
    with cross-class duplicate content.  Both scopes are always emitted,
    because E-014 item 1's requirement is that the DENOMINATOR be declared:
    "63 classes" and "40 classes" are both true statements about this library
    and only one of them is reportable.
    """
    out = {}
    for r in recs:
        if not is_recording_row(r):
            continue
        if own_only and r["root_key"] in split_grouped.NON_OWN_ROOTS:
            continue
        e = out.setdefault(r["tax_id"], {
            "taxonomy_id": r["tax_id"], "tax_short": r["tax_short"],
            "n_recordings": 0, "n_uncut": 0, "sessions": set(),
            "bytes": 0, "nominal_seconds": 0.0, "unique_seconds": 0.0,
            "roots": set(), "duration_fallbacks": 0,
        })
        e["n_recordings"] += 1
        e["bytes"] += r["size"] or 0
        e["roots"].add(r["root_key"])
        if r.get("session_id"):
            e["sessions"].add(r["session_id"])
        sec, fell_back = split_grouped.row_seconds(
            {"delivered_duration_s": r.get("delivered_duration_s"),
             "size_bytes": r["size"]})
        if fell_back:
            e["duration_fallbacks"] += 1
        e["nominal_seconds"] += sec
        if r["is_uncut"]:
            e["n_uncut"] += 1
            e["unique_seconds"] += sec
    return out


CLASS_COLS = (
    "scope", "taxonomy_id", "tax_short",
    "n_sessions", "eval_tier", "tier_basis",
    "n_recordings", "n_uncut", "nominal_seconds", "unique_seconds",
    "bytes", "gib", "duration_fallbacks", "roots",
)


def class_rows(stats, scope):
    rows = []
    for tid in sorted(stats):
        e = stats[tid]
        n_sess = len(e["sessions"])
        rows.append({
            "scope": scope,
            "taxonomy_id": tid,
            "tax_short": e["tax_short"],
            "n_sessions": n_sess,
            "eval_tier": split_grouped.tier_of(n_sess),
            # E-016: the tier is a session count; this says whether the count
            # rests on evidence.  See split_grouped.tier_basis -- 21 of 42
            # all_roots CV tiers come from classes where no two files were ever
            # proven to share a session.
            "tier_basis": split_grouped.tier_basis(
                n_sess, e["n_recordings"], e["unique_seconds"]),
            "n_recordings": e["n_recordings"],
            "n_uncut": e["n_uncut"],
            "nominal_seconds": e["nominal_seconds"],
            "unique_seconds": e["unique_seconds"],
            "bytes": e["bytes"],
            "gib": e["bytes"] / float(1 << 30),
            "duration_fallbacks": e["duration_fallbacks"],
            "roots": "|".join(sorted(e["roots"])),
        })
    return rows


_RE_DATE = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})")


def _date8(mtime_iso):
    if not mtime_iso:
        return "00000000"
    m = _RE_DATE.search(mtime_iso)
    if not m:
        return "00000000"
    return "%s%s%s" % m.groups()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

REC_COLS = (
    "recording_id", "session_id", "taxonomy_id", "tax_short",
    "own_capture", "is_uncut",
    "n_sessions_in_class", "eval_tier",
    "n_sessions_in_class_all_roots", "eval_tier_all_roots",
    "vendor", "model", "device_role", "signal_state", "class_dir",
    "capture_condition",
    "source_root", "source_relpath", "source_file",
    "size_bytes", "sha512", "mtime_iso",
    "datatype", "bytes_per_sample", "dtype_evidence",
    "sample_rate_msps", "rate_source",
    "num_samples", "sample_align_ok",
    "delivered_duration_s", "requested_window_s", "requested_duration_s",
    "delivered_fraction", "truncated_delivery",
    "center_freq_mhz", "vtsbw_mhz", "rx_gain_db",
    "attenuation_label_raw", "snr_nominal_label", "snr_nominal_db", "snr_ordinal",
    "distance_m", "take", "pack_index", "rf_chain", "band_tag",
    "is_slice", "is_partial", "is_misfiled", "is_test_artifact",
    "lost_samples_tag", "variant_unresolved", "artifact_class",
    "stem", "tail_of", "tail_offset_bytes", "tail_match_class",
    "tail_n_candidates", "derived_start_s", "start_label_mismatch",
    "naming_dialect", "unparsed_residue", "parse_ok", "parse_reason",
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build RFUAV v2 manifests from the E1 inventory + hash pass.")
    ap.add_argument("--manifests", default=r"E:\Drone_dataset\RFUAV-v2\manifests",
                    help="output + input directory (only place written to)")
    ap.add_argument("--inventory", default=None, help="default <manifests>/e1_inventory.csv")
    ap.add_argument("--hashes", default=None, help="default <manifests>/e1_hashes.csv")
    ap.add_argument("--prefix", default="E:\\Drone_dataset\\RFUAV\\",
                    help="source-root prefix stripped from inventory paths")
    ap.add_argument("--verify-bytes", type=int, default=0, metavar="N",
                    help="read-only byte-confirm up to N tail candidates (0=skip)")
    ap.add_argument("--out-prefix", default="", help="prefix for output filenames")
    args = ap.parse_args(argv)

    M = args.manifests
    inv_path = args.inventory or os.path.join(M, "e1_inventory.csv")
    hash_path = args.hashes or os.path.join(M, "e1_hashes.csv")

    if not os.path.isdir(M):
        os.makedirs(M)

    _log("reading inventory  %s" % inv_path)
    inv = read_inventory(inv_path, args.prefix)
    _log("  rows = %d" % len(inv))

    _log("reading hashes     %s   (HEADERLESS: path,size,sha512)" % hash_path)
    hashes = read_hashes(hash_path)
    _log("  hashed = %d files, %.1f GB"
         % (len(hashes), sum(v[0] for v in hashes.values()) / 1024.0 ** 3))

    # ---- select signal files -------------------------------------------
    recs = []
    skipped = collections.Counter()
    for row in inv:
        rel = row["relpath"]
        e = _ext(rel)
        if e in ARCHIVE_EXT:
            skipped["archive%s" % e] += 1
            continue
        if e not in SIG_EXT:
            skipped["ext%s" % (e or "(none)")] += 1
            continue
        try:
            p = dialect.parse(rel)
        except Exception as exc:                                  # noqa: BLE001
            skipped["parse_error"] += 1
            _log("  parse error: %s -> %s: %s" % (rel, type(exc).__name__, exc))
            continue

        dt, dt_ev = dtype_for(rel, p.get("source_root"))
        t0, t1 = p.get("time_start_s"), p.get("time_end_s")
        req_dur = None
        req_win = ""
        if t0 not in (None, "") and t1 not in (None, ""):
            try:
                req_dur = float(t1) - float(t0)
                req_win = "%s-%s" % (_fmt(t0), _fmt(t1))
            except (TypeError, ValueError):
                req_dur = None
        vend = p.get("vendor") or ""
        mod = p.get("model") or ""
        role = p.get("device_role") or "unk"
        tax_id = ".".join(x for x in (role, vend, mod) if x) or "unk"
        tax_short = ("%s-%s" % (vend or "unk", mod or "unk")).replace(" ", "").lower()

        rec = {
            "relpath": rel,
            "abspath": row["abspath"],
            "size": row["size"],
            "mtime": row["mtime"],
            "p": p,
            "dtype": dt,
            "dtype_evidence": dt_ev,
            # Loud lookup, not `.get(dt, 8)`.  A silent 8-byte fallback is D9-1
            # in miniature: add a token to DTYPE_RULES without adding it to the
            # vocabulary and every file under that prefix would get float32
            # arithmetic -- halved num_samples, halved duration, wrong derived
            # rate -- with nothing raised.  Unreachable today by construction
            # (every DTYPE_RULES token is in VOCAB); the point is that it stays
            # unreachable after the next edit.
            "bps": iq_dtype.from_manifest_token(dt).bytes_per_sample,
            "requested_duration_s": req_dur,
            "requested_window_s": req_win,
            "root_key": p.get("source_root") or (_wsplit(rel)[0] if _wsplit(rel) else ""),
            "class_key": "%s|%s" % (p.get("source_root") or "", p.get("class_dir") or ""),
            "tax_key": tax_id,
            "tax_id": tax_id,
            "tax_short": tax_short,
            "condition": condition_tag(p.get("class_dir")),
            "is_slice_like": bool(p.get("is_slice")) or bool(req_dur),
            "is_uncut": not (bool(p.get("is_slice")) or bool(req_dur)),
        }
        rec["stem"] = session_stem(rel, p)
        h = hashes.get(row["abspath"]) or hashes.get(rel)
        if h and row["size"] is not None and h[0] != row["size"]:
            skipped["hash_size_mismatch"] += 1
        rec["sha512"] = h[1] if h else ""
        recs.append(rec)

    _log("signal files = %d   (skipped: %s)"
         % (len(recs), ", ".join("%s=%d" % kv for kv in skipped.most_common(8))))

    # ---- rate, duration, tails, sessions -------------------------------
    _log("inferring sample rates (2-pass, evidence recorded per file)")
    infer_rates(recs)
    fill_durations(recs)
    # a second pass: files whose rate came from a vote now have a delivered
    # duration, which the tail search needs on BOTH sides of every pair
    _log("searching for byte-tail sources (substring dedup, ERRATA E-011)")
    tail_pairs = find_tail_sources(recs)
    _log("  tail candidates = %d" % len(tail_pairs))

    if args.verify_bytes:
        _log("byte-verifying up to %d tail candidates (read-only, ~400 KB each)"
             % args.verify_bytes)
        n = verify_tail_pairs(tail_pairs, args.prefix, args.verify_bytes)
        ok = sum(1 for p in tail_pairs if p["verified_bytes"] == "1")
        bad = sum(1 for p in tail_pairs if p["verified_bytes"] == "0")
        err = sum(1 for p in tail_pairs if str(p["verified_bytes"]).startswith("ERROR"))
        _log("  verified %d: identical=%d differ=%d error=%d" % (n, ok, bad, err))

        # A refuted pair is disproven, not merely unproven.  Retract the
        # lineage the arithmetic asserted so the manifest never carries a
        # claim the bytes contradict.
        refuted = set(p["slice_relpath"] for p in tail_pairs
                      if p["verified_bytes"] == "0")
        for r in recs:
            if r["relpath"] in refuted:
                r["tail_of"] = ""
                r["tail_offset_bytes"] = None
                r["derived_start_s"] = None
                r["start_label_mismatch"] = ""
                r["tail_match_class"] = "refuted_by_bytes"
        if refuted:
            _log("  retracted %d arithmetic lineages the bytes refuted" % len(refuted))

        # The complementary half: interior windows, which a tail search cannot
        # see by construction.  Only runs with --verify-bytes because it IS a
        # byte search -- there is no arithmetic shortcut for an interior cut.
        _log("probing claimed-offset interior sources (read-only, 4 KB screen)")
        nfound, nprobed = find_interior_sources(recs, tail_pairs, args.prefix)
        _log("  interior sources found = %d  (%d candidate probes)"
             % (nfound, nprobed))

    _log("assigning session_id (stem union tail-of union sha512)")
    sessions = build_sessions(recs, tail_pairs, use_tail=True, use_sha=True)
    stem_only = build_sessions(recs, tail_pairs, use_tail=False, use_sha=False,
                               assign=False)
    no_sha = build_sessions(recs, tail_pairs, use_tail=True, use_sha=False,
                            assign=False)
    _log("  sessions = %d   (stem-only would be %d; tail-of + sha512 merged %d "
         "further, of which sha512 alone merged %d)"
         % (len(sessions), len(stem_only), len(stem_only) - len(sessions),
            len(no_sha) - len(sessions)))

    # ---- whole-file dedup ----------------------------------------------
    by_hash = collections.defaultdict(list)
    for r in recs:
        if r["sha512"]:
            by_hash[r["sha512"]].append(r)
    dup_groups = [(h, v) for h, v in by_hash.items() if len(v) > 1]
    dup_groups.sort(key=lambda kv: -sum(x["size"] or 0 for x in kv[1]))

    by_size = collections.defaultdict(list)
    for r in recs:
        if r["size"]:
            by_size[r["size"]].append(r)
    samesize_nohash = [(s, v) for s, v in by_size.items()
                       if len(v) > 1 and not all(x["sha512"] for x in v)]
    samesize_nohash.sort(key=lambda kv: -kv[0] * len(kv[1]))

    # ---- write outputs --------------------------------------------------
    op = args.out_prefix
    written = []

    def wcsv(name, cols, rows):
        path = os.path.join(M, op + name)
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            for r in rows:
                w.writerow([_fmt(r.get(c, "")) for c in cols])
        written.append((name, len(rows)))
        return path

    rec_rows = []
    # Both scopes, computed once over the finished session assignment.  These
    # are the fields `--assert-no-leak` and split_grouped.py currently
    # RECOMPUTE from scratch; publishing them means the splitter consumes the
    # same numbers the datasheet cites instead of its own re-derivation.
    cs_own = class_stats(recs, own_only=True)
    cs_all = class_stats(recs, own_only=False)
    for r in recs:
        p = r["p"]
        tid = r["tax_id"]
        e_own, e_all = cs_own.get(tid), cs_all.get(tid)
        n_own = len(e_own["sessions"]) if e_own else 0
        n_all = len(e_all["sessions"]) if e_all else 0
        rec_rows.append({
            "recording_id": r.get("recording_id", ""),
            "session_id": r.get("session_id", ""),
            "taxonomy_id": tid,
            "tax_short": r["tax_short"],
            "own_capture": r["root_key"] not in split_grouped.NON_OWN_ROOTS,
            "is_uncut": r["is_uncut"],
            "n_sessions_in_class": n_own,
            "eval_tier": split_grouped.tier_of(n_own),
            "n_sessions_in_class_all_roots": n_all,
            "eval_tier_all_roots": split_grouped.tier_of(n_all),
            "vendor": p.get("vendor") or "",
            "model": p.get("model") or "",
            "device_role": p.get("device_role") or "",
            "signal_state": p.get("signal_state") or "",
            "class_dir": p.get("class_dir") or "",
            "capture_condition": r["condition"],
            "source_root": r["root_key"],
            "source_relpath": r["relpath"],
            "source_file": p.get("source_file") or _wbase(r["relpath"]),
            "size_bytes": r["size"],
            "sha512": r["sha512"],
            "mtime_iso": r["mtime"],
            "datatype": r["dtype"],
            "bytes_per_sample": r["bps"],
            "dtype_evidence": r["dtype_evidence"],
            "sample_rate_msps": r["rate_msps"],
            "rate_source": r["rate_source"],
            "num_samples": r["num_samples"],
            "sample_align_ok": r["sample_align_ok"],
            "delivered_duration_s": r["delivered_duration_s"],
            "requested_window_s": r["requested_window_s"],
            "requested_duration_s": r["requested_duration_s"],
            "delivered_fraction": r["delivered_fraction"],
            "truncated_delivery": r["truncated_delivery"],
            "center_freq_mhz": p.get("center_freq_mhz"),
            "vtsbw_mhz": p.get("vtsbw_mhz"),
            "rx_gain_db": p.get("rx_gain_db"),
            "attenuation_label_raw": p.get("attenuation_label_raw"),
            "snr_nominal_label": p.get("snr_nominal_label"),
            "snr_nominal_db": p.get("snr_nominal_db"),
            "snr_ordinal": p.get("snr_ordinal"),
            "distance_m": p.get("distance_m"),
            "take": p.get("take"),
            "pack_index": p.get("pack_index"),
            "rf_chain": p.get("rf_chain"),
            "band_tag": p.get("band_tag"),
            "is_slice": p.get("is_slice"),
            "is_partial": p.get("is_partial"),
            "is_misfiled": p.get("is_misfiled"),
            "is_test_artifact": p.get("is_test_artifact"),
            "lost_samples_tag": p.get("lost_samples_tag"),
            "variant_unresolved": p.get("variant_unresolved"),
            "artifact_class": p.get("artifact_class"),
            "stem": r["stem"],
            "tail_of": r["tail_of"],
            "tail_offset_bytes": r["tail_offset_bytes"],
            "derived_start_s": r["derived_start_s"],
            "start_label_mismatch": r["start_label_mismatch"],
            "naming_dialect": p.get("naming_dialect"),
            "unparsed_residue": p.get("unparsed_residue"),
            "parse_ok": p.get("ok"),
            "parse_reason": p.get("reason"),
        })
    wcsv("recordings.csv", REC_COLS, rec_rows)

    wcsv("classes.csv", CLASS_COLS,
         class_rows(cs_own, "own_capture") + class_rows(cs_all, "all_roots"))

    # `unique_seconds_total` lives here rather than in a summary line, because
    # E-014 item 1 requires the denominator to travel WITH the number: a bare
    # total in stdout gets quoted without its scope.  One row per scope, so
    # citing either one forces you to name which.
    totals = []
    for scope, cs in (("own_capture", cs_own), ("all_roots", cs_all)):
        rows = class_rows(cs, scope)

        def in_scope(r):
            return (scope == "all_roots"
                    or r["root_key"] not in split_grouped.NON_OWN_ROOTS)

        n_sess = len(set(
            r.get("session_id") for r in recs
            if is_recording_row(r) and r.get("session_id") and in_scope(r)))
        # E-016b.  sessions.csv and leakage_groups.csv publish 231 groups; this
        # column publishes 227 for all_roots, and until now nothing on disk
        # explained the difference.  Both are correct and they answer different
        # questions.  MEASURED: exactly 4 sessions consist entirely of
        # `experiment_sandbox` rows --
        #     flysky-fs_i6x-...-d20241206-01        5 rows
        #     futaba-t10j-...-d20241206-01          4 rows
        #     futaba-t10j-...-d20241206-02          4 rows
        #     radiolink-at9s_pro-...-d20241206-01   1 row
        # `class_stats` drops them because a sandbox copy is not an independent
        # capture; the leakage tables KEEP them because a sandbox copy that
        # lands across a split boundary from its parent is leakage of exactly
        # the E-013 kind.  So the gap is the inclusion rule working, and the fix
        # is to make the two counts reconcile in the file rather than in
        # somebody's head.
        n_sess_all = len(set(
            r.get("session_id") for r in recs
            if r.get("session_id") and in_scope(r)))
        supported = [r for r in rows if split_grouped.tier_is_supported(
            r["n_sessions"], r["n_recordings"], r["unique_seconds"])]
        totals.append({
            "scope": scope,
            "n_classes": len(rows),
            "n_sessions": n_sess,
            "n_sessions_incl_derived": n_sess_all,
            "n_sessions_derived_only": n_sess_all - n_sess,
            "n_recordings": sum(r["n_recordings"] for r in rows),
            "n_uncut": sum(r["n_uncut"] for r in rows),
            "nominal_seconds_total": sum(r["nominal_seconds"] for r in rows),
            "unique_seconds_total": sum(r["unique_seconds"] for r in rows),
            "unique_minutes_total": sum(r["unique_seconds"] for r in rows) / 60.0,
            "bytes_total": sum(r["bytes"] for r in rows),
            "gib_total": sum(r["gib"] for r in rows),
            "duration_fallbacks": sum(r["duration_fallbacks"] for r in rows),
            "n_grouped_cv": sum(1 for r in rows
                                if r["eval_tier"] == split_grouped.TIER_CV),
            "n_holdout_1v1": sum(1 for r in rows
                                 if r["eval_tier"] == split_grouped.TIER_1V1),
            "n_unsplittable": sum(1 for r in rows
                                  if r["eval_tier"] == split_grouped.TIER_NONE),
            # E-016: the same two tier counts, restricted to classes whose tier
            # rests on measured session structure.  Quote these, not the ones
            # above, in any sentence containing the word "cross-validation".
            "n_grouped_cv_supported": sum(
                1 for r in supported
                if r["eval_tier"] == split_grouped.TIER_CV),
            "n_holdout_1v1_supported": sum(
                1 for r in supported
                if r["eval_tier"] == split_grouped.TIER_1V1),
            "n_tier_unsupported": len(rows) - len(supported),
            # E-016: distinct question from the one above, and a different
            # number.  `n_tier_unsupported` asks "how many tiers assert a split
            # they cannot back" (unsplittable is exempt -- it asserts none);
            # `n_thin_signal` asks "how many classes carry under
            # TIER_MIN_UNIQUE_S of unique signal", regardless of tier.  Both are
            # correct and they reconcile here rather than in somebody's head --
            # the E-016b lesson, applied to E-016's own columns.
            "n_thin_signal": sum(
                1 for r in rows
                if split_grouped.BASIS_THIN in r["tier_basis"]),
            "reportable": scope == "own_capture",
        })
    wcsv("library_totals.csv",
         ("scope", "n_classes", "n_sessions", "n_sessions_incl_derived",
          "n_sessions_derived_only", "n_recordings", "n_uncut",
          "nominal_seconds_total", "unique_seconds_total",
          "unique_minutes_total", "bytes_total", "gib_total",
          "duration_fallbacks", "n_grouped_cv", "n_holdout_1v1",
          "n_unsplittable", "n_grouped_cv_supported",
          "n_holdout_1v1_supported", "n_tier_unsupported", "n_thin_signal",
          "reportable"),
         totals)

    wcsv("sessions.csv",
         ("session_id", "taxonomy_short", "capture_condition", "n_recordings",
          "total_bytes", "capture_date", "date_from", "date_from_is_uncut", "n_uncut",
          "n_slices", "roots", "cross_root", "anchor"),
         sessions)

    wcsv("dedup_substring.csv",
         ("match_class", "n_candidates", "arith_informative",
          "slice_relpath", "slice_size", "source_relpath", "source_size",
          "offset_bytes", "rate_msps", "bytes_per_sample",
          "requested_start_s", "derived_start_s", "start_label_mismatch",
          "cross_root", "residual_bytes", "verified_bytes", "verify_windows"),
         tail_pairs)

    dup_rows = []
    for h, v in dup_groups:
        for r in v:
            dup_rows.append({
                "sha512": h, "size_bytes": r["size"], "relpath": r["relpath"],
                "source_root": r["root_key"], "session_id": r.get("session_id", ""),
                "group_n": len(v),
                "wasted_bytes": (len(v) - 1) * (r["size"] or 0) // max(1, len(v)),
            })
    wcsv("dedup_wholefile.csv",
         ("sha512", "size_bytes", "relpath", "source_root", "session_id",
          "group_n", "wasted_bytes"), dup_rows)

    ss_rows = []
    for s, v in samesize_nohash:
        for r in v:
            ss_rows.append({
                "size_bytes": s, "relpath": r["relpath"],
                "source_root": r["root_key"], "sha512": r["sha512"],
                "group_n": len(v), "session_id": r.get("session_id", ""),
            })
    wcsv("dedup_samesize_unhashed.csv",
         ("size_bytes", "relpath", "source_root", "sha512", "group_n",
          "session_id"), ss_rows)

    lk_rows = []
    for s in sessions:
        lk_rows.append({"group_key": s["session_id"], "n_files": s["n_recordings"],
                        "total_bytes": s["total_bytes"], "reason": "session",
                        "note": "cross_root" if s["cross_root"] else ""})
    wcsv("leakage_groups.csv",
         ("group_key", "n_files", "total_bytes", "reason", "note"), lk_rows)

    ev = collections.Counter(r["rate_source"] for r in recs)
    rate_rows = []
    for r in recs:
        rate_rows.append({
            "relpath": r["relpath"], "rate_msps": r["rate_msps"],
            "rate_source": r["rate_source"], "datatype": r["dtype"],
            "bytes_per_sample": r["bps"], "size_bytes": r["size"],
            "requested_duration_s": r["requested_duration_s"],
            "delivered_duration_s": r["delivered_duration_s"],
            "size_over_bps_over_reqdur_msps": (
                (r["size"] / float(r["bps"]) / r["requested_duration_s"] / 1e6)
                if (r["size"] and r["requested_duration_s"]) else ""),
        })
    wcsv("rate_evidence.csv",
         ("relpath", "rate_msps", "rate_source", "datatype", "bytes_per_sample",
          "size_bytes", "requested_duration_s", "delivered_duration_s",
          "size_over_bps_over_reqdur_msps"), rate_rows)

    anom = []

    def add(kind, r, detail):
        anom.append({"kind": kind, "relpath": r["relpath"],
                     "session_id": r.get("session_id", ""),
                     "size_bytes": r["size"], "detail": detail})

    for r in recs:
        if r["truncated_delivery"] is True:
            add("E-008_truncated_delivery", r,
                "delivered %.4fs of requested %.1fs (%.1f%%) rate=%s src=%s"
                % (r["delivered_duration_s"], r["requested_duration_s"],
                   100.0 * r["delivered_fraction"], _fmt(r["rate_msps"]),
                   r["tail_of"] or "NOT_FOUND"))
        if r["dtype"] != "cf32_le":
            add("E-009b_non_cf32_dtype", r, "%s (%d B/sample) via %s"
                % (r["dtype"], r["bps"], r["dtype_evidence"]))
        if r["dtype_evidence"].startswith("ASSUMED"):
            add("dtype_unprobed", r,
                "dtype %s is ASSUMED, not measured (%s) -- probe before sidecar"
                % (r["dtype"], r["dtype_evidence"]))
        if r["sample_align_ok"] is False:
            add("misalignment", r, "size %% %d != 0" % r["bps"])
        if r["start_label_mismatch"] is True and r["tail_match_class"] in (
                "arith_unique_offset_shift", "interior_offset_shift"):
            add("E-012_fabricated_window", r,
                "filename claims start %s, bytes say %s (%s, source %s)"
                % (_fmt(r["p"].get("time_start_s")), _fmt(r["derived_start_s"]),
                   r["tail_match_class"], r["tail_of"]))
        if r["tail_match_class"] == "arith_ambiguous":
            add("tail_ambiguous", r,
                "%d sources fit the tail arithmetic; offset NOT asserted"
                % r["tail_n_candidates"])
        if r["rate_source"] == "UNKNOWN_no_evidence":
            add("rate_unknown", r,
                "no filename rate, no window, no sibling vote -> sample_rate "
                "and duration_s left NULL (NOT defaulted to 100 MS/s); probe "
                "capture metadata before this file gets a sidecar")
        if r["is_slice_like"] and not r["tail_of"] and r["rate_msps"] is not None:
            # a truncated slice with no source is a real gap in provenance; a
            # whole-second slice with no source is just a source that was never
            # kept, which is the normal case and should not drown the ledger
            if r["truncated_delivery"] is True:
                add("orphan_truncated_slice", r,
                    "short delivery and no TAIL source; note the search finds "
                    "tails only -- a head/prefix extract (the （局部） case) is "
                    "structurally invisible to it and needs byte verification "
                    "at offset 0 instead")
            else:
                add("slice_source_absent", r,
                    "no source found by tail arithmetic NOR by interior probe "
                    "at the claimed offset; this is evidence of absence only if "
                    "--verify-bytes ran, since an interior window is invisible "
                    "to arithmetic alone")
        if r["p"].get("is_misfiled"):
            add("E-004_misfiled", r, "class_dir=%s model=%s"
                % (r["p"].get("class_dir"), r["p"].get("model")))
        if r["p"].get("lost_samples_tag"):
            add("lost_samples", r, "tag=%s" % r["p"].get("lost_samples_tag"))
    for h, v in dup_groups:
        for r in v[1:]:
            add("E-011_wholefile_duplicate", r, "sha512 group of %d, first=%s"
                % (len(v), v[0]["relpath"]))
    wcsv("anomalies.csv",
         ("kind", "relpath", "session_id", "size_bytes", "detail"), anom)

    # ---- summary --------------------------------------------------------
    _log("")
    _log("=== WROTE (into %s) ===" % M)
    for n, k in written:
        _log("  %-32s %6d rows" % (op + n, k))

    tot_b = sum(r["size"] or 0 for r in recs)
    trunc = [r for r in recs if r["truncated_delivery"] is True]
    _log("")
    _log("=== HEADLINE ===")
    _log("  signal files            %d   (%.1f GB)" % (len(recs), tot_b / 1024.0 ** 3))
    _log("  sessions (leakage grps) %d" % len(sessions))
    _log("  files per session       %.2f mean, %d max"
         % (float(len(recs)) / max(1, len(sessions)),
            max([s["n_recordings"] for s in sessions] or [0])))
    _log("  cross-root sessions     %d" % sum(1 for s in sessions if s["cross_root"]))
    _log("  hashed / total          %d / %d" % (sum(1 for r in recs if r["sha512"]), len(recs)))
    _log("  whole-file dup groups   %d  (%d files)"
         % (len(dup_groups), sum(len(v) for _, v in dup_groups)))
    _log("  substring pairs         %d  (cross-root %d)"
         % (len(tail_pairs), sum(1 for p in tail_pairs if p["cross_root"])))
    mc = collections.Counter(p["match_class"] for p in tail_pairs)
    _log("    by class: " + ", ".join("%s=%d" % kv for kv in mc.most_common()))
    vb = collections.Counter(str(p["verified_bytes"]) for p in tail_pairs)
    _log("    byte-verified: identical=%s refuted=%s unchecked=%s"
         % (vb.get("1", 0), vb.get("0", 0), vb.get("", 0)))
    _log("  truncated deliveries    %d  (with source found %d)"
         % (len(trunc), sum(1 for r in trunc if r["tail_of"])))
    _log("  fabricated windows      %d  (E-012; byte-proven only)"
         % sum(1 for r in recs
               if r["start_label_mismatch"] is True
               and r["tail_match_class"] in ("arith_unique_offset_shift",
                                             "interior_offset_shift")))
    _log("  non-cf32 dtype          %d" % sum(1 for r in recs if r["dtype"] != "cf32_le"))
    _log("  dtype ASSUMED (unprobed)%d"
         % sum(1 for r in recs if r["dtype_evidence"].startswith("ASSUMED")))
    _log("  unknown rates (NULL)    %d  (left empty, never defaulted to 100)"
         % sum(1 for r in recs if r["rate_source"] == "UNKNOWN_no_evidence"))
    _log("")
    _log("  rate evidence: " + ", ".join("%s=%d" % kv for kv in ev.most_common()))
    dist = collections.Counter(_fmt(r["rate_msps"]) for r in recs)
    _log("  rate values  : " + ", ".join("%s MS/s=%d" % kv for kv in dist.most_common()))
    dtd = collections.Counter(r["dtype"] for r in recs)
    _log("  dtypes       : " + ", ".join("%s=%d" % kv for kv in dtd.most_common()))
    kd = collections.Counter(a["kind"] for a in anom)
    _log("  anomalies    : " + ", ".join("%s=%d" % kv for kv in kd.most_common()))

    # ---- class scale, both denominators (E-014.1) ------------------------
    # Printed as a table rather than a sentence on purpose.  Every previous
    # version of this number was quoted without its scope, and "63 classes"
    # read as a claim about RFUAV's own capture when 23 of those 63 exist only
    # inside the int16-converted foreign roots.
    _log("")
    _log("=== CLASS SCALE (E-014.1: the denominator is part of the number) ===")
    _log("  %-11s %7s %8s %9s %7s %11s %11s %10s"
         % ("scope", "class", "session", "rec", "uncut", "nom_sec", "uniq_sec", "GiB"))
    for t in totals:
        _log("  %-11s %7d %8d %9d %7d %11.1f %11.1f %10.1f%s"
             % (t["scope"], t["n_classes"], t["n_sessions"], t["n_recordings"],
                t["n_uncut"], t["nominal_seconds_total"],
                t["unique_seconds_total"], t["gib_total"],
                "   <- REPORTABLE" if t["reportable"] else ""))
    _log("  E-015.6: none of rec / nom_sec / GiB is reportable alone --")
    _log("    rec overstates independence up to 27x (one capture, 27 re-cuts);")
    _log("    GiB overstates 200 MS/s 2x for the same signal;")
    _log("    nom_sec double-counts every re-cut window.  uniq_sec is the")
    _log("    only column that counts each captured second exactly once.")
    for t in totals:
        _log("  %-11s eval tiers: grouped_cv=%d  holdout_1v1=%d  unsplittable=%d"
             % (t["scope"], t["n_grouped_cv"], t["n_holdout_1v1"],
                t["n_unsplittable"]))
        # E-016.  The line above is a session count; this one is the claim.
        # "unsupported" counts only the split-asserting tiers -- an unsplittable
        # class asserts no split, so thin signal cannot make it overclaim one.
        # n_thin_signal is printed alongside so the thin unsplittable rows are
        # still counted somewhere instead of vanishing from the aggregate.
        _log("  %-11s   of those, tier SUPPORTED by measured session structure "
             "+ >=%.0fs signal: grouped_cv=%d  holdout_1v1=%d  "
             "(overclaiming a split=%d; classes under %.0fs of signal=%d)"
             % (t["scope"], split_grouped.TIER_MIN_UNIQUE_S,
                t["n_grouped_cv_supported"], t["n_holdout_1v1_supported"],
                t["n_tier_unsupported"], split_grouped.TIER_MIN_UNIQUE_S,
                t["n_thin_signal"]))
        if t["n_grouped_cv_supported"] != t["n_grouped_cv"]:
            _log("  %-11s   E-016: %d of %d grouped_cv tiers rest on classes "
                 "where no two files were proven to share a session, or on "
                 "under %.0fs of signal. Do NOT quote grouped_cv=%d as "
                 "\"cross-validatable classes\"."
                 % (t["scope"],
                    t["n_grouped_cv"] - t["n_grouped_cv_supported"],
                    t["n_grouped_cv"], split_grouped.TIER_MIN_UNIQUE_S,
                    t["n_grouped_cv"]))
    # E-016b.  sessions.csv / leakage_groups.csv count leakage groups; the table
    # above counts independent captures.  Print the bridge so the two figures
    # are never diffed by a reader with no way to explain the difference.
    for t in totals:
        if t["n_sessions_derived_only"]:
            _log("  %-11s sessions: %d capture-bearing + %d built only from "
                 "derived/sandbox rows = %d leakage groups (sessions.csv)"
                 % (t["scope"], t["n_sessions"],
                    t["n_sessions_derived_only"], t["n_sessions_incl_derived"]))
            _log("  %-11s   a derived-only session is not a capture (dropped "
                 "here) but IS a leakage group (kept there) -- both counts are "
                 "correct and they answer different questions." % "")
    fb = sum(t["duration_fallbacks"] for t in totals)
    if fb:
        _log("  WARNING: %d rows had no delivered_duration_s -- seconds above "
             "fall back to size/8/100e6 for those" % fb)
    own = [t for t in totals if t["reportable"]][0]
    _log("  unique_seconds_total (own_capture) = %.1f s = %.1f min"
         % (own["unique_seconds_total"], own["unique_minutes_total"]))
    _log("  E-014 expects, if the inclusion rule is right:")
    _log("    own_capture 40 classes / 134 sessions / 700 rec;"
         " all_roots 63 / 227 / 793")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
    sys.exit(main())
