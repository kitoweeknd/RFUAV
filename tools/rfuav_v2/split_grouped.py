# -*- coding: utf-8 -*-
"""
split_grouped.py -- the leakage-clean splitter and the CI check that keeps it
honest.

    check   given any (item, group, split) assignment, prove no group straddles
            two splits.  Exit 1 if one does.  This is the check that would have
            caught E-013 before publication.

    make    build a split from recordings.csv by assigning whole SESSIONS to
            folds, never frames and never recordings.  Emits eval_tier and
            n_sessions_in_class (spec section 6, E-014 entry action 1).

WRITE SCOPE: only the --out file (and only in `make`).  Reads manifests only.

------------------------------------------------------------------------------
WHY THE GROUPING KEY IS session_id AND NOTHING ELSE
------------------------------------------------------------------------------
Four measured properties of this library (spec section 6 ERRATA) rule out every
cheaper key:

E-013   the released train/valid/test directories put the same pack on both
        sides -- 12,080 / 12,080 held-out frames (100.0 %).  The mechanism is
        measured, not guessed: the splitter computed int(0.3 * n) INSIDE each
        group (124 (train,valid) cycles, strict alternation in 37/37 classes,
        train fraction 0.2968 +- 0.0042 across a 13.6x group-size range).  So
        not one of the 124 groups was held out.  A per-frame or per-block key
        cannot express "hold this group out"; only a group key can.

E-011   the dangerous duplication is SUBSTRING duplication, invisible to
        whole-file SHA-512: 631 byte-proven substring pairs, 706/719 of them
        crossing source roots.  session_id already absorbs these (stem grouping
        plus substring lineage merged 286 -> 231 sessions).  Grouping by
        filename, by recording, or by whole-file hash puts the same bytes on
        both sides.

E-011.3 PublicDataset is a re-slicing layer over rawdata, not an independent
        capture set.  "train on rawdata, test on PublicDataset" is the same
        byte stream on both sides.  Only session_id crosses that boundary
        correctly, because it is root-agnostic by construction.

E-014   sessions, not recordings, bound what evaluation is possible.
        rc.yunzhuo.h30 has 27 recordings and 1 session; rc.siyi.mk32 has 52 and
        2.  Reporting independence per recording overstates it by up to 27x.

------------------------------------------------------------------------------
THE TWO METHODOLOGY LESSONS THIS FILE IS BUILT TO NOT REPEAT
------------------------------------------------------------------------------
Both were paid for once already (spec section 6, E-013 methodology note):

1.  A GROUP KEY IS NOT A CONDITION KEY.  close/middle/far and the SNR labels
    are measurement conditions.  The same session legitimately appears under
    several of them.  Folding conditions into the split vocabulary manufactures
    false positives; ignoring the distinction manufactures false negatives.
    So `check` takes --group-col and --split-col explicitly and guesses
    nothing, and --condition-col is reported but never treated as a split.

2.  A CORRECT COUNT IS NOT A READABLE VERDICT.  The first version of the E-013
    audit computed the right leak count and never printed which splits were
    involved, because it iterated a dict's keys and unpacked a tuple into two
    scalars.  So every violation here prints the full offending rows, and the
    summary is derived from those rows rather than counted separately.
"""

import argparse
import collections
import csv
import hashlib
import os
import sys


# --------------------------------------------------------------------------
# tiers -- see spec section 6, E-014 item 4
# --------------------------------------------------------------------------
TIER_CV = "grouped_cv"        # >= 3 sessions: K-fold over sessions
TIER_1V1 = "holdout_1v1"      # == 2 sessions: one train, one test, no CV
TIER_NONE = "unsplittable"    # == 1 session: no leakage-clean split exists

# Roots that are not RFUAV's own capture.  E-005/E-006: the ZJU tree is
# int16-converted with cross-class duplicates.  E-009b/D7: .dat_pic is
# DroneRFa-derived (ci16_le, D7_dronerfa_rf dialect), not a capture of ours.
NON_OWN_ROOTS = ("浙大被预处理为IQ的数据", ".dat_pic")


def tier_of(n_sessions):
    if n_sessions >= 3:
        return TIER_CV
    if n_sessions == 2:
        return TIER_1V1
    return TIER_NONE


# E-016.  `tier_of` counts sessions and nothing else, which is the right rule
# for "does a leakage-clean split EXIST" and the wrong one for "is this class
# cross-validatable".  Measured on the 2026-07-31 manifest run:
#
#   own_capture grouped_cv (20 classes): uniq_s min 31.27  med 49.26  max 334.50
#   all_roots   grouped_cv (42 classes): uniq_s min  1.80  med 31.27  max 334.50
#     -> 21 of those 42 hold under 10 s of unique signal.  0 of 20 in own_capture.
#
# And the reason they are tiered CV at all is worse than thin: for 21 of them
# `n_sessions == n_recordings`, i.e. NO TWO FILES WERE EVER PROVEN TO SHARE A
# SESSION.  The stem-union/tail-of grouping found no evidence, so every file
# became its own singleton group and the class scored 3+ sessions by default.
# The tier is high BECAUSE THE GROUPING EVIDENCE IS ABSENT, which is the exact
# inverse of what a reader takes "grouped_cv" to mean.  In own_capture that
# shape occurs once in 40 classes; across all_roots, 24 times in 63.
#
# The inversion it produces, verbatim from that run:
#
#   uav.dji.mavic3    grouped_cv    3 sessions    1.80 s    3 recordings
#   bg.kong           unsplittable  1 session    13.74 s    8 recordings
#   rc.herelink.hx4   unsplittable  1 session    12.77 s   16 recordings
#
# A class with 7.6x less signal outranks one with eight recordings, purely
# because a foreign delivery scattered three files across three directories.
#
# `tier_of` is deliberately NOT changed: it is imported by three modules and
# pinned by two test classes, and re-tiering it here would silently re-cut every
# published split.  The tier stays; the basis travels beside it.  Same treatment
# E-015 gave the four size dimensions -- publish both, never one.
TIER_MIN_UNIQUE_S = 10.0
# Not a round number chosen for looking principled: 3-fold grouped CV over a
# class leaves each fold ~1/3 of its unique signal, so 10 s total is ~3.3 s per
# fold -- already below one STFT window budget for most of this library.  Under
# it, the fold structure is arithmetic rather than evaluation.  It also happens
# to separate the two scopes cleanly (0 own-capture classes, 21 foreign ones),
# which is evidence the floor is measuring the right thing and not just cutting
# the distribution somewhere convenient.

BASIS_MEASURED = "measured"
BASIS_SINGLETON = "singleton_sessions"
BASIS_THIN = "thin_signal"


def tier_basis(n_sessions, n_recordings, unique_seconds):
    """Why a class's ``eval_tier`` is what it is -- ``measured`` or the defect.

    Returns ``BASIS_MEASURED`` when the tier rests on proven session structure
    and enough signal to fold, else a ``+``-joined list of the failures.  A
    single-session class is exempt from the singleton check: ``unsplittable``
    claims nothing about grouping, so there is nothing to qualify.

    This is a *disclosure*, not a veto.  A class can be genuinely thin and
    still belong in the library; what it cannot do is be counted in a
    "42 classes are cross-validatable" sentence without the reader being told
    that 21 of them have never had two files proven to share a capture.
    """
    flags = []
    if n_sessions >= 2 and n_sessions == n_recordings:
        flags.append(BASIS_SINGLETON)
    if unique_seconds < TIER_MIN_UNIQUE_S:
        flags.append(BASIS_THIN)
    return "+".join(flags) if flags else BASIS_MEASURED


def tier_is_supported(n_sessions, n_recordings, unique_seconds):
    """True iff the tier may be quoted as an evaluation claim.

    Tier-aware on purpose.  ``unsplittable`` asserts that *no* leakage-clean
    split exists; thin signal cannot refute that claim, it is a reason for it.
    Counting such a class as unsupported inflates the disclosure with the rows
    whose tier is the most defensible in the library.  MEASURED on the first
    E-016 run: ``all_roots`` published ``n_tier_unsupported = 23`` = 21
    ``grouped_cv`` + 2 ``unsplittable``, printed directly beneath a line
    listing only the two split-asserting tiers -- which reads as "23 classes
    overclaim a split" when 21 do.  Over-reporting a defect count is the same
    failure as under-reporting it, so only the tiers that assert a split can
    fail to support one.

    The ``thin_signal`` flag stays on those two rows in ``tier_basis``: it is
    true, and a trainer should see it before using a 2-second class.  What it
    must not do is answer a splittability question the tier never asked.  The
    ``n_thin_signal`` total exists so those rows stay counted somewhere.
    """
    if tier_of(n_sessions) == TIER_NONE:
        return True
    return tier_basis(n_sessions, n_recordings,
                      unique_seconds) == BASIS_MEASURED


def stable_rank(seed, key):
    """Deterministic pseudo-random ordering.

    Not random.shuffle(): that depends on the interpreter's PRNG, so a split
    would silently change across Python versions and stop being reproducible
    from the manifest alone.  A hash of (seed, key) is stable forever.
    """
    h = hashlib.sha256(("%s\x00%s" % (seed, key)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def row_seconds(r):
    """Signal time carried by one recording, in seconds.

    E-009 measured that the sample rate is NOT constant across this library
    (100, 200, 75, 15, 5 MS/s).  Balancing folds by bytes therefore over-weights
    the 200 MS/s recordings by 2x relative to the number of frames a loader will
    actually cut from them.  delivered_duration_s (derived from bytes / bps /
    rate, E-008) is the unit that matches what a model sees.

    Returns (seconds, fell_back).  fell_back=True means duration was missing and
    a 100 MS/s cf32 assumption was used -- reported, never silent.
    """
    try:
        d = float(r.get("delivered_duration_s") or 0)
    except ValueError:
        d = 0.0
    if d > 0:
        return d, False
    try:
        return int(r.get("size_bytes") or 0) / 8.0 / 100e6, True
    except ValueError:
        return 0.0, True


def row_bytes(r):
    try:
        return int(r.get("size_bytes") or 0)
    except ValueError:
        return 0


# The predicate is DEFINED once, in build_manifests.py, at the only place where
# it can be computed correctly:
#
#     is_uncut = not (bool(p["is_slice"]) or bool(req_dur))
#
# `req_dur` comes from parsing a start/stop pair out of the filename, and it is
# the half that cannot be recovered downstream by eye: a row can carry
# is_slice=0 and still be a cut window because its NAME declares the window.
# This library has rows of exactly that shape, so
#
#     is_uncut  !=  not is_slice
#
# and the difference is not a rounding detail -- it is signal counted twice.
# Every consumer therefore READS the exported column instead of re-deriving it.
# There is deliberately no fallback: guessing from is_slice alone is the bug
# these two functions exist to make impossible, and a loose guess that silently
# over-counts is worse than a manifest that has to be regenerated.
_NO_IS_UNCUT = (
    "FATAL: %s has no `is_uncut` column.\n"
    "  This manifest predates the column.  Re-run build_manifests.py;\n"
    "  do NOT substitute `not is_slice` -- that counts filename-declared\n"
    "  windows (requested_duration_s) as originals and double-counts their\n"
    "  signal.  See E-015 item 1.")


def require_is_uncut(fieldnames, path):
    """Refuse to proceed on a manifest that predates ``is_uncut``.

    Called once per file, immediately after the header is read, so the run dies
    before it writes a split or a labels.json built on a guess.
    """
    if "is_uncut" not in (fieldnames or ()):
        raise SystemExit(_NO_IS_UNCUT % path)


def row_is_uncut(r):
    """True if this row is an original capture, not a re-cut window of one."""
    v = r.get("is_uncut")
    if v is None:
        raise SystemExit(_NO_IS_UNCUT % "this recordings row")
    return str(v).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# check -- the CI primitive
# --------------------------------------------------------------------------
def cmd_check(args):
    with open(args.assignment, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("FAIL: %s has no rows" % args.assignment)
        return 2

    for col in (args.group_col, args.split_col):
        if col not in rows[0]:
            print("FAIL: column %r not in %s (have: %s)"
                  % (col, args.assignment, ", ".join(sorted(rows[0]))))
            return 2

    ignore = set(s.strip() for s in args.ignore_splits.split(",") if s.strip())

    # group -> split -> [rows].  Conditions are carried for reporting only;
    # they are never part of the key (methodology lesson 1).
    per_group = collections.defaultdict(lambda: collections.defaultdict(list))
    per_group_class = collections.defaultdict(set)
    n_used = 0
    for r in rows:
        split = (r.get(args.split_col) or "").strip()
        group = (r.get(args.group_col) or "").strip()
        if not group:
            continue
        if split in ignore:
            continue
        n_used += 1
        per_group[group][split].append(r)
        if args.class_col and r.get(args.class_col):
            per_group_class[group].add(r[args.class_col].strip())

    item_col = args.item_col if (args.item_col and args.item_col in rows[0]) else None

    def show(r):
        if item_col:
            return r[item_col]
        return " | ".join("%s=%s" % (k, r[k]) for k in sorted(r) if r[k])[:200]

    # ---- violation 1: a group appears under more than one split ----------
    straddling = {g: d for g, d in per_group.items() if len(d) > 1}

    # ---- violation 2: a group spans more than one class -------------------
    # If it does, "hold out this session" is not even well defined, and the
    # class labels of both sides are suspect (cf. E-004, KONG cross-class).
    multiclass = {g: cs for g, cs in per_group_class.items() if len(cs) > 1}

    splits_seen = sorted({s for d in per_group.values() for s in d})
    print("assignment      : %s" % args.assignment)
    print("rows            : %d total, %d used (%d ignored as %s)"
          % (len(rows), n_used, len(rows) - n_used, sorted(ignore) or "-"))
    print("group column    : %s  -> %d distinct groups" % (args.group_col, len(per_group)))
    print("split column    : %s  -> %s" % (args.split_col, ", ".join(splits_seen)))
    if args.condition_col and rows and args.condition_col in rows[0]:
        conds = sorted({(r.get(args.condition_col) or "").strip() for r in rows})
        print("condition column: %s  -> %s  (reported, NEVER a split key)"
              % (args.condition_col, ", ".join(c for c in conds if c)))

    rc = 0
    if straddling:
        rc = 1
        n_items = sum(len(rs) for d in straddling.values() for rs in d.values())
        print("\nLEAK: %d of %d groups straddle a split boundary (%d items)"
              % (len(straddling), len(per_group), n_items))
        for g in sorted(straddling)[:args.max_report]:
            d = straddling[g]
            print("  group %s  ->  %s"
                  % (g, "  ".join("%s x%d" % (s, len(d[s])) for s in sorted(d))))
            for s in sorted(d):
                for r in d[s][:args.evidence]:
                    print("      [%s] %s" % (s, show(r)))
        if len(straddling) > args.max_report:
            print("  ... +%d more groups" % (len(straddling) - args.max_report))
    else:
        print("\nOK: no group appears in more than one split (%d groups checked)"
              % len(per_group))

    if multiclass:
        rc = 1
        print("\nILL-DEFINED: %d groups span more than one class" % len(multiclass))
        for g in sorted(multiclass)[:args.max_report]:
            print("  group %s  ->  %s" % (g, ", ".join(sorted(multiclass[g]))))

    print("\nVERDICT: %s" % ("LEAK DETECTED" if rc else "CLEAN"))
    return rc


# --------------------------------------------------------------------------
# make -- build the split
# --------------------------------------------------------------------------
def cmd_make(args):
    with open(args.recordings, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        require_is_uncut(rdr.fieldnames, args.recordings)
        rows = list(rdr)

    keep = [r for r in rows
            if r.get("artifact_class") == args.artifact_class
            and (r.get("taxonomy_id") or "").strip()
            and (r.get("session_id") or "").strip()]

    # Excluding the foreign roots is the DEFAULT, not an opt-in.  Measured
    # reason: of the 63 taxonomy_ids present when they are included, 23 exist
    # ONLY inside them -- and 63 is close enough to the >=60-class goal that a
    # default-flag run produces a table which looks like the goal is already
    # met.  E-005/E-006 measured the ZJU tree as int16-converted with
    # cross-class duplicate content, and E-014 item 3 forbids counting its
    # classes toward that goal at all.  An opt-in exclusion for data known to
    # be unusable is the same failure shape as E-013: a plausible default that
    # silently produces a publishable-looking wrong number.
    dropped = collections.Counter()
    if not args.include_foreign_roots:
        cls_before = {r["taxonomy_id"] for r in keep}
        kept = []
        for r in keep:
            if r.get("source_root") in NON_OWN_ROOTS:
                dropped[r.get("source_root", "")] += 1
            else:
                kept.append(r)
        keep = kept
        dropped_classes = sorted(cls_before - {r["taxonomy_id"] for r in keep})
    else:
        dropped_classes = []

    # class -> session -> rows
    per_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in keep:
        per_class[r["taxonomy_id"]][r["session_id"]].append(r)

    # A session must belong to exactly one class or the grouping is undefined.
    owner = {}
    conflicts = []
    for cls, sess in per_class.items():
        for s in sess:
            if s in owner and owner[s] != cls:
                conflicts.append((s, owner[s], cls))
            owner[s] = cls
    if conflicts:
        print("REFUSING TO SPLIT: %d session(s) span two classes" % len(conflicts))
        for s, a, b in conflicts[:20]:
            print("  session %s -> %s AND %s" % (s, a, b))
        return 2

    def sess_load(rs):
        tot = 0.0
        fell_back = False
        for r in rs:
            d, fb = row_seconds(r)
            tot += d
            fell_back = fell_back or fb
        return tot, fell_back

    out_rows = []
    report = []
    n_fallback = 0
    for cls in sorted(per_class):
        sess = per_class[cls]
        n = len(sess)
        tier = tier_of(n)

        if tier == TIER_NONE:
            only = list(sess)[0]
            for r in sess[only]:
                out_rows.append((r, "excluded", -1, tier, n,
                                 "single session: no leakage-clean split exists"))
            report.append((cls, n, tier, {"excluded": len(sess[only])}, [], -1))
            continue

        # Greedy largest-first bin packing over sessions, balancing SIGNAL TIME.
        # Session sizes span 1..52 recordings (E-014 item 6), so round-robin
        # would leave folds lopsided; and folds must be balanced by content,
        # not by session count.  Hash breaks ties so the result is stable.
        k = 2 if tier == TIER_1V1 else min(args.folds, n)
        loads = {}
        for s in sess:
            L, fb = sess_load(sess[s])
            loads[s] = L
            if fb:
                n_fallback += 1
        order = sorted(sess, key=lambda s: (-loads[s], stable_rank(args.seed, s)))
        load = [0.0] * k
        assign = {}
        for s in order:
            f = min(range(k), key=lambda i: (load[i], i))
            assign[s] = f
            load[f] += loads[s]

        # The held-out fold is the LIGHTEST one, so training never gets less
        # signal than testing.  With k>=3 the folds are already near-equal so
        # this is a marginal choice; with k==2 it is the whole ballgame --
        # otherwise a 2-session class trains on 1 recording and tests on 26.
        held = min(range(k), key=lambda i: (load[i], i))

        counts = collections.Counter()
        for s, f in assign.items():
            split = args.test_name if f == held else args.train_name
            for r in sess[s]:
                out_rows.append((r, split, f, tier, n, ""))
            counts[split] += len(sess[s])
        report.append((cls, n, tier, dict(counts), load, held))

    # ---- write ----------------------------------------------------------
    # `is_uncut` travels with the split for the same reason it is exported from
    # the manifest: whoever recomputes seconds-per-fold from this file must not
    # have to re-derive it, because the obvious re-derivation is wrong.
    cols = ["recording_id", "session_id", "taxonomy_id", "capture_condition",
            "source_root", "source_relpath", "size_bytes",
            "delivered_duration_s", "is_slice", "is_uncut",
            "split", "fold", "eval_tier", "n_sessions_in_class", "exclusion_reason"]
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r, split, fold, tier, n, reason in out_rows:
            w.writerow([r.get("recording_id", ""), r["session_id"], r["taxonomy_id"],
                        r.get("capture_condition", ""), r.get("source_root", ""),
                        r.get("source_relpath", ""), r.get("size_bytes", ""),
                        r.get("delivered_duration_s", ""), r.get("is_slice", ""),
                        "1" if row_is_uncut(r) else "0",
                        split, fold, tier, n, reason])

    # ---- report ---------------------------------------------------------
    print("recordings in    : %s  (%d rows, %d usable %s)"
          % (args.recordings, len(rows), len(keep), args.artifact_class))
    if dropped:
        print("foreign roots    : DROPPED %d rows (default; E-005/E-006/E-014.3)"
              % sum(dropped.values()))
        for root, n in dropped.most_common():
            print("                   %-28s %4d rows" % (root, n))
        if dropped_classes:
            print("                   -> %d taxonomy_ids exist ONLY there and are"
                  " NOT counted: %s"
                  % (len(dropped_classes), ", ".join(dropped_classes)))
    if args.include_foreign_roots:
        print("foreign roots    : *** INCLUDED by --include-foreign-roots ***")
        print("                   Class counts from this run are NOT reportable"
              " as RFUAV coverage")
        print("                   (ZJU tree is int16-converted with cross-class"
              " duplicates, E-005/E-006).")
    print("split written to : %s  (%d rows)\n" % (args.out, len(out_rows)))

    print("%-30s %3s %-13s %-4s %-22s %s"
          % ("CLASS", "ses", "eval_tier", "held", "recordings per split",
             "seconds per fold"))
    for cls, n, tier, counts, load, held in report:
        print("%-30s %3d %-13s %-4s %-22s %s"
              % (cls[:30], n, tier, ("f%d" % held) if held >= 0 else "-",
                 " ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())),
                 " ".join("%.1f" % L for L in load)))

    tt = collections.Counter(t for _, _, t, _, _, _ in report)
    print("\nclasses by tier : %s"
          % "  ".join("%s=%d" % (k, tt[k]) for k in (TIER_CV, TIER_1V1, TIER_NONE)))

    # Aggregate by split AND by fold, in recordings / seconds / unique seconds
    # / GB.  Four columns because no one of them is reportable alone:
    #   rec    -- overstates independence up to 27x (E-014 item 6)
    #   GB     -- over-weights the 200 MS/s captures 2x (E-009)
    #   sec    -- double-counts, because 531 of 700 recordings are re-cut
    #             windows of an uncut parent in the same session (E-011.3)
    #   uniq_s -- seconds of signal that exist only once: uncut rows only,
    #             read from the manifest's `is_uncut` column via row_is_uncut.
    #             It is NOT `is_slice != 1`: that predicate calls a
    #             filename-declared window an original and re-inflates the very
    #             double-count this column exists to remove.
    def agg(keyfn):
        d = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
        for r, split, fold, tier, n, reason in out_rows:
            k = keyfn(split, fold, tier)
            s, _ = row_seconds(r)
            d[k][0] += 1
            d[k][1] += s
            if row_is_uncut(r):
                d[k][2] += s
            d[k][3] += row_bytes(r)
        return d

    by_split = agg(lambda split, fold, tier: split)
    by_fold = agg(lambda split, fold, tier: fold)
    n_sess_by_split = collections.defaultdict(set)
    for r, split, fold, tier, n, reason in out_rows:
        n_sess_by_split[split].add(r["session_id"])

    print("\n%-10s %5s %5s %9s %9s %9s"
          % ("SPLIT", "ses", "rec", "sec", "uniq_s", "GB"))
    for k in sorted(by_split):
        c, s, u, b = by_split[k]
        print("%-10s %5d %5d %9.1f %9.1f %9.1f"
              % (k, len(n_sess_by_split[k]), c, s, u, b / 1e9))
    print("\n%-10s %5s %5s %9s %9s %9s"
          % ("FOLD", "", "rec", "sec", "uniq_s", "GB"))
    for k in sorted(by_fold):
        c, s, u, b = by_fold[k]
        print("%-10s %5s %5d %9.1f %9.1f %9.1f"
              % (("f%d" % k) if k >= 0 else "excluded", "", c, s, u, b / 1e9))

    # Folds are NOT globally equal-sized and cannot be: a 2-session class can
    # only populate 2 folds, so with --folds 3 the higher folds see fewer
    # classes.  That is a property of the library (E-014), not a bug here, but
    # it must be stated rather than left for a reader to discover.
    if tt[TIER_1V1] and args.folds > 2:
        print("\nNOTE: %d classes have exactly 2 sessions, so they populate only "
              "f0/f1.\n      Folds are balanced WITHIN each class, not globally; "
              "report per-class,\n      or run --folds 2 for a globally uniform "
              "K." % tt[TIER_1V1])
    if n_fallback:
        print("\nNOTE: %d sessions had no delivered_duration_s; load fell back to "
              "bytes\n      at a 100 MS/s cf32 assumption." % n_fallback)

    excl = [c for c, _, t, _, _, _ in report if t == TIER_NONE]
    if excl:
        print("\nEXCLUDED from eval (single session, spec E-014 item 4): %s"
              % ", ".join(excl))
    print("\nNow prove it:  python split_grouped.py check --assignment %s" % args.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Leakage-clean grouped splitter and its CI check "
                    "(spec section 6: E-011, E-013 entry action 3, E-014).")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="assert no group straddles a split")
    c.add_argument("--assignment", required=True,
                   help="CSV with one row per item")
    c.add_argument("--group-col", default="session_id",
                   help="the grouping key; the ONLY defensible one here is "
                        "session_id (see module docstring)")
    c.add_argument("--split-col", default="split")
    c.add_argument("--class-col", default="taxonomy_id")
    c.add_argument("--item-col", default="source_relpath",
                   help="printed as evidence for each violation")
    c.add_argument("--condition-col", default="capture_condition",
                   help="reported, never used as a split key")
    c.add_argument("--ignore-splits", default="excluded",
                   help="comma-separated splits to exclude from the check")
    c.add_argument("--max-report", type=int, default=20)
    c.add_argument("--evidence", type=int, default=2,
                   help="rows printed per (group, split) violation")
    c.set_defaults(func=cmd_check)

    m = sub.add_parser("make", help="build a session-grouped split")
    m.add_argument("--recordings", required=True, help="recordings.csv from E1")
    m.add_argument("--out", required=True)
    m.add_argument("--folds", type=int, default=3,
                   help="K for classes with >= 3 sessions; capped at n_sessions")
    m.add_argument("--seed", default="rfuav-v2",
                   help="string seed; hashed, so stable across interpreters")
    m.add_argument("--artifact-class", default="recording")
    m.add_argument("--train-name", default="train")
    m.add_argument("--test-name", default="test")
    m.add_argument("--include-foreign-roots", action="store_true",
                   help="DANGER: also split the ZJU and .dat_pic roots. They are "
                        "dropped by default (E-005/E-006 measured the ZJU tree as "
                        "int16-converted with cross-class duplicates; E-014.3 "
                        "forbids counting its classes). Class counts from a run "
                        "with this flag are not reportable as RFUAV coverage.")
    m.add_argument("--own-capture-only", action="store_true",
                   help="accepted and ignored: this is now the default")
    m.set_defaults(func=cmd_make)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
