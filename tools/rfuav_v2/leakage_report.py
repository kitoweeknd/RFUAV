"""leakage_report.py -- the naive-vs-grouped honesty report (P0.4).

WHY THIS EXISTS
---------------
Three numbers must always travel together, or a leakage-clean split still reads
as dishonest:

  (1) NAIVE leakage   -- byte-identity of frames across the split boundary.
      This is E-013 (100 %) and is measured by ``dupsplit2.py`` over an IMAGE
      inventory.  After ``split_images_grouped.py`` it must fall to ~0.
  (2) GROUPED leakage -- a session_id appearing in more than one split.
      This is measured by ``split_grouped.py check`` and must be exactly 0.
  (3) FILES / SESSION -- the shape gap-58 warns about: a ``grouped_cv`` tier can
      hide that 159 / 231 sessions are singletons (1.00 files/session), i.e. the
      "group" that made the split clean is one file, so the tier is thin.

`check` prints (2) and `dupsplit2` prints (1), but nothing prints (3) beside
them, and (3) is exactly what makes a grouped number honest instead of merely
clean.  This module prints all three and REFUSES to omit any: if the image
inventory for (1) is absent it says so loudly rather than leaving a blank.

It re-uses `split_grouped` for every tier decision (never re-implements a tier)
and `dupsplit2` for the byte-identity pass.  Reads only; writes nothing.
"""

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_grouped as sg   # noqa: E402  -- tier_of / tier_basis / row_seconds ...


def _sessions_by_class(rows):
    """class -> session_id -> [recording rows], excluding rows with no group."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        cls = (r.get("taxonomy_id") or "").strip()
        sid = (r.get("session_id") or "").strip()
        if cls and sid:
            per[cls][sid].append(r)
    return per


def grouped_leakage(rows, group_col="session_id", split_col="split",
                    ignore=("excluded",)):
    """Count sessions that straddle a split boundary (must be 0).

    Same rule as ``split_grouped.cmd_check`` violation 1, computed here so the
    report is self-contained; ``check`` remains the CI primitive with evidence.
    """
    per = collections.defaultdict(set)
    for r in rows:
        g = (r.get(group_col) or "").strip()
        s = (r.get(split_col) or "").strip()
        if not g or s in ignore:
            continue
        per[g].add(s)
    straddling = [g for g, ss in per.items() if len(ss) > 1]
    return len(straddling), len(per), straddling


def files_per_session_stats(per_class):
    """Distribution of recordings-per-session across all sessions."""
    counts = []
    for sess in per_class.values():
        for rs in sess.values():
            counts.append(len(rs))
    counts.sort()
    n = len(counts)
    singletons = sum(1 for c in counts if c == 1)
    median = counts[n // 2] if n else 0
    return {
        "n_sessions": n,
        "singletons": singletons,
        "singleton_frac": (singletons / n) if n else 0.0,
        "median": median,
        "max": counts[-1] if n else 0,
    }


def tier_summary(per_class):
    """Per-class tier + whether it may be quoted as an evaluation claim.

    Every tier decision is delegated to `split_grouped`; unique_seconds mirrors
    `build_labels` (uncut recordings only -- a slice's seconds are not new).
    """
    rows_out = []
    n_supported = 0
    n_unsupported = 0
    for cls in sorted(per_class):
        sess = per_class[cls]
        n_sessions = len(sess)
        recs = [r for rs in sess.values() for r in rs]
        n_recordings = len(recs)
        unique_seconds = sum(sg.row_seconds(r)[0] for r in recs
                             if sg.row_is_uncut(r))
        tier = sg.tier_of(n_sessions)
        basis = sg.tier_basis(n_sessions, n_recordings, unique_seconds)
        supported = sg.tier_is_supported(n_sessions, n_recordings, unique_seconds)
        n_supported += bool(supported)
        n_unsupported += (not supported)
        rows_out.append({
            "class": cls, "n_sessions": n_sessions, "n_recordings": n_recordings,
            "unique_seconds": round(unique_seconds, 2), "eval_tier": tier,
            "tier_basis": basis, "supported": supported,
        })
    return rows_out, n_supported, n_unsupported


def _byte_identity_section(image_inventory, cap_gb):
    """(1) The naive byte-identity number is dupsplit2's, not this report's.

    dupsplit2 computes ident%% by SHA-256'ing every held-out .jpg and every train
    .jpg and counting held-out frames whose bytes match a train frame.  That
    requires reading the real image files (``E:\\...\\*.jpg``) and cannot be done
    from a CSV on another machine.  So this report NEVER prints an ident%% figure
    of its own -- printing ``payload()`` sizes here (total bytes/frames, not
    matches) under a "leakage" heading is the precise mislabel this codebase
    fights.  What it CAN do, given the inventory, is a pre-flight: which trees
    dupsplit2 will hash, and (at a cap) which it would skip as unmeasured.
    """
    head = ("  NAIVE (byte-identity) leakage is measured by dupsplit2.py on the\n"
            "  box (it hashes the real .jpg frames).  A leakage-clean image split\n"
            "  must report ident% ~ 0.  This report does not compute it.")
    if not image_inventory:
        return head + ("\n  Pre-flight: NONE (no --image-inventory given).\n"
                       "    -> run dupsplit2.py on the box for the ident% figure.")
    import dupsplit2  # noqa: E402
    with open(image_inventory, "r", encoding="utf-8-sig", newline="") as fh:
        trees = dupsplit2.load_frames(fh)
    lines = [head, "  Pre-flight (what dupsplit2 will scan; sizes are payload, "
             "NOT leakage):"]
    hashed = skipped = 0
    for tree, splits in sorted(trees.items()):
        tb, tn = dupsplit2.payload(splits)
        no_train = "train" not in splits
        over = dupsplit2.over_cap(tb, cap_gb) if cap_gb is not None else False
        tag = (" (no train split -> not scanned)" if no_train
               else " (> %.1f GB cap -> UNMEASURED)" % cap_gb if over else "")
        if no_train or over:
            skipped += 1
        else:
            hashed += 1
        lines.append("    %-40s  %d frames / %.2f GB%s"
                     % (tree, tn, tb / 1024.0**3, tag))
    lines.append("    trees dupsplit2 would hash: %d   would skip: %d"
                 % (hashed, skipped))
    return "\n".join(lines)


def report(assignment, image_inventory=None, cap_gb=None):
    with open(assignment, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    per_class = _sessions_by_class(rows)

    n_straddle, n_groups, straddling = grouped_leakage(rows)
    fps = files_per_session_stats(per_class)
    tier_rows, n_sup, n_unsup = tier_summary(per_class)

    print("=" * 74)
    print("LEAKAGE REPORT (naive-vs-grouped) -- %s" % assignment)
    print("=" * 74)

    print("\n(1) " + _byte_identity_section(image_inventory, cap_gb).lstrip())

    print("\n(2) GROUPED leakage (session in >1 split):")
    if n_straddle == 0:
        print("  OK: 0 of %d sessions straddle a split boundary." % n_groups)
    else:
        print("  LEAK: %d of %d sessions straddle a split boundary: %s%s"
              % (n_straddle, n_groups, ", ".join(sorted(straddling)[:10]),
                 " ..." if n_straddle > 10 else ""))

    print("\n(3) FILES / SESSION (gap-58 -- a grouped tier can hide singletons):")
    print("  sessions        : %d" % fps["n_sessions"])
    print("  singleton (1 file) sessions: %d  (%.1f%%)"
          % (fps["singletons"], 100.0 * fps["singleton_frac"]))
    print("  median files/session       : %d" % fps["median"])
    print("  max files/session          : %d" % fps["max"])

    print("\n(4) EVAL-TIER SUPPORT (every tier decision delegated to split_grouped):")
    print("  %-24s %8s %8s %10s  %-14s %-18s %s"
          % ("class", "sessions", "recs", "uniq_s", "eval_tier", "tier_basis", "quotable"))
    for tr in tier_rows:
        print("  %-24s %8d %8d %10.2f  %-14s %-18s %s"
              % (tr["class"][:24], tr["n_sessions"], tr["n_recordings"],
                 tr["unique_seconds"], tr["eval_tier"], tr["tier_basis"],
                 "yes" if tr["supported"] else "NO"))
    print("  ----")
    print("  quotable classes : %d    tier-UNSUPPORTED (thin/none): %d"
          % (n_sup, n_unsup))

    print("\nHONESTY NOTE: report all four together.  A clean (2) with a high "
          "singleton\n  fraction in (3) or many NO rows in (4) is 'leakage-clean "
          "but thin', not\n  'validated'.  Never quote an accuracy without this "
          "block beside it.")
    return 0 if n_straddle == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assignment", required=True,
                    help="output CSV of `split_grouped.py make`")
    ap.add_argument("--image-inventory", default=None,
                    help="optional .jpg inventory CSV for the naive byte-identity "
                         "pass (dupsplit2 format: path,size[,mtime])")
    ap.add_argument("--cap-gb", type=float, default=None,
                    help="dupsplit2 tree size cap (passthrough)")
    args = ap.parse_args(argv)
    return report(args.assignment, args.image_inventory, args.cap_gb)


if __name__ == "__main__":
    sys.exit(main())
