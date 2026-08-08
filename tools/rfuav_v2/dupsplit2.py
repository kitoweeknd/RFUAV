# -*- coding: utf-8 -*-
"""Provenance-free leakage test, run 2: is a held-out frame the SAME FILE as a
training frame?

Why a second script rather than a re-run of ``dupsplit.py``.  Run 1 measured six
trees at 100.0% byte-identical and was then cut off mid-seventh: it printed three
example pairs per split, each a pair of ~130-char Windows paths, and the captured
stdout ended at 7,980 bytes with exit code 0.  The scan was fine; the transport
was the limit.  So this version writes every result to a file ON THE SERVER,
inside ``RFUAV-v2\\manifests`` (the only tree these scripts may write to --
rollback is deleting that directory), and keeps stdout to one line per tree.

It also flushes after every tree, so a truncated run is a partial result rather
than a lost one, and it appends rather than truncating an existing report.

Read-only over the originals: the only filesystem reads are the .jpg frames.

THE BUG THIS FILE CARRIED, AND WHY THE FIX CHANGES THE DEFAULT
------------------------------------------------------------------------------
Run 2's own numbers are sound, but the script that produced them was protecting
nothing.  It read ``row["size_bytes"]`` while ``e1_inventory.csv``'s header is
``path,size,mtime_iso``, so every row raised KeyError into a bare
``except Exception: sz = 0``.  Consequences, all silent:

  * every tree totalled 0 bytes, so ``if tot/1e9 > CAP_GB`` could never fire and
    ``CAP_GB = 30.0`` was inert for the whole run;
  * the 152.17 GB ``alldrone_dataset_aug`` tree was therefore hashed in full
    rather than skipped -- which is how we know it is 0.0% duplicated, a result
    the earlier ledger had written off as "over cap, unmeasured";
  * the sort key degenerated to ``(0, n_frames, tree)``, i.e. frame count
    ascending, which is exactly the line order of the report -- the bug is
    reverse-verifiable from its own output.

So the cap now defaults to OFF (``CAP_GB = None``).  Turning it on while fixing
the column name would make a re-run *skip* the largest trees and thereby destroy
evidence we already hold; the measured outcome is that hashing everything
completes.  A guard that never fires is worse than no guard, because a reader
assumes it is guarding -- so the report header states the cap's state on every
run, and a missing size column is now fatal instead of zero.

The scan body lives under ``main()`` behind a ``__main__`` guard so that
``load_frames`` can be tested off-server; importing this module used to try to
open ``E:\\...`` at line 27.
"""
import csv, io, os, sys, collections, hashlib

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
ROOT = "E:\\Drone_dataset\\RFUAV-v2\\split_images\\"
OUT = M + r"\dupsplit_p0.txt"
ROOT_MARKER = "Drone_dataset\\RFUAV-v2\\split_images\\"
TRUE = ("train", "valid", "val", "test", "tests", "eval")

# None = hash every tree, whatever its size.  See the module docstring: this is
# the setting under which run 2's numbers were actually produced.  Set a float
# to skip trees above that many GB -- and the skipped trees are then reported as
# unmeasured, which is a claim about coverage, not a detail.
CAP_GB = None

# The inventory's size column.  "size" is what E1 writes; "size_bytes" is the
# manifest spelling, accepted so a recordings.csv can be fed in too.  There is
# deliberately no third option and no fallback to zero.
_SIZE_COLUMNS = ("size", "size_bytes")
_PATH_COLUMNS = ("path", "relpath", "source_relpath")


def _pick_column(fieldnames, candidates, what):
    """Return the first of `candidates` present, or raise naming what was found.

    Raising beats defaulting: the defaulting version of this lookup is the bug
    documented in the module docstring, and it cost a run's worth of trust in a
    cap that was never enforced.
    """
    names = list(fieldnames or ())
    lowered = {n.strip().lower(): n for n in names}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise KeyError(
        "inventory has no %s column: looked for %s, header is %s"
        % (what, list(candidates), names)
    )


def load_frames(fh, root_marker="Drone_dataset\\RFUAV\\"):
    """Group the inventory's .jpg rows into tree -> split -> [(relpath, size)].

    A tree is the path prefix above the deepest split-named component, so
    ``x/y/train/CLASS/f.jpg`` and ``x/y/test/CLASS/f.jpg`` land in the same
    tree ``x/y`` under splits ``train`` and ``test``.
    """
    reader = csv.DictReader(fh)
    path_col = _pick_column(reader.fieldnames, _PATH_COLUMNS, "path")
    size_col = _pick_column(reader.fieldnames, _SIZE_COLUMNS, "size")

    T = collections.defaultdict(lambda: collections.defaultdict(list))
    for lineno, row in enumerate(reader, start=2):
        p = row.get(path_col) or ""
        if not p.lower().endswith(".jpg"):
            continue
        rel = p.split(root_marker, 1)[-1]
        parts = rel.split("\\")
        low = [s.lower() for s in parts]
        hit = [i for i, s in enumerate(low) if s in TRUE]
        if not hit:
            continue
        i = hit[-1]
        tree = "\\".join(parts[:i])
        raw = (row.get(size_col) or "").strip()
        try:
            sz = int(raw)
        except ValueError:
            # Not survivable-by-zeroing: a zeroed row understates its tree's
            # total, which is the exact mechanism that disabled the cap.
            raise ValueError(
                "line %d: %s=%r is not an integer for %s"
                % (lineno, size_col, raw, rel)
            )
        T[tree][low[i]].append((rel, sz))
    return T


def payload(splits):
    """(bytes, frames) over every split of one tree."""
    b = sum(sz for sp in splits.values() for _, sz in sp)
    n = sum(len(sp) for sp in splits.values())
    return b, n


def over_cap(total_bytes, cap_gb=CAP_GB):
    """Whether a tree of `total_bytes` is skipped under `cap_gb`.

    Split out so the "no cap" case is a tested property rather than a truthiness
    accident: ``CAP_GB = None`` must mean never skip, and ``CAP_GB = 0`` must
    mean skip everything rather than being read as "off".
    """
    if cap_gb is None:
        return False
    return total_bytes / 1e9 > cap_gb


def sha(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.digest()


def main():
    with io.open(M + r"\e1_inventory_split.csv", "r", encoding="utf-8", newline="") as f:
        T = load_frames(f, root_marker=ROOT_MARKER)

    order = sorted((payload(v) + (k,) for k, v in T.items()))

    rep = io.open(OUT, "a", encoding="utf-8")

    def say(line):
        rep.write(line + "\n")
        rep.flush()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    say("=== dupsplit2: byte-identity across split boundaries (SHA-256, full file) ===")
    say("cap: %s" % ("NONE -- every tree is hashed" if CAP_GB is None
                     else "%.1f GB -- larger trees are NOT read and count as "
                          "unmeasured" % CAP_GB))
    say("%7s %7s %7s %6s  tree" % ("train", "held", "ident", "pct"))

    g_held = g_dup = 0
    n_full = n_part = n_skip = 0
    for tot, n, tree in order:
        if "train" not in T[tree]:
            say("%7s %7s %7s %6s  %s   (no train split)" % ("-", "-", "-", "-", tree))
            continue
        if over_cap(tot):
            say("%7s %7s %7s %6s  %s   (%.1f GB > cap, NOT READ)"
                % ("-", "-", "-", "-", tree, tot / 1e9))
            n_skip += 1
            continue
        trh = {}
        for rel, sz in T[tree]["train"]:
            d = sha(ROOT + rel)
            if d:
                trh.setdefault(d, rel)
        held = dup = 0
        ex = None
        for split in sorted(T[tree]):
            if split == "train":
                continue
            for rel, sz in T[tree][split]:
                d = sha(ROOT + rel)
                if d is None:
                    continue
                held += 1
                if d in trh:
                    dup += 1
                    if ex is None:
                        ex = (rel, trh[d])
        pct = 100.0 * dup / max(1, held)
        say("%7d %7d %7d %5.1f%%  %s   (train distinct %d)"
            % (len(T[tree]["train"]), held, dup, pct, tree, len(trh)))
        if ex:
            say("        e.g. %s\n          == %s" % (ex[0], ex[1]))
        g_held += held
        g_dup += dup
        if held and dup == held:
            n_full += 1
        elif dup:
            n_part += 1

    say("")
    say("TOTAL held-out frames hashed : %d" % g_held)
    say("TOTAL byte-identical to train: %d  (%.1f%%)"
        % (g_dup, 100.0 * g_dup / max(1, g_held)))
    say("trees 100%% duplicated: %d   partially: %d   over cap (unmeasured): %d"
        % (n_full, n_part, n_skip))
    rep.close()


if __name__ == "__main__":
    main()
