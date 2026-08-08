"""split_images_grouped.py -- place spectrogram images into leakage-clean splits.

WHAT THIS FIXES (P0.3, the glue that was missing)
--------------------------------------------------
`split_grouped.py make` already builds a leakage-clean split, but it works at
the RECORDING/session level and writes a CSV -- it never touches the spectrogram
images a model actually trains on.  The image split in use is
`utils/preprocessor.py::split_images`, which does `random.shuffle` over the .jpg
files of each class directory and cuts at `int(len * ratio)`.  That is E-013
itself, one level down: frames of the same recording land on both sides, so
12,080 / 12,080 held-out frames had a train twin (100 %).

This module carries the session-level fold assignment down to the images.  It
does NOT re-decide any split; it consumes `split_grouped make` output verbatim.

HOW AN IMAGE IS MAPPED TO ITS RECORDING
---------------------------------------
`graphic/RawDataProcessor.py` (see ``generate_images``) writes each frame to
``<location>/<file>/<pack>/<file> ({frame_count}).jpg`` where ``{file}`` is the
CLASS and ``{pack}`` is the source .iq basename.  So on disk the tree is

    <images>/<class>/<iq_basename.iq>/<class> (frame).jpg   # NESTED (current)

and the recording identity lives in the RECORDING DIRECTORY (``pack``), NOT in
the frame filename (which is just the class name -- ``DJI AVATA2 (0).jpg``).
A directory is treated as a recording iff its extension-stripped name is a key
in the split map; a directory that is not (a bundled ``train/valid`` dataset dir,
a foreign-root or dropped class) is skipped and counted, never guessed.

A legacy FLAT tree also exists, where the frame filename itself is the .iq
basename::

    <images>/<class>/<iq_basename> (frame).jpg              # FLAT (legacy)

There the recording is recovered from the filename.  The frame suffix has a
LEADING SPACE (``... (5).jpg``); a source take number is attached WITHOUT one
(``...10m(3)``).  So the recovery strips only ``\\s+\\(\\d+\\)`` at the end -- it
must keep the take number, because ``...(3)`` is a different recording from
``...`` (same rule as session_stem keeping ``(N)``).  ``plan_placement`` handles
both layouts.

LEAKAGE-SAFE BY CONSTRUCTION
----------------------------
An image whose recording is not in the split file, or whose basename maps to
more than one split (a cross-class/cross-session basename collision), is NEVER
guessed into train or valid.  It goes to ``unassigned/`` and is counted in the
report.  Silent random placement is exactly the failure this module exists to
remove, so it is refused.  Images of an ``excluded`` recording (a single-session
class with no leakage-clean split) go to ``excluded/``, not into train/valid.

WRITE SCOPE
-----------
Reads the split CSV and the image tree; writes ONLY under ``--out``.  Copies
(never moves) so the source image tree, like the raw .iq, stays read-only.
"""

import argparse
import collections
import csv
import os
import re
import shutil
import sys

# The frame suffix RawDataProcessor appends: a space, then "(<int>)", at the end
# of the stem.  Anchored at end-of-stem so a mid-name take number is untouched.
_RE_FRAME_SUFFIX = re.compile(r"\s+\(\d+\)$")

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Buckets for images we refuse to place in train/valid.  Kept out of the split
# so they can never leak, but preserved (copied) so nothing is lost silently.
BUCKET_UNASSIGNED = "unassigned"   # basename not found, or collided across splits
BUCKET_EXCLUDED = "excluded"       # recording is in a single-session (unsplittable) class


def recover_iq_basename(image_filename):
    """Recover the source .iq basename (no extension) from an image filename.

    ``'DJI AVTA2-...10m(3) (5).jpg'`` -> ``'DJI AVTA2-...10m(3)'`` : the trailing
    ``' (5)'`` frame suffix is removed, the ``(3)`` take number is kept.
    """
    stem = os.path.splitext(image_filename)[0]
    return _RE_FRAME_SUFFIX.sub("", stem).strip()


def _basename_of_relpath(relpath):
    """Basename (no extension) of a manifest source_relpath (Windows or POSIX)."""
    tail = relpath.replace("/", "\\").split("\\")[-1]
    return os.path.splitext(tail)[0].strip()


def load_split_map(split_file):
    """Read a `split_grouped make` CSV into ``basename -> split``.

    Returns ``(split_map, meta)`` where ``split_map[basename]`` is one of the
    split names, ``BUCKET_EXCLUDED``, or the sentinel ``CONFLICT`` (the basename
    resolved to more than one distinct split -- unsafe to place).  ``meta`` maps
    basename -> (session_id, taxonomy_id, eval_tier) for the report.
    """
    CONFLICT = "__CONFLICT__"
    seen = collections.defaultdict(set)      # basename -> {split-or-EXCLUDED}
    meta = {}
    with open(split_file, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        need = {"source_relpath", "split"}
        missing = need - set(rdr.fieldnames or ())
        if missing:
            raise SystemExit("split file %s lacks column(s): %s"
                             % (split_file, ", ".join(sorted(missing))))
        for r in rdr:
            base = _basename_of_relpath(r["source_relpath"])
            if not base:
                continue
            split = r["split"]
            if split == "excluded":
                split = BUCKET_EXCLUDED
            seen[base].add(split)
            meta.setdefault(base, (r.get("session_id", ""),
                                   r.get("taxonomy_id", ""),
                                   r.get("eval_tier", "")))
    split_map = {}
    for base, splits in seen.items():
        real = {s for s in splits if s != BUCKET_EXCLUDED}
        if len(real) > 1:
            split_map[base] = CONFLICT
        elif real:
            split_map[base] = real.pop()
        else:
            split_map[base] = BUCKET_EXCLUDED
    return split_map, meta, CONFLICT


def _classify(base, split_map, conflict_sentinel):
    """Map a recovered recording basename to ``(dest_bucket, stats_kind)``.

    A basename absent from the split map is ``unmatched``; one that resolved to
    more than one split is a ``collision``.  Both are refused into
    ``unassigned/`` -- never guessed into train/valid.
    """
    dest = split_map.get(base)
    if dest is None:
        return BUCKET_UNASSIGNED, "unmatched"
    if dest == conflict_sentinel:
        return BUCKET_UNASSIGNED, "collision"
    if dest == BUCKET_EXCLUDED:
        return BUCKET_EXCLUDED, "excluded"
    return dest, "placed:%s" % dest


def plan_placement(images_root, split_map, conflict_sentinel):
    """Decide the destination split for every image, without any IO.

    Handles BOTH on-disk layouts (see the module docstring):

    * NESTED (current): ``<images>/<class>/<iq_basename.iq>/<class> (frame).jpg``.
      A sub-directory is a recording iff its extension-stripped name is a key in
      ``split_map``; other dirs (a bundled ``train/valid`` dataset dir, a
      foreign-root or dropped class) are skipped and their frames counted under
      ``skipped:not_in_split`` -- never guessed into a split.
    * FLAT (legacy): ``<images>/<class>/<iq_basename> (frame).jpg`` -- the
      recording is recovered from the frame filename.

    Returns ``(placements, stats)``.  ``placements`` is a list of
    ``(class_dir, rel_name, dest_split)`` where ``rel_name`` is relative to the
    class dir (it includes the recording sub-dir in the nested layout).  ``stats``
    is a Counter over outcome kinds for the report.  Pure -- unit-testable
    against a tiny image tree.
    """
    placements = []
    stats = collections.Counter()
    for class_dir in sorted(os.listdir(images_root)):
        cdir = os.path.join(images_root, class_dir)
        if not os.path.isdir(cdir):
            continue
        for entry in sorted(os.listdir(cdir)):
            epath = os.path.join(cdir, entry)
            if os.path.isdir(epath):
                # NESTED: the recording is this directory (strip its extension).
                base = os.path.splitext(entry)[0].strip()
                if base not in split_map:
                    # Not a recording of this split (foreign root, dropped class,
                    # or a bundled dataset dir like train/valid).  Count the frames
                    # so the skip is visible in the report; never place them.
                    n = sum(1 for f in os.listdir(epath)
                            if f.lower().endswith(IMAGE_EXTS))
                    if n:
                        stats["skipped:not_in_split"] += n
                    continue
                dest, kind = _classify(base, split_map, conflict_sentinel)
                for name in sorted(os.listdir(epath)):
                    if not name.lower().endswith(IMAGE_EXTS):
                        continue
                    stats[kind] += 1
                    placements.append((class_dir, os.path.join(entry, name), dest))
            elif entry.lower().endswith(IMAGE_EXTS):
                # FLAT: the frame filename carries the recording basename.
                base = recover_iq_basename(entry)
                dest, kind = _classify(base, split_map, conflict_sentinel)
                stats[kind] += 1
                placements.append((class_dir, entry, dest))
    return placements, stats


def split_images_grouped(images_root, split_file, out_root, copy=True):
    """Carry a session-level split down to the spectrogram images.

    Copies each image into ``out_root/<split>/<class>/``.  ``unassigned`` and
    ``excluded`` are real subdirectories, not train/valid, so they can never
    contaminate an evaluation.
    """
    split_map, meta, conflict = load_split_map(split_file)
    placements, stats = plan_placement(images_root, split_map, conflict)

    for class_dir, name, dest in placements:
        # ``name`` may include the recording sub-dir (nested layout), so create
        # the full parent chain, not just <out>/<split>/<class>.
        dst = os.path.join(out_root, dest, class_dir, name)
        dst_parent = os.path.dirname(dst)
        if not os.path.isdir(dst_parent):
            os.makedirs(dst_parent)
        src = os.path.join(images_root, class_dir, name)
        if copy:
            shutil.copy2(src, dst)
    return stats, len(split_map)


def _print_report(stats, n_recordings, images_root, split_file):
    total = sum(stats.values())
    print("split_images_grouped")
    print("  images from   : %s" % images_root)
    print("  split file    : %s  (%d recordings)" % (split_file, n_recordings))
    print("  images seen   : %d" % total)
    for kind in sorted(stats):
        print("  %-16s: %d" % (kind, stats[kind]))
    unassigned = stats.get("unmatched", 0) + stats.get("collision", 0)
    if unassigned:
        print("  NOTE: %d image(s) could not be placed leakage-safely and went "
              "to %s/ (never train/valid)." % (unassigned, BUCKET_UNASSIGNED))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="root of the image tree; supports both nested "
                         "<images>/<class>/<iq.iq>/<frame>.jpg and flat "
                         "<images>/<class>/<frame>.jpg")
    ap.add_argument("--split-file", required=True,
                    help="output CSV of `split_grouped.py make`")
    ap.add_argument("--out", required=True,
                    help="destination root; writes <out>/<split>/<class>/")
    ap.add_argument("--dry-run", action="store_true",
                    help="report placement counts without copying any file")
    args = ap.parse_args(argv)

    if args.dry_run:
        split_map, meta, conflict = load_split_map(args.split_file)
        _, stats = plan_placement(args.images, split_map, conflict)
        _print_report(stats, len(split_map), args.images, args.split_file)
        return 0

    stats, n_rec = split_images_grouped(args.images, args.split_file, args.out,
                                         copy=True)
    _print_report(stats, n_rec, args.images, args.split_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
