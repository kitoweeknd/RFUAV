# -*- coding: utf-8 -*-
"""Unit tests for split_images_grouped (P0.3): image -> session -> fold.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only.

WHY THESE TESTS EXIST
---------------------
`split_grouped make` builds a leakage-clean split at the recording level but
never touches the .jpg frames a model trains on; the image split in use was the
leaky `preprocessor.split_images` (random.shuffle per class -> E-013, 100 %).
This adapter carries the session-level fold down to the images.  Its whole value
is that it NEVER guesses: an image whose recording isn't in the split, or whose
basename collides across splits, goes to `unassigned/`, never train/valid.
These tests pin (a) provenance recovery keeps the take number but strips the
frame suffix, (b) a cross-split basename collision is refused, (c) an excluded
recording is bucketed as excluded, (d) an orphan image is bucketed unassigned.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                        # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import split_images_grouped as sig  # noqa: E402


class TestRecoverBasename(unittest.TestCase):

    def test_frame_suffix_stripped_take_number_kept(self):
        # trailing " (5)" is the frame index (leading space) -> stripped;
        # "(3)" is a take number (no space) -> kept.
        self.assertEqual(
            sig.recover_iq_basename("DJI AVTA2-85db_5765m_100m_10m(3) (5).jpg"),
            "DJI AVTA2-85db_5765m_100m_10m(3)")

    def test_plain_frame_suffix(self):
        self.assertEqual(sig.recover_iq_basename("foo (12).png"), "foo")

    def test_no_suffix_is_identity(self):
        self.assertEqual(sig.recover_iq_basename("bar.jpg"), "bar")

    def test_take_number_without_space_is_kept(self):
        self.assertEqual(sig.recover_iq_basename("baz(3).jpg"), "baz(3)")


def _write_split_csv(path, rows, header="source_relpath,split,session_id,"
                                        "taxonomy_id,eval_tier"):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(r) + "\n")


class TestLoadSplitMap(unittest.TestCase):

    def test_basename_maps_to_its_split(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "split.csv")
            _write_split_csv(csv_path, [
                ("root\\a\\recA.iq", "train", "sA", "uav.a", "grouped_cv"),
                ("root\\b\\recB.iq", "valid", "sB", "uav.a", "grouped_cv"),
            ])
            m, meta, conflict = sig.load_split_map(csv_path)
            self.assertEqual(m["recA"], "train")
            self.assertEqual(m["recB"], "valid")

    def test_excluded_recording_bucketed_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "split.csv")
            _write_split_csv(csv_path, [
                ("root\\c\\recC.iq", "excluded", "sC", "uav.c", "unsplittable"),
            ])
            m, meta, conflict = sig.load_split_map(csv_path)
            self.assertEqual(m["recC"], sig.BUCKET_EXCLUDED)

    def test_basename_in_two_splits_is_a_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "split.csv")
            _write_split_csv(csv_path, [
                ("rootX\\dup.iq", "train", "s1", "uav.a", "grouped_cv"),
                ("rootY\\dup.iq", "valid", "s2", "uav.b", "grouped_cv"),
            ])
            m, meta, conflict = sig.load_split_map(csv_path)
            self.assertEqual(m["dup"], conflict)


class TestPlanPlacement(unittest.TestCase):

    def _tree(self, d, files):
        """files: {class_dir: [image_filename, ...]}"""
        for cls, names in files.items():
            cdir = os.path.join(d, cls)
            os.makedirs(cdir)
            for n in names:
                with open(os.path.join(cdir, n), "w") as fh:
                    fh.write("x")

    def test_buckets_cover_placed_excluded_collision_and_orphan(self):
        with tempfile.TemporaryDirectory() as d:
            images = os.path.join(d, "images")
            os.makedirs(images)
            self._tree(images, {
                "uav.a": ["recA (1).jpg", "recA (2).jpg", "recB (1).jpg"],
                "uav.c": ["recC (1).jpg"],
                "uav.x": ["dup (1).jpg", "orphan (1).jpg"],
            })
            split_map = {
                "recA": "train", "recB": "valid",
                "recC": sig.BUCKET_EXCLUDED,
                "dup": "__CONFLICT__",
            }
            placements, stats = sig.plan_placement(images, split_map, "__CONFLICT__")
            self.assertEqual(stats["placed:train"], 2)   # recA (1),(2)
            self.assertEqual(stats["placed:valid"], 1)   # recB
            self.assertEqual(stats["excluded"], 1)       # recC
            self.assertEqual(stats["collision"], 1)      # dup
            self.assertEqual(stats["unmatched"], 1)      # orphan
            # collision + unmatched are both refused into the unassigned bucket
            dests = {(c, n): dest for c, n, dest in placements}
            self.assertEqual(dests[("uav.x", "dup (1).jpg")], sig.BUCKET_UNASSIGNED)
            self.assertEqual(dests[("uav.x", "orphan (1).jpg")], sig.BUCKET_UNASSIGNED)

    def test_nested_recording_dir_layout(self):
        # Real RawDataProcessor tree: <class>/<iq.iq>/<class> (frame).jpg.
        # The recording identity is the .iq DIRECTORY, not the frame filename
        # (which is just the class name).  A bundled non-recording dir whose
        # stripped name is not a split key must be skipped, never guessed.
        with tempfile.TemporaryDirectory() as d:
            images = os.path.join(d, "images")
            rec = os.path.join(images, "DJI AVATA2",
                               "DJI AVTA2-pack1-x(3).iq")
            os.makedirs(rec)
            for fr in ("DJI AVATA2 (0).jpg", "DJI AVATA2 (1).jpg"):
                with open(os.path.join(rec, fr), "w") as fh:
                    fh.write("x")
            other = os.path.join(images, "DJI AVATA2", "train")  # not a recording
            os.makedirs(other)
            with open(os.path.join(other, "DJI AVATA2 (9).jpg"), "w") as fh:
                fh.write("x")
            split_map = {"DJI AVTA2-pack1-x(3)": "train"}
            placements, stats = sig.plan_placement(images, split_map, "__C__")
            self.assertEqual(stats["placed:train"], 2)      # the 2 recording frames
            self.assertEqual(stats["skipped:not_in_split"], 1)  # the bundled dir
            # frames are placed UNDER the recording sub-dir, class preserved
            placed = {n for c, n, dest in placements if dest == "train"}
            self.assertTrue(all(n.startswith("DJI AVTA2-pack1-x(3).iq")
                                for n in placed))

    def test_nested_filename_is_class_not_recording(self):
        # Guard the exact bug that motivated the fix: recovering the recording
        # from the FILENAME here yields the class ("DJI AVATA2"), which is NOT a
        # split key -> the frames would wrongly go unmatched.  The dir-based
        # recovery must place them instead.
        with tempfile.TemporaryDirectory() as d:
            images = os.path.join(d, "images")
            rec = os.path.join(images, "DJI AVATA2", "DJI AVTA2-pack1-x(3).iq")
            os.makedirs(rec)
            with open(os.path.join(rec, "DJI AVATA2 (0).jpg"), "w") as fh:
                fh.write("x")
            # class name present as a key would be a false match; it is absent.
            split_map = {"DJI AVTA2-pack1-x(3)": "valid"}
            _, stats = sig.plan_placement(images, split_map, "__C__")
            self.assertEqual(stats["placed:valid"], 1)
            self.assertEqual(stats.get("unmatched", 0), 0)


if __name__ == "__main__":
    unittest.main()
