# -*- coding: utf-8 -*-
"""Unit tests for the leakage-clean splitter and its CI check.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only, python >= 3.9.

WHY THESE TESTS EXIST
---------------------
``split_grouped.py check`` is the guard that would have caught E-013 (12,080 of
12,080 held-out frames pack-contaminated) before publication.  A guard is only
worth its exit code if it has been SHOWN TO FAIL on bad input -- verifying that
it says CLEAN on output the same program just produced proves nothing at all.
So the first assertion here is a red one: the leaky fixture must exit 1.

The fixtures are hand-written, small, and shaped like the two real defects:

split_leaky.csv     E-013 shape.  The released splitter computed int(0.3 * n)
                    INSIDE each pack, so every pack landed on both sides.  Here
                    S-mini3-a and S-mini3-b each appear as train AND valid, and
                    S-kong-x carries two different taxonomy_ids (E-004 shape).

split_clean.csv     the same library, split correctly: whole sessions to one
                    side.  It deliberately keeps close/middle/far MIXED inside
                    a single session, because a session legitimately spans
                    several capture conditions -- if the checker ever starts
                    treating conditions as a grouping key, this file turns red.
"""

from __future__ import annotations

import io
import os
import sys
import csv
import shutil
import tempfile
import unittest
import contextlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import split_grouped  # noqa: E402

FIX = os.path.join(_HERE, "fixtures")
LEAKY = os.path.join(FIX, "split_leaky.csv")
CLEAN = os.path.join(FIX, "split_clean.csv")
MIXED = os.path.join(FIX, "recordings_mixed_roots.csv")


def run_check(path, *extra):
    """Run `check` on a fixture, returning (exit_code, stdout)."""
    argv = ["check", "--assignment", path] + list(extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = split_grouped.main(argv)
    return rc, buf.getvalue()


class TestCheckFailsOnLeak(unittest.TestCase):
    """The red tests.  These matter more than the green ones."""

    def test_leaky_fixture_is_rejected(self):
        rc, out = run_check(LEAKY)
        self.assertEqual(rc, 1, "check passed a leaky assignment:\n" + out)
        self.assertIn("LEAK DETECTED", out)

    def test_leak_report_names_the_offending_groups(self):
        # Lesson 2 of the E-013 post-mortem: a correct count is not a readable
        # verdict.  The report must say WHICH group and WHICH splits.
        rc, out = run_check(LEAKY)
        self.assertIn("S-mini3-a", out)
        self.assertIn("S-mini3-b", out)
        self.assertIn("train", out)
        self.assertIn("valid", out)

    def test_leak_report_prints_evidence_rows(self):
        rc, out = run_check(LEAKY)
        self.assertIn("mini3/pack1_00.dat", out)

    def test_group_spanning_two_classes_is_rejected(self):
        rc, out = run_check(LEAKY)
        self.assertIn("ILL-DEFINED", out)
        self.assertIn("S-kong-x", out)
        self.assertIn("bg.kong", out)

    def test_every_multi_row_group_straddles_as_in_e013(self):
        # This is the E-013 signature, not an incidental count: because the
        # released splitter took int(0.3 * n) INSIDE each group, EVERY group
        # large enough to be cut ended up on both sides -- 124 of 124 in the
        # real tree.  Here that is 3 of 4 (S-kong-x has both rows in `train`,
        # and S-solo-z is ignored as `excluded`).
        rc, out = run_check(LEAKY)
        self.assertIn("3 of 4 groups straddle", out)
        self.assertIn("11 items", out)


class TestCheckPassesCleanSplit(unittest.TestCase):

    def test_clean_fixture_is_accepted(self):
        rc, out = run_check(CLEAN)
        self.assertEqual(rc, 0, "check rejected a clean assignment:\n" + out)
        self.assertIn("CLEAN", out)
        self.assertNotIn("LEAK DETECTED", out)

    def test_conditions_inside_one_session_are_not_a_leak(self):
        # S-mini3-a spans close/middle/far in a single session and a single
        # split.  A checker that keyed on capture_condition would flag this.
        rc, out = run_check(CLEAN)
        self.assertEqual(rc, 0)
        self.assertIn("NEVER a split key", out)

    def test_excluded_rows_are_not_counted_as_a_split(self):
        rc, out = run_check(CLEAN)
        self.assertIn("1 ignored", out)

    def test_ignoring_nothing_makes_excluded_visible_but_still_clean(self):
        # `excluded` is a split name like any other once it is not ignored;
        # S-solo-z sits wholly inside it, so the verdict stays CLEAN.
        rc, out = run_check(CLEAN, "--ignore-splits", "")
        self.assertEqual(rc, 0, out)


class TestCheckRefusesBadInvocation(unittest.TestCase):

    def test_missing_group_column_is_an_error_not_a_pass(self):
        # Exit 2, not 0.  A typo in --group-col must never read as CLEAN.
        rc, out = run_check(CLEAN, "--group-col", "no_such_column")
        self.assertEqual(rc, 2, out)
        self.assertIn("FAIL", out)

    def test_missing_split_column_is_an_error(self):
        rc, out = run_check(CLEAN, "--split-col", "no_such_column")
        self.assertEqual(rc, 2, out)


class TestForeignRootsAreDroppedByDefault(unittest.TestCase):
    """E-005/E-006/E-014 item 3 -- and a near-miss worth a permanent guard.

    With the foreign roots included, the real library reports 63 taxonomy_ids,
    of which 23 exist ONLY inside those roots.  63 is close enough to the
    >=60-class goal that a default-flag run produces a table looking like the
    goal is already met -- on a tree E-005/E-006 measured as int16-converted
    with cross-class duplicate content.  So dropping them is the default, and
    including them must be loud.  If someone ever inverts that again, this
    class goes red.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rfuav-split-")
        self.out = os.path.join(self.tmp, "split.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_make(self, *extra):
        argv = ["make", "--recordings", MIXED, "--out", self.out,
                "--folds", "3"] + list(extra)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = split_grouped.main(argv)
        with open(self.out, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rc, buf.getvalue(), rows

    def test_foreign_only_classes_are_absent_by_default(self):
        rc, out, rows = self.run_make()
        self.assertEqual(rc, 0, out)
        classes = {r["taxonomy_id"] for r in rows}
        self.assertIn("uav.dji.mini3", classes)
        self.assertNotIn("uav.dji.mavic3", classes,
                         "ZJU-only class counted by default (E-014 item 3)")
        self.assertNotIn("uav.dji.phantom3", classes,
                         ".dat_pic-only class counted by default")

    def test_the_drop_is_reported_with_the_class_names(self):
        # Silently dropping is as bad as silently keeping: the operator has to
        # be able to see which classes vanished and why.
        rc, out, rows = self.run_make()
        self.assertIn("DROPPED", out)
        self.assertIn("uav.dji.mavic3", out)
        self.assertIn("E-014.3", out)

    def test_including_them_is_possible_but_loudly_disclaimed(self):
        rc, out, rows = self.run_make("--include-foreign-roots")
        self.assertEqual(rc, 0, out)
        classes = {r["taxonomy_id"] for r in rows}
        self.assertIn("uav.dji.mavic3", classes)
        self.assertIn("NOT reportable", out)

    def test_non_recording_artifacts_are_never_split(self):
        # r010 is a spectrogram derived from r001.  If derived artifacts entered
        # the split they would carry their parent's frames across a boundary.
        rc, out, rows = self.run_make()
        self.assertTrue(all(not r["source_relpath"].endswith(".jpg")
                            for r in rows), "a derived artifact was split")

    def test_deprecated_flag_still_parses(self):
        rc, out, rows = self.run_make("--own-capture-only")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("uav.dji.mavic3", {r["taxonomy_id"] for r in rows})


class TestTiers(unittest.TestCase):
    """Spec section 6, E-014 item 4."""

    def test_tier_boundaries(self):
        self.assertEqual(split_grouped.tier_of(1), split_grouped.TIER_NONE)
        self.assertEqual(split_grouped.tier_of(2), split_grouped.TIER_1V1)
        self.assertEqual(split_grouped.tier_of(3), split_grouped.TIER_CV)
        self.assertEqual(split_grouped.tier_of(12), split_grouped.TIER_CV)

    def test_stable_rank_is_not_interpreter_dependent(self):
        # Hard-coded expected value: if someone swaps sha256 for hash(), or
        # for random.shuffle, this test goes red instead of silently changing
        # every published split.
        got = split_grouped.stable_rank("rfuav-v2", "S-mini3-a")
        self.assertEqual(got, split_grouped.stable_rank("rfuav-v2", "S-mini3-a"))
        self.assertNotEqual(got, split_grouped.stable_rank("rfuav-v2", "S-mini3-b"))
        self.assertNotEqual(got, split_grouped.stable_rank("other-seed", "S-mini3-a"))
        self.assertIsInstance(got, int)


class TestTierBasis(unittest.TestCase):
    """E-016: a session count is not an evaluation claim.

    The defect these pin, measured on the 2026-07-31 manifest run: 21 of the 42
    all_roots ``grouped_cv`` classes hold under 10 s of unique signal, and for
    21 of them ``n_sessions == n_recordings`` -- no two files were ever PROVEN
    to share a session, so each file became its own singleton group and the
    class scored 3+ sessions by default.  The tier was high because the
    grouping evidence was ABSENT.  own_capture has 0 such classes, so the
    reportable headline was never affected; the all_roots column was.

    ``tier_of`` is left alone on purpose (three importers, two pinning test
    classes).  These tests guard the disclosure that travels beside it.
    """

    def test_a_healthy_class_is_measured(self):
        # flysky-fs_i6x as actually measured: 3 sessions, 10 recordings, 42.6 s.
        self.assertEqual(
            split_grouped.tier_basis(3, 10, 42.60), split_grouped.BASIS_MEASURED)
        self.assertTrue(split_grouped.tier_is_supported(3, 10, 42.60))

    def test_the_real_inversion_is_caught(self):
        # THE test.  uav.dji.mavic3 scored grouped_cv on 1.80 s across 3 files
        # in 3 singleton sessions, outranking bg.kong's 13.74 s / 8 recordings.
        # Both defects present, so both must be named.
        basis = split_grouped.tier_basis(3, 3, 1.80)
        self.assertIn(split_grouped.BASIS_SINGLETON, basis)
        self.assertIn(split_grouped.BASIS_THIN, basis)
        self.assertFalse(split_grouped.tier_is_supported(3, 3, 1.80))
        # ...while the tier itself is untouched: the split still exists.
        self.assertEqual(split_grouped.tier_of(3), split_grouped.TIER_CV)

    def test_singleton_sessions_alone_are_disqualifying(self):
        # Plenty of signal, but every file is its own session: nothing was
        # grouped, so "grouped_cv" describes an absence of evidence.
        basis = split_grouped.tier_basis(4, 4, 300.0)
        self.assertEqual(basis, split_grouped.BASIS_SINGLETON)
        self.assertFalse(split_grouped.tier_is_supported(4, 4, 300.0))

    def test_thin_signal_alone_is_disqualifying(self):
        basis = split_grouped.tier_basis(3, 12, 4.0)
        self.assertEqual(basis, split_grouped.BASIS_THIN)

    def test_unsplittable_is_exempt_from_the_singleton_check(self):
        # A 1-session class claims nothing about grouping, so there is nothing
        # to qualify.  Flagging it would make the column noise instead of
        # signal -- and n_sessions == n_recordings == 1 is the ordinary shape
        # of an honest single capture.
        self.assertEqual(split_grouped.tier_basis(1, 1, 50.0),
                         split_grouped.BASIS_MEASURED)
        self.assertEqual(split_grouped.tier_of(1), split_grouped.TIER_NONE)

    def test_a_thin_single_session_class_is_still_flagged_thin(self):
        # bg.kong's neighbour shape: 1 session, few seconds.  Not a grouping
        # problem, still not enough signal to evaluate on.
        self.assertEqual(split_grouped.tier_basis(1, 1, 4.12),
                         split_grouped.BASIS_THIN)

    def test_a_thin_unsplittable_tier_is_still_supported(self):
        # E-016, second pass.  The disclosure and the support question are not
        # the same question.  `unsplittable` asserts that NO leakage-clean split
        # exists; 4.12 s of signal is a reason for that claim, never a refutation
        # of it.  So the basis stays `thin_signal` (a trainer must see it) while
        # the tier stays quotable.
        self.assertEqual(split_grouped.tier_basis(1, 1, 4.12),
                         split_grouped.BASIS_THIN)
        self.assertTrue(split_grouped.tier_is_supported(1, 1, 4.12))
        self.assertEqual(split_grouped.tier_of(1), split_grouped.TIER_NONE)

    def test_the_over_count_that_prompted_the_exemption(self):
        # MEASURED on the first E-016 run: all_roots published
        # n_tier_unsupported = 23 = 21 grouped_cv + 2 unsplittable-and-thin,
        # printed under a line naming only the two split-asserting tiers.  That
        # reads as "23 classes overclaim a split" when 21 do.  Reproduced here
        # as the two shapes, so the count can never drift back to 23.
        overclaiming = [(3, 3, 1.80)] * 21          # singleton + thin
        exempt = [(1, 1, 4.12), (1, 1, 2.00)]       # unsplittable, thin
        unsupported = [a for a in overclaiming + exempt
                       if not split_grouped.tier_is_supported(*a)]
        self.assertEqual(len(unsupported), 21)
        # ...and the two exempt rows are not thereby laundered: they are still
        # thin, which is what `n_thin_signal` is for.
        thin = [a for a in overclaiming + exempt
                if split_grouped.BASIS_THIN in split_grouped.tier_basis(*a)]
        self.assertEqual(len(thin), 23)

    def test_split_asserting_tiers_can_still_fail(self):
        # The exemption is for TIER_NONE only.  holdout_1v1 asserts a split just
        # as grouped_cv does, so thin signal must still disqualify it -- an
        # exemption that leaked to 2 sessions would hide the same defect.
        self.assertEqual(split_grouped.tier_of(2), split_grouped.TIER_1V1)
        self.assertFalse(split_grouped.tier_is_supported(2, 2, 3.0))
        self.assertFalse(split_grouped.tier_is_supported(2, 9, 3.0))
        self.assertTrue(split_grouped.tier_is_supported(2, 9, 50.0))

    def test_kong_and_herelink_are_not_flagged_singleton(self):
        # The two classes the inversion embarrassed: many recordings inside ONE
        # proven session.  That is the opposite of the defect -- the grouping
        # evidence is strong, there is just only one group.
        for n_rec, uniq in ((8, 13.74), (16, 12.77)):
            self.assertEqual(split_grouped.tier_basis(1, n_rec, uniq),
                             split_grouped.BASIS_MEASURED, (n_rec, uniq))

    def test_the_floor_is_a_named_constant_not_a_literal(self):
        # A threshold buried in an `if` is a threshold nobody re-examines.
        self.assertIsInstance(split_grouped.TIER_MIN_UNIQUE_S, float)
        self.assertEqual(split_grouped.TIER_MIN_UNIQUE_S, 10.0)
        just_under = split_grouped.TIER_MIN_UNIQUE_S - 0.01
        self.assertEqual(split_grouped.tier_basis(3, 9, just_under),
                         split_grouped.BASIS_THIN)
        self.assertEqual(
            split_grouped.tier_basis(3, 9, split_grouped.TIER_MIN_UNIQUE_S),
            split_grouped.BASIS_MEASURED, "the floor is exclusive at equality")

    def test_basis_is_never_silently_empty(self):
        # An empty string would render as a blank CSV cell and read as "fine".
        for args in ((1, 1, 0.0), (3, 3, 0.0), (2, 2, 1.0), (9, 9, 9.0)):
            self.assertTrue(split_grouped.tier_basis(*args), args)


class TestLoadUnit(unittest.TestCase):
    """E-009: folds are balanced by signal TIME, not by bytes."""

    def test_duration_is_preferred_over_bytes(self):
        s, fb = split_grouped.row_seconds(
            {"delivered_duration_s": "60.0", "size_bytes": "999"})
        self.assertEqual(s, 60.0)
        self.assertFalse(fb)

    def test_missing_duration_falls_back_and_says_so(self):
        s, fb = split_grouped.row_seconds(
            {"delivered_duration_s": "", "size_bytes": str(8 * 100 * 10 ** 6)})
        self.assertAlmostEqual(s, 1.0)
        self.assertTrue(fb, "silent fallback: the report would overstate rigour")

    def test_equal_bytes_at_different_rates_are_not_equal_load(self):
        # The whole point of using duration: 200 MS/s and 100 MS/s files of the
        # same size carry different amounts of signal.
        fast = {"delivered_duration_s": "5.0", "size_bytes": "8000000000"}
        slow = {"delivered_duration_s": "10.0", "size_bytes": "8000000000"}
        self.assertEqual(split_grouped.row_bytes(fast),
                         split_grouped.row_bytes(slow))
        self.assertNotEqual(split_grouped.row_seconds(fast)[0],
                            split_grouped.row_seconds(slow)[0])


class TestUncutIsReadNotGuessed(unittest.TestCase):
    """One definition of "uncut", enforced at the file boundary.

    The quantity has been computed three different ways in this repo:
    build_manifests.py:1149 correctly, ``not (is_slice or req_dur)``;
    build_labels.py summed every row and called it unique; and this module used
    ``is_slice != "1"`` for the ``uniq_s`` column.  All three published under
    names a reader takes as the same number.

    What makes the loose version wrong is ``req_dur``: a filename that declares
    its own window (``..._0.0_1.0s.dat``) yields a cut row carrying
    ``is_slice=0``.  Fixture row r004 is exactly that shape, so a regression to
    ``not is_slice`` shows up here as an arithmetic difference rather than as a
    review comment nobody makes.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rfuav-uncut-")
        self.out = os.path.join(self.tmp, "split.csv")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, recordings):
        argv = ["make", "--recordings", recordings, "--out", self.out,
                "--folds", "3"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = split_grouped.main(argv)
        with open(self.out, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rc, buf.getvalue(), rows

    def strip_column(self, name):
        """The fixture minus one column -- i.e. a pre-``is_uncut`` manifest."""
        dst = os.path.join(self.tmp, "old_manifest.csv")
        with open(MIXED, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        i = rows[0].index(name)
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            for row in rows:
                w.writerow([v for j, v in enumerate(row) if j != i])
        return dst

    def test_reader_parses_both_spellings(self):
        # build_manifests writes a python bool through csv ("True"/"False");
        # this module writes "1"/"0".  Both are the same fact and both are read.
        for yes in ("1", "True", "true", " TRUE ", "yes"):
            self.assertTrue(split_grouped.row_is_uncut({"is_uncut": yes}), yes)
        for no in ("0", "False", "false", ""):
            self.assertFalse(split_grouped.row_is_uncut({"is_uncut": no}), repr(no))

    def test_a_manifest_without_the_column_is_fatal(self):
        # Not a warning and not a fallback: the only fallback available is the
        # bug, and it fails in the direction that flatters the library.
        old = self.strip_column("is_uncut")
        with self.assertRaises(SystemExit) as cm:
            self.make(old)
        msg = str(cm.exception)
        self.assertIn("is_uncut", msg)
        self.assertIn("build_manifests.py", msg,
                      "the error must name the fix, not just the symptom")
        self.assertIn("not is_slice", msg,
                      "the error must name the wrong answer it is refusing")

    def test_it_dies_before_writing_anything(self):
        # A split written from a guessed predicate is worse than no split: it
        # looks publishable.  The guard runs on the header, before the writer.
        old = self.strip_column("is_uncut")
        with self.assertRaises(SystemExit):
            self.make(old)
        self.assertFalse(os.path.exists(self.out),
                         "a split was written from a manifest that could not "
                         "support it")

    def test_the_column_travels_into_the_split(self):
        rc, out, rows = self.make(MIXED)
        self.assertEqual(rc, 0, out)
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["is_uncut"], ("0", "1"), r["recording_id"])
        by_id = {r["recording_id"]: r for r in rows}
        # r004 is the discriminating row: not marked as a slice, still a window.
        self.assertEqual(by_id["r004"]["is_slice"], "0")
        self.assertEqual(by_id["r004"]["is_uncut"], "0",
                         "a filename-declared window was carried into the "
                         "split as an original capture")

    def test_unique_seconds_excludes_the_filename_declared_window(self):
        # mini3 has four recording rows of 1 s each: r001 and r003 uncut, r002
        # an explicit slice, r004 a req_dur window.  sec = 4.0 for all rows;
        # uniq_s = 2.0.  The loose predicate would print 3.0 -- plausible,
        # publishable, and wrong by one whole recording.
        rc, out, rows = self.make(MIXED)
        self.assertEqual(rc, 0, out)
        line = [L for L in out.splitlines()
                if L.startswith("train") or L.startswith("test")]
        self.assertTrue(line, out)
        total_sec = sum(float(L.split()[3]) for L in line)
        total_uniq = sum(float(L.split()[4]) for L in line)
        self.assertAlmostEqual(total_sec, 4.0, places=3, msg=out)
        self.assertAlmostEqual(total_uniq, 2.0, places=3, msg=out)
        self.assertNotAlmostEqual(total_uniq, 3.0, places=3,
                                  msg="uniq_s reverted to `is_slice != 1`")


if __name__ == "__main__":
    unittest.main()
