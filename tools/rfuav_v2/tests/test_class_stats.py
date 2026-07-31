# -*- coding: utf-8 -*-
"""Unit tests for the class-scale counters in ``build_manifests.py``.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only, python >= 3.9.

WHY THESE TESTS EXIST
---------------------
``class_stats`` produces the number the datasheet will print after the words
"RFUAV v2 contains".  Three ways that number can be wrong, and each gets a test
class here:

1. THE INCLUSION RULE DRIFTS.  ``build_labels.py::evidence`` counts a row iff
   ``(artifact_class or "recording") == "recording"``.  build_manifests.py now
   counts rows too, and if the two predicates ever disagree, labels.json and
   classes.csv publish different class counts for the same library while both
   look internally consistent.  ``TestInclusionRule`` pins the predicate --
   including the empty-string default, which is the half that would silently
   drop every row whose artifact_class column happens to be blank.

2. THE DENOMINATOR GOES MISSING.  63 observed classes and 40 own-capture
   classes are both true of this library; only 40 is reportable, because the
   other 23 live solely in the roots E-005/E-006 measured as int16-converted
   with cross-class duplicate content.  ``TestBothScopes`` asserts the two
   scopes are computed separately and that dropping the foreign roots strictly
   shrinks the count -- if they ever coincide, the scoping stopped working.

3. UNIQUE SECONDS STOP BEING UNIQUE.  E-015 defines ``unique_seconds`` over
   UNCUT ROWS ONLY, because 531 of 700 own-capture rows are re-cut windows of
   signal their parent already counts.  ``build_labels.py::evidence`` once summed
   ``delivered_duration_s`` over every recording row and published the total as
   ``unique_seconds`` -- so the name has already been attached to a
   double-counting sum inside this repo, and it looked correct for as long as
   nobody checked it against a class whose re-cuts outnumbered its originals.
   That is fixed (both files now read the exported ``is_uncut`` column, defined
   once at build_manifests.py:1149), and ``TestUniqueSecondsExcludesRecuts``
   keeps it fixed here: it builds exactly such a class and asserts the two sums
   come out DIFFERENT.  A build that returns nominal == unique for that input has
   silently reverted.

The fixtures are hand-built ``recs`` dicts rather than a CSV, because
``class_stats`` runs BEFORE recordings.csv is written -- it consumes the
in-memory records.  Testing it through a CSV would test the writer instead.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import split_grouped     # noqa: E402
import build_manifests   # noqa: E402

FOREIGN = split_grouped.NON_OWN_ROOTS[0]
RATE_BYTES_PER_SEC = 8 * 100 * 10 ** 6      # cf32 @ 100 MS/s, the fallback rate


def rec(tax="uav.dji.mini3", root="rawdata", session="S-1", uncut=True,
        seconds=10.0, size=None, artifact_class="recording", **p_extra):
    """One in-memory record, shaped as build_manifests builds them."""
    p = {"artifact_class": artifact_class}
    p.update(p_extra)
    return {
        "p": p,
        "tax_id": tax,
        "tax_short": tax.rsplit(".", 1)[-1],
        "root_key": root,
        "session_id": session,
        "is_uncut": uncut,
        "size": int(seconds * RATE_BYTES_PER_SEC) if size is None else size,
        "delivered_duration_s": seconds,
    }


class TestInclusionRule(unittest.TestCase):
    """Defect 1: the predicate must match build_labels.py exactly."""

    def test_recording_counts(self):
        self.assertTrue(build_manifests.is_recording_row(rec()))

    def test_blank_artifact_class_counts_as_recording(self):
        # The dangerous half.  `artifact_class` is written by dialect.py and a
        # row that predates the column, or whose parse failed, carries "".
        # Treating that as "not a recording" would drop real captures from the
        # published count while every total stayed self-consistent.
        for blank in ("", None):
            self.assertTrue(build_manifests.is_recording_row(
                rec(artifact_class=blank)), repr(blank))

    def test_derived_artifacts_do_not_count(self):
        for cls in ("derived_slice", "derived_cache", "archive_rawdata",
                    "archive_experiment", "experiment_sandbox"):
            self.assertFalse(build_manifests.is_recording_row(
                rec(artifact_class=cls)), cls)

    def test_predicate_agrees_with_build_labels(self):
        # Not a paraphrase of the rule -- the actual other implementation, run
        # over the same rows.  If either side is edited alone, this goes red.
        sys.path.insert(0, os.path.join(_PKG, "taxonomy"))
        import build_labels  # noqa: E402
        src = os.path.join(_PKG, "taxonomy", "build_labels.py")
        with open(src, "r", encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn('(r.get("artifact_class") or "recording") != "recording"',
                      body,
                      "build_labels.py's inclusion rule changed shape -- "
                      "re-check is_recording_row against it")
        self.assertTrue(hasattr(build_labels, "evidence"))

    def test_excluded_rows_are_absent_from_the_stats(self):
        recs = [rec(session="S-1"),
                rec(session="S-2", artifact_class="derived_slice")]
        cs = build_manifests.class_stats(recs, own_only=True)
        self.assertEqual(cs["uav.dji.mini3"]["n_recordings"], 1)
        self.assertEqual(len(cs["uav.dji.mini3"]["sessions"]), 1,
                         "a derived artifact contributed a session")


class TestBothScopes(unittest.TestCase):
    """Defect 2: E-014 item 1 -- the denominator must be declared."""

    def setUp(self):
        self.recs = [
            rec(tax="uav.dji.mini3", root="rawdata", session="S-1"),
            rec(tax="uav.dji.mini3", root="rawdata", session="S-2"),
            rec(tax="uav.dji.mini3", root="rawdata", session="S-3"),
            # exists ONLY in the foreign root
            rec(tax="uav.dji.mavic3", root=FOREIGN, session="S-9"),
        ]

    def test_foreign_only_class_is_absent_from_own_capture(self):
        own = build_manifests.class_stats(self.recs, own_only=True)
        self.assertNotIn("uav.dji.mavic3", own)
        self.assertIn("uav.dji.mini3", own)

    def test_foreign_only_class_is_present_in_all_roots(self):
        allr = build_manifests.class_stats(self.recs, own_only=False)
        self.assertIn("uav.dji.mavic3", allr)

    def test_own_scope_is_strictly_smaller(self):
        # The guard against the scoping quietly becoming a no-op.
        own = build_manifests.class_stats(self.recs, own_only=True)
        allr = build_manifests.class_stats(self.recs, own_only=False)
        self.assertLess(len(own), len(allr))

    def test_foreign_sessions_do_not_inflate_an_own_class(self):
        # A class captured by us AND present in the foreign root: its
        # own-capture session count must not borrow the foreign session, or its
        # eval_tier gets promoted on evidence that is not ours.
        recs = self.recs + [rec(tax="uav.dji.mini3", root=FOREIGN,
                                session="S-foreign")]
        own = build_manifests.class_stats(recs, own_only=True)
        allr = build_manifests.class_stats(recs, own_only=False)
        self.assertEqual(len(own["uav.dji.mini3"]["sessions"]), 3)
        self.assertEqual(len(allr["uav.dji.mini3"]["sessions"]), 4)

    def test_scope_label_travels_with_every_row(self):
        own = build_manifests.class_stats(self.recs, own_only=True)
        rows = build_manifests.class_rows(own, "own_capture")
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["scope"], "own_capture")
            self.assertIn("scope", build_manifests.CLASS_COLS)

    def test_every_emitted_key_is_a_declared_column(self):
        rows = build_manifests.class_rows(
            build_manifests.class_stats(self.recs, own_only=False), "all_roots")
        for r in rows:
            self.assertEqual(sorted(r), sorted(build_manifests.CLASS_COLS))


class TestUniqueSecondsExcludesRecuts(unittest.TestCase):
    """Defect 3: E-015 item 1 -- unique means uncut, and must be provably so."""

    def setUp(self):
        # One 10 s original plus three 4 s re-cuts of it.  nominal = 22 s of
        # row-seconds; unique = the 10 s that exist once.
        self.recs = [rec(session="S-1", uncut=True, seconds=10.0)] + [
            rec(session="S-1", uncut=False, seconds=4.0) for _ in range(3)]

    def stats(self):
        return build_manifests.class_stats(self.recs, own_only=True)["uav.dji.mini3"]

    def test_nominal_counts_every_row(self):
        self.assertAlmostEqual(self.stats()["nominal_seconds"], 22.0)

    def test_unique_counts_only_uncut_rows(self):
        self.assertAlmostEqual(self.stats()["unique_seconds"], 10.0)

    def test_the_two_sums_are_not_the_same_number(self):
        # THE test.  build_labels.py::evidence once computed the nominal sum and
        # labelled it `unique_seconds`; if that shortcut is ever taken here, or
        # re-taken there, this fails.
        e = self.stats()
        self.assertNotEqual(e["nominal_seconds"], e["unique_seconds"])
        self.assertLess(e["unique_seconds"], e["nominal_seconds"])

    def test_recut_rows_still_count_as_recordings(self):
        # They are real files and the manifest must not lose them -- the point
        # is that they do not add SIGNAL, not that they do not exist.
        e = self.stats()
        self.assertEqual(e["n_recordings"], 4)
        self.assertEqual(e["n_uncut"], 1)

    def test_recut_only_class_has_zero_unique_seconds(self):
        # A class present only as slices has no signal that exists once.  Zero
        # is the honest answer; a positive number here would be double-counted
        # signal attributed to a class we never independently captured.
        recs = [rec(tax="rc.kong.k2", uncut=False, seconds=4.0)]
        e = build_manifests.class_stats(recs, own_only=True)["rc.kong.k2"]
        self.assertEqual(e["unique_seconds"], 0.0)
        self.assertEqual(e["n_recordings"], 1)

    def test_bytes_and_seconds_disagree_across_rates(self):
        # E-015 item 6: GiB overstates 200 MS/s material 2x.  Two rows with the
        # same duration and different byte sizes must report equal seconds and
        # unequal bytes -- otherwise `gib` is being used as a proxy for signal.
        recs = [rec(tax="a", seconds=5.0, size=100),
                rec(tax="b", seconds=5.0, size=200)]
        cs = build_manifests.class_stats(recs, own_only=True)
        self.assertEqual(cs["a"]["unique_seconds"], cs["b"]["unique_seconds"])
        self.assertNotEqual(cs["a"]["bytes"], cs["b"]["bytes"])

    def test_missing_duration_is_counted_as_a_fallback(self):
        # split_grouped.row_seconds falls back to size/8/100e6 and says so.
        # The count must surface, because a silent fallback would make the
        # seconds column look measured when it was inferred.
        r = rec(seconds=10.0)
        r["delivered_duration_s"] = None
        e = build_manifests.class_stats([r], own_only=True)["uav.dji.mini3"]
        self.assertEqual(e["duration_fallbacks"], 1)
        self.assertAlmostEqual(e["unique_seconds"], 10.0, places=3)

    def test_present_duration_is_not_a_fallback(self):
        e = build_manifests.class_stats([rec()], own_only=True)["uav.dji.mini3"]
        self.assertEqual(e["duration_fallbacks"], 0)


class TestTierBasisColumn(unittest.TestCase):
    """E-016: the tier disclosure reaches classes.csv, imported not re-derived."""

    def test_column_is_declared_and_emitted(self):
        self.assertIn("tier_basis", build_manifests.CLASS_COLS)
        rows = build_manifests.class_rows(
            build_manifests.class_stats([rec()], own_only=True), "own_capture")
        self.assertIn("tier_basis", rows[0])

    def test_the_real_inversion_survives_the_round_trip(self):
        # mavic3's measured shape, built as records: 3 files, 3 sessions, 1.8 s
        # total.  eval_tier must still say grouped_cv (a split does exist) and
        # tier_basis must refuse to endorse it.
        recs = [rec(tax="uav.dji.mavic3", session="S-%d" % i, seconds=0.60)
                for i in range(3)]
        r = build_manifests.class_rows(
            build_manifests.class_stats(recs, own_only=True), "own_capture")[0]
        self.assertEqual(r["eval_tier"], split_grouped.TIER_CV)
        self.assertEqual(r["n_sessions"], 3)
        self.assertAlmostEqual(r["unique_seconds"], 1.80, places=6)
        self.assertIn(split_grouped.BASIS_SINGLETON, r["tier_basis"])
        self.assertIn(split_grouped.BASIS_THIN, r["tier_basis"])

    def test_a_healthy_class_reads_measured(self):
        # 3 sessions, 10 recordings, 42.6 s -- flysky-fs_i6x's shape.
        recs = ([rec(session="S-1", seconds=14.2)]
                + [rec(session="S-1", uncut=False, seconds=1.0) for _ in range(7)]
                + [rec(session="S-2", seconds=14.2), rec(session="S-3", seconds=14.2)])
        r = build_manifests.class_rows(
            build_manifests.class_stats(recs, own_only=True), "own_capture")[0]
        self.assertEqual(r["n_sessions"], 3)
        self.assertEqual(r["n_recordings"], 10)
        self.assertAlmostEqual(r["unique_seconds"], 42.6, places=6)
        self.assertEqual(r["tier_basis"], split_grouped.BASIS_MEASURED)

    def test_basis_uses_unique_not_nominal_seconds(self):
        # A class whose re-cuts push nominal over the floor while its real
        # signal sits under it.  Reading nominal here would launder exactly the
        # double-counting E-015 exists to prevent.
        recs = [rec(session="S-%d" % i, seconds=2.0) for i in range(3)] + [
            rec(session="S-0", uncut=False, seconds=2.0) for _ in range(10)]
        r = build_manifests.class_rows(
            build_manifests.class_stats(recs, own_only=True), "own_capture")[0]
        self.assertGreater(r["nominal_seconds"], split_grouped.TIER_MIN_UNIQUE_S)
        self.assertLess(r["unique_seconds"], split_grouped.TIER_MIN_UNIQUE_S)
        self.assertIn(split_grouped.BASIS_THIN, r["tier_basis"])

    def test_no_local_copy_of_the_basis_rule(self):
        with open(os.path.join(_PKG, "build_manifests.py"),
                  "r", encoding="utf-8") as fh:
            body = fh.read()
        # NOT in this list: '"measured"'.  build_manifests.py has said
        # "measured" since long before E-016 -- it is the dtype-provenance token
        # in SAMPLE_LAYOUT (build_manifests.py:126, "this prefix was actually
        # probed").  Same word, unrelated vocabulary; asserting on it fails on
        # the pre-existing table instead of on a reimplementation.  The two flag
        # names and the constant are unambiguous, so they carry the test.
        for lit in ('"singleton_sessions"', '"thin_signal"',
                    "TIER_MIN_UNIQUE_S ="):
            self.assertNotIn(lit, body,
                             "E-016 rule inlined in build_manifests.py -- "
                             "import split_grouped.tier_basis instead")
        self.assertIn("split_grouped.tier_basis", body)

    def test_scope_split_matches_what_was_measured(self):
        # The finding that makes this an all_roots defect and not a headline
        # one: the thin/singleton shape lives in the foreign roots.  Same class
        # shape in each scope -> flagged there, absent here.
        recs = [rec(tax="uav.dji.mavic3", root=FOREIGN, session="S-%d" % i,
                    seconds=0.60) for i in range(3)]
        self.assertNotIn("uav.dji.mavic3",
                         build_manifests.class_stats(recs, own_only=True))
        r = build_manifests.class_rows(
            build_manifests.class_stats(recs, own_only=False), "all_roots")[0]
        self.assertFalse(split_grouped.tier_is_supported(
            r["n_sessions"], r["n_recordings"], r["unique_seconds"]))


class TestDerivedOnlySessionsAreNotCaptures(unittest.TestCase):
    """E-016b: the 231-vs-227 session gap, pinned as arithmetic.

    MEASURED on the 2026-07-31 run: sessions.csv and leakage_groups.csv publish
    231 groups, library_totals all_roots publishes 227, and exactly 4 sessions
    account for the difference -- each built entirely from ``experiment_sandbox``
    rows (flysky x1, futaba x2, radiolink x1; 14 rows in total).  Both counts
    are correct.  A sandbox copy is not an independent capture, so class_stats
    drops it; a sandbox copy landing across a split boundary from its parent is
    E-013 leakage, so the leakage tables keep it.  What was missing was anything
    on disk saying so, which is how two right numbers become a discrepancy.
    """

    def test_a_sandbox_only_session_contributes_no_class_session(self):
        recs = [rec(session="S-real"),
                rec(session="S-sandbox", artifact_class="experiment_sandbox")]
        e = build_manifests.class_stats(recs, own_only=True)["uav.dji.mini3"]
        self.assertEqual(len(e["sessions"]), 1)
        self.assertNotIn("S-sandbox", e["sessions"])

    def test_the_measured_four_are_reproduced_in_miniature(self):
        # The real shape: 4 sessions of sandbox rows (5+4+4+1 = 14) alongside
        # capture-bearing ones.  Capture sessions must count; sandbox must not.
        recs = [rec(session="S-cap-%d" % i) for i in range(3)]
        for sid, n in (("S-sb-1", 5), ("S-sb-2", 4), ("S-sb-3", 4), ("S-sb-4", 1)):
            recs += [rec(session=sid, artifact_class="experiment_sandbox")
                     for _ in range(n)]
        e = build_manifests.class_stats(recs, own_only=True)["uav.dji.mini3"]
        all_sessions = {r["session_id"] for r in recs}
        self.assertEqual(len(all_sessions), 7)
        self.assertEqual(len(e["sessions"]), 3,
                         "sandbox-only sessions leaked into the capture count")
        self.assertEqual(len(all_sessions) - len(e["sessions"]), 4)
        self.assertEqual(e["n_recordings"], 3)

    def test_totals_declare_both_session_counts(self):
        # The reconciliation must exist as columns, not as a stdout aside.
        src = os.path.join(_PKG, "build_manifests.py")
        with open(src, "r", encoding="utf-8") as fh:
            body = fh.read()
        for col in ("n_sessions_incl_derived", "n_sessions_derived_only"):
            self.assertIn('"%s"' % col, body, col)
        # ...and both must be written, not merely computed.
        head = body[body.index('wcsv("library_totals.csv"'):][:900]
        for col in ("n_sessions", "n_sessions_incl_derived",
                    "n_sessions_derived_only", "n_grouped_cv_supported",
                    "n_tier_unsupported", "n_thin_signal"):
            self.assertIn(col, head, col + " computed but not written")


class TestThinSignalTotalIsPublished(unittest.TestCase):
    """E-016, second pass: the two totals must disagree, and both must ship.

    ``n_tier_unsupported`` answers "how many tiers assert a split they cannot
    back"; ``n_thin_signal`` answers "how many classes carry under
    TIER_MIN_UNIQUE_S of unique signal".  On the measured library those are 21
    and 23 -- the gap is the two ``unsplittable`` classes, which assert no split
    and so cannot overclaim one.  If a future edit makes the two columns equal
    again, one of the two questions has stopped being asked.
    """

    def test_the_two_totals_are_computed_from_different_predicates(self):
        with open(os.path.join(_PKG, "build_manifests.py"),
                  "r", encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn('"n_thin_signal"', body)
        # n_thin_signal must read the basis string, not the support boolean --
        # deriving it from `supported` would collapse it onto the other column.
        blk = body[body.index('"n_thin_signal"'):][:400]
        self.assertIn("split_grouped.BASIS_THIN", blk)
        self.assertIn('r["tier_basis"]', blk)
        self.assertNotIn("supported", blk)

    def test_a_thin_unsplittable_class_lands_in_thin_but_not_unsupported(self):
        # The exact shape of the two classes: 1 session, 1 recording, ~4 s.
        r = build_manifests.class_rows(
            build_manifests.class_stats([rec(seconds=4.12)], own_only=True),
            "own_capture")[0]
        self.assertEqual(r["eval_tier"], split_grouped.TIER_NONE)
        self.assertEqual(r["tier_basis"], split_grouped.BASIS_THIN)
        self.assertTrue(split_grouped.tier_is_supported(
            r["n_sessions"], r["n_recordings"], r["unique_seconds"]))


class TestTierIsImportedNotReimplemented(unittest.TestCase):
    """The same guard test_labels.py applies to labels.json, applied here."""

    def test_tier_matches_the_splitter_rule(self):
        recs = ([rec(tax="cv", session="S-%d" % i) for i in range(3)]
                + [rec(tax="h2", session="S-a"), rec(tax="h2", session="S-b")]
                + [rec(tax="none", session="S-solo")])
        rows = {r["taxonomy_id"]: r for r in build_manifests.class_rows(
            build_manifests.class_stats(recs, own_only=True), "own_capture")}
        for tid, r in rows.items():
            self.assertEqual(r["eval_tier"],
                             split_grouped.tier_of(r["n_sessions"]), tid)
        self.assertEqual(rows["cv"]["eval_tier"], split_grouped.TIER_CV)
        self.assertEqual(rows["h2"]["eval_tier"], split_grouped.TIER_1V1)
        self.assertEqual(rows["none"]["eval_tier"], split_grouped.TIER_NONE)

    def test_no_local_copy_of_the_tiering_rule(self):
        # A literal "grouped_cv" in build_manifests.py would mean the rule was
        # inlined instead of imported.
        with open(os.path.join(_PKG, "build_manifests.py"),
                  "r", encoding="utf-8") as fh:
            body = fh.read()
        for lit in ('"grouped_cv"', '"holdout_1v1"', '"unsplittable"'):
            self.assertNotIn(lit, body,
                             "tiering rule inlined in build_manifests.py -- "
                             "import split_grouped.tier_of instead")

    def test_no_local_copy_of_the_foreign_root_list(self):
        # Note what this does NOT assert: `.dat_pic` appears in
        # build_manifests.py at DTYPE_BY_ROOT, where it carries a measured
        # ci16_le, and that is a different fact about the same directory.  What
        # must not exist is a second BINDING of the eval-scope list, since that
        # is what would let the manifest and the splitter disagree about which
        # roots are reportable.
        with open(os.path.join(_PKG, "build_manifests.py"),
                  "r", encoding="utf-8") as fh:
            body = fh.read()
        for name in ("NON_OWN_ROOTS =", "FOREIGN_ROOTS =", "NON_OWN_ROOTS=",
                     "FOREIGN_ROOTS="):
            self.assertNotIn(name, body,
                             "foreign-root list re-bound in "
                             "build_manifests.py -- reference "
                             "split_grouped.NON_OWN_ROOTS instead")
        self.assertIn("split_grouped.NON_OWN_ROOTS", body,
                      "the scope filter no longer reads the imported list")


class TestRecordingColumns(unittest.TestCase):
    """The six new recordings.csv columns exist and are declared."""

    def test_new_columns_are_in_rec_cols(self):
        for c in ("own_capture", "is_uncut", "n_sessions_in_class",
                  "eval_tier", "n_sessions_in_class_all_roots",
                  "eval_tier_all_roots"):
            self.assertIn(c, build_manifests.REC_COLS, c)

    def test_rec_cols_has_no_duplicates(self):
        self.assertEqual(len(build_manifests.REC_COLS),
                         len(set(build_manifests.REC_COLS)))

    def test_unique_seconds_is_not_a_row_level_column(self):
        # E-015 defines unique_seconds as a PER-CLASS quantity.  A row-level
        # column of the same name would be read as "this row's unique seconds",
        # which is not a thing that exists.  `is_uncut` is exported instead so
        # the class figure stays recomputable from recordings.csv.
        self.assertNotIn("unique_seconds", build_manifests.REC_COLS)
        self.assertIn("is_uncut", build_manifests.REC_COLS)
        self.assertIn("unique_seconds", build_manifests.CLASS_COLS)


if __name__ == "__main__":
    unittest.main()
