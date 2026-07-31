# -*- coding: utf-8 -*-
"""Unit tests for the class registry ``taxonomy/labels.json`` and its generator.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only, python >= 3.9.

WHY THESE TESTS EXIST
---------------------
labels.json is the file the datasheet will cite for "how many classes does RFUAV
have".  Two ways it can lie, and each gets a test class here:

1. IT DRIFTS.  Someone adds a device to dialect.py, or renames a protocol in
   link_protocols.json, and labels.json keeps reporting the old world.  The
   generator has three assertions against exactly this -- so the first job of
   this file is to prove those assertions FAIL on broken input.  A guard that
   has only ever been observed passing is not a guard.  ``TestGeneratorRefuses``
   breaks each of the three deliberately.

2. IT MISCOUNTS.  The dangerous number is not the wrong one, it is the
   plausible one.  The library reports 63 observed taxonomy_ids and the goal is
   ">=60 classes" -- but 23 of those 63 exist ONLY inside the foreign roots that
   E-005/E-006 measured as int16-converted with cross-class duplicate content,
   so the count defensible as RFUAV's own capture is 40.  A build that quietly
   reports 63, or an eval-tier histogram that folds those 23 zero-session
   classes into `unsplittable` (which the first build did), produces a table
   that looks like the goal is met.  ``TestHonestyCounts`` pins the arithmetic.

The evidence fields (sessions, seconds, bands) come from a recordings manifest
that lives on the data server, not in the repo, so nothing here asserts their
real values.  What is asserted is everything derivable from files that ARE
committed -- vocabulary, protocol assignment, tiering rule, and the internal
consistency of the counts against each other.  That is the part that can drift
without anyone noticing.

3. IT PUBLISHES A DOUBLE-COUNT UNDER AN HONEST NAME.  ``evidence()`` used to sum
   ``delivered_duration_s`` over EVERY recording row and emit the result as
   ``unique_seconds`` -- and 531 of 700 own-capture rows are re-cut windows of a
   parent in the same session, so that figure overstated the library by roughly
   the amount the name promised to exclude.  It could not have done otherwise:
   the column that separates the two did not exist yet.  Now it does, so
   ``TestSecondsAreNotDoubleCounted`` pins the two sums apart on a fixture built
   to make them differ, and pins the missing-column case as fatal.
"""

from __future__ import annotations

import io
import os
import sys
import csv
import json
import copy
import shutil
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # tools/rfuav_v2
_TAX = os.path.join(_PKG, "taxonomy")
for _p in (_PKG, _TAX):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dialect          # noqa: E402
import split_grouped    # noqa: E402
import build_labels     # noqa: E402

FIX = os.path.join(_HERE, "fixtures")
# Reused deliberately: this fixture already carries own-capture rows, ZJU rows
# and .dat_pic rows, which is exactly the three-way shape the honesty counts
# depend on.  A second near-identical fixture would be one more thing to keep
# in sync.
RECORDINGS = os.path.join(FIX, "recordings_mixed_roots.csv")

COMMITTED = os.path.join(_TAX, "labels.json")
REGISTRY = os.path.join(_TAX, "link_protocols.json")


def load(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def registry_ids():
    return {p["id"] for p in load(REGISTRY)["protocols"]}


class TestCommittedFileIsConsistentWithItsSources(unittest.TestCase):
    """The drift tests, green side: the file on disk still matches the repo."""

    @classmethod
    def setUpClass(cls):
        cls.doc = load(COMMITTED)
        cls.rows = {r["taxonomy_id"]: r for r in cls.doc["labels"]}
        cls.valid = registry_ids()

    def test_every_dialect_class_is_present(self):
        # The Parrot row is the one intentional difference: dialect.py still
        # emits the directory spelling, labels.json publishes the canonical id
        # and keeps the old one in `supersedes` (E-007).
        vocab = set(build_labels.vocabulary())
        vocab.discard(build_labels.PARROT_OBSERVED)
        vocab.add(build_labels.PARROT_CANONICAL)
        missing = sorted(vocab - set(self.rows))
        self.assertEqual(missing, [], "classes in dialect._V absent from "
                                      "labels.json -- rebuild it")
        extra = sorted(set(self.rows) - vocab)
        self.assertEqual(extra, [], "labels.json has classes dialect._V does "
                                    "not produce")

    def test_every_protocol_id_exists_in_the_registry(self):
        bad = sorted({r["taxonomy_id"] for r in self.doc["labels"]
                      if r["link_protocol"] is not None
                      and r["link_protocol"] not in self.valid})
        self.assertEqual(bad, [], "link_protocol not in link_protocols.json")

    def test_every_assignment_carries_its_basis(self):
        # F2: an unsourced protocol claim is indistinguishable from a guess.
        for tid, r in sorted(self.rows.items()):
            self.assertTrue((r.get("link_protocol_basis") or "").strip(),
                            "%s has no link_protocol_basis" % tid)

    def test_evidence_level_matches_the_assignment(self):
        for tid, r in sorted(self.rows.items()):
            pid, lvl = r["link_protocol"], r["link_protocol_evidence"]
            if pid is None:
                self.assertEqual(lvl, "not_applicable", tid)
            elif pid == "unknown":
                self.assertEqual(lvl, "unknown", tid)
            elif tid == build_labels.PARROT_CANONICAL:
                self.assertEqual(lvl, "verified", tid)
            else:
                self.assertEqual(lvl, "factory_default_prior", tid)

    def test_exactly_one_row_is_verified(self):
        # If a second row ever claims `verified`, it needs its own forensic
        # write-up in ERRATA before this test is relaxed.
        ver = sorted(r["taxonomy_id"] for r in self.doc["labels"]
                     if r["link_protocol_evidence"] == "verified")
        self.assertEqual(ver, [build_labels.PARROT_CANONICAL])

    def test_committed_file_matches_a_fresh_build(self):
        # Only over the fields that do NOT come from the recordings manifest --
        # the manifest is server-side, so a repo-only test cannot reproduce
        # session counts.  Everything below is authored or derived from files
        # under version control, so a mismatch here means labels.json is stale.
        keys = ("vendor", "model", "device_role", "vendor_family", "raw_keys",
                "supersedes", "rename_errata", "link_protocol",
                "link_protocol_basis", "link_protocol_evidence",
                "variant_unresolved")
        fresh = {r["taxonomy_id"]: r for r in build_labels.build(RECORDINGS)["labels"]}
        self.assertEqual(sorted(fresh), sorted(self.rows),
                         "fresh build and committed file disagree on the class set")
        for tid in sorted(fresh):
            for k in keys:
                self.assertEqual(fresh[tid].get(k), self.rows[tid].get(k),
                                 "labels.json is stale at %s.%s -- re-run "
                                 "build_labels.py" % (tid, k))


class TestGeneratorRefuses(unittest.TestCase):
    """The red tests.  These matter more than the green ones.

    Each mutates ``_PROTOCOL`` to create one of the three drift shapes the
    docstring of build_labels.py promises to catch, and asserts the build dies
    instead of emitting a file that merely looks right.
    """

    def setUp(self):
        self._saved = copy.deepcopy(build_labels._PROTOCOL)
        self.addCleanup(self._restore)

    def _restore(self):
        build_labels._PROTOCOL.clear()
        build_labels._PROTOCOL.update(self._saved)

    def _build_fails(self):
        with open(os.devnull, "w") as null:
            old, sys.stderr = sys.stderr, null
            try:
                with self.assertRaises(SystemExit) as cm:
                    build_labels.build(RECORDINGS)
            finally:
                sys.stderr = old
        return str(cm.exception)

    def test_stale_protocol_id_is_rejected(self):
        tid = build_labels.PARROT_CANONICAL
        build_labels._PROTOCOL[tid] = ("wifi.no_such_protocol", "typo")
        self.assertIn("consistency error", self._build_fails())

    def test_unassigned_class_is_rejected(self):
        # The "someone added a device to dialect.py" case.  Silence here would
        # publish a class with no protocol field at all.
        victim = sorted(build_labels.vocabulary())[0]
        build_labels._PROTOCOL.pop(victim, None)
        build_labels._PROTOCOL.pop(build_labels.PARROT_CANONICAL, None)
        self.assertIn("consistency error", self._build_fails())

    def test_dead_assignment_is_rejected(self):
        # The "class was renamed, its assignment was not" case.
        build_labels._PROTOCOL["uav.nosuchvendor.nosuchmodel"] = ("unknown", "x")
        self.assertIn("consistency error", self._build_fails())

    def test_a_valid_table_still_builds(self):
        # Control: the three tests above must fail for the reason claimed, not
        # because build() is broken for every input.
        doc = build_labels.build(RECORDINGS)
        self.assertEqual(len(doc["labels"]), len(build_labels.vocabulary()))


class TestHonestyCounts(unittest.TestCase):
    """The miscount tests.  E-014.3, and the histogram defect it produced."""

    @classmethod
    def setUpClass(cls):
        cls.doc = load(COMMITTED)
        cls.rows = cls.doc["labels"]
        cls.h = cls.doc["headline_counts"]

    def test_headline_counts_agree_with_the_rows(self):
        own = [r for r in self.rows if r["own_capture"]]
        obs = [r for r in self.rows if r["observed_in_library"]]
        foreign = [r for r in obs if not r["own_capture"]]
        self.assertEqual(self.h["taxonomy_ids_total"], len(self.rows))
        self.assertEqual(self.h["observed_in_library"], len(obs))
        self.assertEqual(self.h["reportable_own_capture"], len(own))
        self.assertEqual(self.h["foreign_root_only"], len(foreign))
        self.assertEqual(self.h["registered_but_never_observed"],
                         len(self.rows) - len(obs))

    def test_reportable_is_strictly_smaller_than_observed(self):
        # The whole point.  If these ever coincide, either the foreign roots
        # were dropped from the manifest (fine -- delete this test and say why)
        # or own_capture stopped being computed (not fine).
        self.assertLess(self.h["reportable_own_capture"],
                        self.h["observed_in_library"])
        self.assertEqual(self.h["reportable_own_capture"], 40)
        self.assertEqual(self.h["foreign_root_only"], 23)

    def test_honesty_note_states_the_smaller_number(self):
        note = self.h["honesty_note"]
        self.assertIn("40", note)
        self.assertIn("63", note)

    def test_foreign_only_rows_are_not_reportable(self):
        for r in self.rows:
            if r["observed_in_library"] and not r["own_capture"]:
                self.assertFalse(r["reportable_in_class_count"], r["taxonomy_id"])
                self.assertEqual(r["n_sessions_own_capture"], 0, r["taxonomy_id"])

    def test_histogram_covers_the_reportable_classes_only(self):
        # The defect this pins: computing the histogram over all 64 rows folds
        # the 23 zero-own-session classes into `unsplittable` and makes the
        # own-capture library look far less splittable than it is.
        hist = self.doc["eval_tier_histogram_own_capture"]
        self.assertEqual(sum(hist.values()), self.h["reportable_own_capture"])
        self.assertNotEqual(sum(hist.values()), self.h["taxonomy_ids_total"])

    def test_tier_matches_the_splitter_rule(self):
        # labels.json must not carry a second, divergent copy of the tiering
        # rule -- the splitter is the single definition.
        for r in self.rows:
            self.assertEqual(r["eval_tier"],
                             split_grouped.tier_of(r["n_sessions_own_capture"]),
                             r["taxonomy_id"])
            self.assertEqual(r["eval_tier_including_foreign"],
                             split_grouped.tier_of(r["n_sessions_total"]),
                             r["taxonomy_id"])

    def test_unsplittable_classes_have_exactly_one_own_session(self):
        for r in self.rows:
            if r["own_capture"] and r["eval_tier"] == split_grouped.TIER_NONE:
                self.assertEqual(r["n_sessions_own_capture"], 1, r["taxonomy_id"])


class TestOpenDecisionsStayVisible(unittest.TestCase):
    """Two deviations from the spec are recorded in the file, not hidden.

    Both are things a reader of the datasheet would need to know, and both are
    the kind of thing that gets quietly dropped in a later rebuild.
    """

    @classmethod
    def setUpClass(cls):
        cls.doc = load(COMMITTED)
        cls.rows = {r["taxonomy_id"]: r for r in cls.doc["labels"]}

    def test_vendor_merge_deviation_is_declared_open(self):
        d = self.doc["vendor_merge_decision"]
        self.assertIn("OPEN", d["status"])
        self.assertIn("NOT merged", d["decision"])

    def test_the_two_h12_radios_remain_distinct_classes(self):
        # F1: Skydroid H12 and Yunzhuo H12 are different physical radios, both
        # captured.  A vendor merge would fuse them and lose a real class.
        self.assertIn("rc.skydroid.h12", self.rows)
        self.assertIn("rc.yunzhuo.h12", self.rows)
        for tid in ("rc.skydroid.h12", "rc.yunzhuo.h12"):
            self.assertEqual(self.rows[tid].get("vendor_family"),
                             "skydroid_yunzhuo", tid)

    def test_parrot_rename_carries_its_migration(self):
        r = self.rows[build_labels.PARROT_CANONICAL]
        self.assertNotIn(build_labels.PARROT_OBSERVED, self.rows)
        self.assertIn(build_labels.PARROT_OBSERVED, r["supersedes"])
        self.assertEqual(r["rename_errata"], "E-007")
        self.assertIn("build_manifests.py", r["migration_required"])

    def test_registered_but_unassigned_protocols_are_explained(self):
        u = self.doc["protocols_registered_but_unassigned"]
        assigned = {r["link_protocol"] for r in self.doc["labels"]}
        for pid in u["ids"]:
            self.assertIn(pid, registry_ids())
            self.assertNotIn(pid, assigned)
        self.assertIn("F7", u["why"])

    def test_grouping_key_warning_names_session_id(self):
        w = self.doc["grouping_key_warning"]
        self.assertIn("session_id", w)
        self.assertIn("never taxonomy_id", w)


class TestSecondsAreNotDoubleCounted(unittest.TestCase):
    """E-015 item 1 and item 6, measured on a fixture built to expose them.

    ``recordings_mixed_roots.csv`` gives ``uav.dji.mini3`` four 1 s recording
    rows: r001 and r003 uncut, r002 an explicit slice, r004 a window declared by
    its own filename (``c_00_0.0_1.0s.dat``) and therefore carrying
    ``is_slice=0``.  So nominal = 4.0 s, unique = 2.0 s, and the predicate
    ``not is_slice`` -- the one that looks equivalent -- yields 3.0.

    That third value is what makes this worth a test rather than a comment: it is
    wrong by one whole recording while remaining entirely plausible.
    """

    @classmethod
    def setUpClass(cls):
        cls.ev = build_labels.evidence(RECORDINGS)
        cls.rows = {r["taxonomy_id"]: r
                    for r in build_labels.build(RECORDINGS)["labels"]}

    def test_the_two_sums_differ(self):
        e = self.ev["uav.dji.mini3"]
        self.assertAlmostEqual(e["nominal_seconds"], 4.0, places=3)
        self.assertAlmostEqual(e["unique_seconds"], 2.0, places=3)
        self.assertLess(e["unique_seconds"], e["nominal_seconds"])

    def test_the_loose_predicate_would_have_given_a_different_answer(self):
        # Stated as arithmetic so a regression cannot pass by coincidence: three
        # of mini3's four rows have is_slice=0, only two are uncut.
        with io.open(RECORDINGS, "r", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r["taxonomy_id"] == "uav.dji.mini3"
                    and (r.get("artifact_class") or "recording") == "recording"]
        loose = sum(split_grouped.row_seconds(r)[0] for r in rows
                    if (r.get("is_slice") or "0") != "1")
        strict = sum(split_grouped.row_seconds(r)[0] for r in rows
                     if split_grouped.row_is_uncut(r))
        self.assertAlmostEqual(loose, 3.0, places=3)
        self.assertAlmostEqual(strict, 2.0, places=3)

    def test_both_figures_are_published_side_by_side(self):
        # E-015 item 6: replacing the nominal sum with the unique sum does not
        # make the file honest, it just moves which number is missing.  A reader
        # needs the ratio, so both ship, plus the row count each is over.
        r = self.rows["uav.dji.mini3"]
        for k in ("nominal_seconds", "unique_seconds", "n_recordings",
                  "n_uncut_recordings", "duration_inferred_rows"):
            self.assertIn(k, r)
        self.assertEqual(r["n_recordings"], 4)
        self.assertEqual(r["n_uncut_recordings"], 2)

    def test_no_row_claims_more_unique_than_nominal(self):
        for tid, r in sorted(self.rows.items()):
            self.assertLessEqual(r["unique_seconds"], r["nominal_seconds"], tid)
            self.assertLessEqual(r["n_uncut_recordings"], r["n_recordings"], tid)

    def test_inferred_durations_are_counted_not_assumed(self):
        # The ZJU rows carry no delivered_duration_s, so their seconds come from
        # size/8/100e6.  That is a guess about sample rate; it may be right, but
        # it must not sit in the same column as a measured value unlabelled.
        e = self.ev["uav.dji.mavic3"]
        self.assertEqual(e["duration_fallbacks"], 3)
        self.assertAlmostEqual(e["unique_seconds"], 1.5, places=3)
        self.assertEqual(self.ev["uav.dji.mini3"]["duration_fallbacks"], 0)

    def test_derived_artifacts_contribute_no_seconds(self):
        # r010 is a spectrogram of r001.  Counting it would add signal that was
        # already counted, in a unit (seconds) it does not even have.
        self.assertEqual(self.ev["uav.dji.mini3"]["n_recordings"], 4)

    def test_a_manifest_without_is_uncut_is_fatal(self):
        # The guard is fatal, not a fallback, because the only fallback
        # available is the bug: `not is_slice` over-counts, and it over-counts
        # in the direction that flatters the dataset.
        tmp = tempfile.mkdtemp(prefix="rfuav-labels-")
        self.addCleanup(shutil.rmtree, tmp, True)
        old = os.path.join(tmp, "old_manifest.csv")
        with io.open(RECORDINGS, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        i = rows[0].index("is_uncut")
        with io.open(old, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            for row in rows:
                w.writerow([v for j, v in enumerate(row) if j != i])
        with self.assertRaises(SystemExit) as cm:
            build_labels.evidence(old)
        msg = str(cm.exception)
        self.assertIn("is_uncut", msg)
        self.assertIn("build_manifests.py", msg)
        self.assertIn("E-015", msg)

    def test_the_guard_is_shared_with_the_splitter(self):
        # One guard, one message.  A second copy in this file would be a second
        # thing to keep in sync with the definition in build_manifests.py.
        self.assertIs(build_labels.split_grouped.require_is_uncut,
                      split_grouped.require_is_uncut)
        self.assertIs(build_labels.split_grouped.row_is_uncut,
                      split_grouped.row_is_uncut)


if __name__ == "__main__":
    unittest.main()
