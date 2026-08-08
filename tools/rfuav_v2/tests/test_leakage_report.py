# -*- coding: utf-8 -*-
"""Unit tests for leakage_report (P0.4): the naive-vs-grouped honesty report.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only.

WHY THESE TESTS EXIST
---------------------
The report's job is to make three numbers travel together so a leakage-clean
split can't read as more than it is: (2) grouped leakage must be 0, (3) the
files-per-session shape (gap 58: a grouped_cv tier can be 159/231 singletons),
and (4) per-class tier support delegated to split_grouped.  These tests pin the
pure computations: straddle detection, the singleton distribution, and that the
tier verdicts are exactly split_grouped's (measured / singleton / unsplittable),
never a re-implementation that could drift from the frozen tier_of.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                        # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import leakage_report as lr  # noqa: E402
import split_grouped as sg   # noqa: E402


def _row(cls, sid, split, dur="6", uncut="1"):
    return {"taxonomy_id": cls, "session_id": sid, "split": split,
            "delivered_duration_s": dur, "is_uncut": uncut}


class TestGroupedLeakage(unittest.TestCase):

    def test_clean_split_has_zero_straddle(self):
        rows = [_row("a", "s1", "train"), _row("a", "s2", "valid")]
        n, total, straddling = lr.grouped_leakage(rows)
        self.assertEqual(n, 0)
        self.assertEqual(total, 2)

    def test_session_in_two_splits_is_flagged(self):
        rows = [_row("a", "s1", "train"), _row("a", "s1", "valid")]
        n, total, straddling = lr.grouped_leakage(rows)
        self.assertEqual(n, 1)
        self.assertEqual(straddling, ["s1"])

    def test_excluded_split_is_ignored(self):
        # a session that only appears in excluded must not count as a group.
        rows = [_row("a", "s1", "excluded"), _row("a", "s1", "excluded")]
        n, total, straddling = lr.grouped_leakage(rows)
        self.assertEqual(total, 0)


class TestFilesPerSession(unittest.TestCase):

    def test_singleton_fraction(self):
        per = lr._sessions_by_class([
            _row("a", "s1", "train"), _row("a", "s1", "train"),  # s1 = 2 files
            _row("a", "s2", "train"),                            # s2 = 1 file
            _row("b", "s3", "valid"),                            # s3 = 1 file
        ])
        stats = lr.files_per_session_stats(per)
        self.assertEqual(stats["n_sessions"], 3)
        self.assertEqual(stats["singletons"], 2)
        self.assertAlmostEqual(stats["singleton_frac"], 2 / 3)
        self.assertEqual(stats["max"], 2)


class TestTierSummaryDelegates(unittest.TestCase):

    def test_verdicts_match_split_grouped_exactly(self):
        rows = [
            # class A: 3 sessions, 4 recordings -> measured grouped_cv, quotable
            _row("uav.a", "sA1", "train"), _row("uav.a", "sA1", "train"),
            _row("uav.a", "sA2", "train"), _row("uav.a", "sA3", "valid"),
            # class B: 2 singleton sessions -> holdout_1v1 + singleton, NOT quotable
            _row("uav.b", "sB1", "train"), _row("uav.b", "sB2", "valid"),
            # class C: 1 session -> unsplittable, quotable (exempt)
            _row("uav.c", "sC1", "excluded", dur="3"),
        ]
        per = lr._sessions_by_class(rows)
        tier_rows, n_sup, n_unsup = lr.tier_summary(per)
        by_cls = {t["class"]: t for t in tier_rows}

        self.assertEqual(by_cls["uav.a"]["eval_tier"], sg.TIER_CV)
        self.assertEqual(by_cls["uav.a"]["tier_basis"], sg.BASIS_MEASURED)
        self.assertTrue(by_cls["uav.a"]["supported"])

        self.assertEqual(by_cls["uav.b"]["eval_tier"], sg.TIER_1V1)
        self.assertEqual(by_cls["uav.b"]["tier_basis"], sg.BASIS_SINGLETON)
        self.assertFalse(by_cls["uav.b"]["supported"])

        self.assertEqual(by_cls["uav.c"]["eval_tier"], sg.TIER_NONE)
        self.assertTrue(by_cls["uav.c"]["supported"])   # unsplittable is exempt

        self.assertEqual((n_sup, n_unsup), (2, 1))


if __name__ == "__main__":
    unittest.main()
