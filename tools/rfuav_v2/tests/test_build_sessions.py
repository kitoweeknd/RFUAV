# -*- coding: utf-8 -*-
"""Unit tests for build_manifests.build_sessions relation 3 (sha512 union, P0.5).

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only.

WHY THESE TESTS EXIST
---------------------
gap 75: two byte-identical files ARE the same emission and must never straddle
a train/test split, but if a duplicate was renamed out of its stem group AND is
not a substring of its twin, neither relation 1 (stem) nor relation 2 (tail-of)
catches it.  P0.5 adds relation 3: union recordings that share a whole-file
SHA-512.  The effect is narrow by measurement (1/25 hash groups crossed
sessions, 0.80 GB) and is gated by ``use_sha`` so its marginal size stays
reportable -- these tests pin (a) that it merges a stem/tail-invisible duplicate
and (b) that an empty hash never merges (an unhashed row is not "identical to"
another unhashed row).
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                        # tools/rfuav_v2
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import build_manifests as bm  # noqa: E402


def _rec(relpath, stem, sha512, tax="T", cond="C"):
    return {"relpath": relpath, "tax_key": tax, "condition": cond,
            "stem": stem, "sha512": sha512}


def _n_sessions(recs, **kw):
    kw.setdefault("assign", False)
    return len(bm.build_sessions(recs, [], **kw))


class TestSha512Union(unittest.TestCase):

    def test_identical_hash_merges_across_distinct_stems(self):
        recs = [_rec("a.iq", "stemA", "HASH"),
                _rec("b.iq", "stemB", "HASH")]   # different stems, same bytes
        self.assertEqual(_n_sessions(recs, use_tail=False, use_sha=True), 1)

    def test_without_sha_the_same_pair_stays_split(self):
        recs = [_rec("a.iq", "stemA", "HASH"),
                _rec("b.iq", "stemB", "HASH")]
        # use_sha=False recomputes without relation 3 -> the leak reappears,
        # which is exactly what makes relation 3's marginal effect measurable.
        self.assertEqual(_n_sessions(recs, use_tail=False, use_sha=False), 2)

    def test_empty_hash_never_merges(self):
        recs = [_rec("a.iq", "stemA", ""),
                _rec("b.iq", "stemB", "")]
        self.assertEqual(_n_sessions(recs, use_tail=False, use_sha=True), 2)

    def test_stem_relation_still_groups_regardless_of_hash(self):
        recs = [_rec("a.iq", "stemA", "H1"),
                _rec("b.iq", "stemA", "H2")]     # same stem, different bytes
        self.assertEqual(_n_sessions(recs, use_tail=False, use_sha=True), 1)


if __name__ == "__main__":
    unittest.main()
