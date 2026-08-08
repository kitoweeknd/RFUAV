# -*- coding: utf-8 -*-
"""Unit tests for build_manifests.session_stem's E-016 ZJU-SNR merge.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only.

WHY THESE TESTS EXIST
---------------------
E-016: the anti-leakage grouping itself leaked, because the ZJU ordinal
attenuation steps ``-SNR1-`` / ``-SNR2-`` / ``-SNR3-`` of ONE capture were
counted as three independent sessions.  P0.2 strips that ordinal in
``session_stem`` so the steps fuse into one session (documented v2 re-cut:
81 -> 27 sessions, 21 classes degraded off grouped_cv).

The regex that does it is ``-SNR\\d+(?=-|$)``.  The plan originally specified
``-SNR\\d+(?!dB)``, which is WRONG: on ``DJI AVTA2-SNR10dB-...`` the ``\\d+``
backtracks from ``10`` to ``1``, the ``(?!dB)`` lookahead then passes on the
``0``, and the name is mangled to ``AVATA20dB`` -- silently fusing the AVATA2
NATIVE dB ladder (a real calibration asset for goal 3a) into garbage.  These
tests pin both halves: ZJU ordinals MUST fuse, AVATA2 ``SNR{n}dB`` MUST survive.
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


class TestZjuOrdinalMerges(unittest.TestCase):

    def test_snr_steps_of_one_capture_fuse_to_one_stem(self):
        stems = {bm.session_stem("root\\DJMINI3-SNR%s-5770m.iq" % n, None)
                 for n in ("1", "2", "3", "10")}
        self.assertEqual(stems, {"DJMINI3-5770m"},
                         "ZJU SNR ordinal steps must share one session stem")

    def test_ordinal_at_end_of_stem_is_stripped(self):
        # -SNR followed by end-of-stem (no trailing -{cf}m) must also strip.
        self.assertEqual(bm.session_stem("root\\FOO-SNR2.iq", None), "FOO")

    def test_take_number_kept_slice_window_stripped(self):
        # The take number (N) identifies a distinct recording and is KEPT; the
        # _A-Bs slice window is stripped; the SNR ordinal is stripped.
        self.assertEqual(
            bm.session_stem("root\\DJMINI3-SNR1-5770m(2)_22-24s.iq", None),
            "DJMINI3-5770m(2)")

    def test_partial_mark_stripped_alongside_ordinal(self):
        name = "DJMINI3-SNR1-5770m" + bm.dialect.PARTIAL_MARK
        self.assertEqual(bm.session_stem("root\\%s.iq" % name, None),
                         "DJMINI3-5770m")


class TestAvata2LadderPreserved(unittest.TestCase):

    def test_db_ladder_is_not_touched(self):
        for db in ("0", "2", "10"):
            name = "DJI AVTA2-SNR%sdB-85db_5765m_100m_10m" % db
            self.assertEqual(bm.session_stem("root\\%s.iq" % name, None), name,
                             "AVATA2 native dB ladder must NOT be stripped")

    def test_regex_does_not_backtrack_on_two_digit_db(self):
        # The exact failure of the rejected (?!dB) pattern.
        self.assertEqual(bm._RE_ZJU_SNR.sub("", "DJI AVTA2-SNR10dB-x"),
                         "DJI AVTA2-SNR10dB-x")

    def test_regex_strips_bare_ordinal(self):
        self.assertEqual(bm._RE_ZJU_SNR.sub("", "m-SNR3-5770m"), "m-5770m")


if __name__ == "__main__":
    unittest.main()
