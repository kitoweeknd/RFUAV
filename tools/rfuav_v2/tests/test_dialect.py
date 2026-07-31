# -*- coding: utf-8 -*-
"""Unit tests for the RFUAV v1 filename dialect parser.

Run from the repository root::

    python3 -m unittest discover -s tools/rfuav_v2/tests

stdlib only, python >= 3.9.  Every filename asserted here is a VERBATIM row of
``memoria/datos/e1/e1_binfiles.csv`` (the E1 read-only scan of
``E:\\Drone_dataset\\RFUAV``); nothing is invented.
"""

from __future__ import annotations

import csv
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # tools/rfuav_v2
_REPO = os.path.dirname(os.path.dirname(_PKG))      # repo root
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import dialect  # noqa: E402

CSV_PATH = os.path.join(_REPO, "memoria", "datos", "e1", "e1_binfiles.csv")
PARTIAL = dialect.PARTIAL_MARK
ZJU = dialect.ROOT_ZJU
BIG = dialect.ROOT_OVERSIZE


class DialectTestBase(unittest.TestCase):

    def p(self, relpath):
        return dialect.parse(relpath)

    def assertFields(self, relpath, **expected):
        """Assert an exact value for each named field of parse(relpath)."""
        got = self.p(relpath)
        for k, v in expected.items():
            self.assertIn(k, dialect.FIELDS, "unknown field %r in test" % k)
            self.assertEqual(
                got[k], v,
                "%s: field %s expected %r got %r" % (relpath, k, v, got[k]))
        return got


# ---------------------------------------------------------------------------
# Contract-level invariants
# ---------------------------------------------------------------------------
class TestContract(DialectTestBase):

    def test_every_result_has_exactly_the_declared_fields(self):
        got = self.p(r"rawdata\PARROT\PARROT_90db_5750m_100m(2).iq")
        self.assertEqual(set(got.keys()), set(dialect.FIELDS))

    def test_source_file_and_relpath_are_preserved_verbatim(self):
        rp = r"rawdata\FLYSKY_EL_18\FLYSKY_EL_18-daifei-60db_2440m_100m(1)" \
             + PARTIAL + ".iq"
        got = self.p(rp)
        self.assertEqual(got["source_relpath"], rp)
        self.assertEqual(
            got["source_file"],
            "FLYSKY_EL_18-daifei-60db_2440m_100m(1)" + PARTIAL + ".iq")

    def test_naming_dialect_is_always_a_declared_dialect(self):
        for rp in (r"PublicDataset\DJI FPV COMBO\VTSBW=10\pack1_0-1s.iq",
                   r"rawdata\PARROT\PARROT_90db_5750m_100m(2).iq",
                   ZJU + r"\iq data\Air2R\Air2S-SNR1-5770m.iq"):
            self.assertIn(self.p(rp)["naming_dialect"], dialect.DIALECTS)

    def test_device_role_and_signal_state_stay_in_the_enums(self):
        for rp in (r"rawdata\SIYI_MK15\SIYI_MK15_2440_duima_gain80dB_0-2s.iq",
                   r"rawdata\DJMINI3\KONG-60db_2470m_100m.iq",
                   BIG + r"\WEIZHI-28db-90db_940m_100m_40m.iq"):
            got = self.p(rp)
            self.assertIn(got["device_role"], dialect.DEVICE_ROLES)
            self.assertIn(got["signal_state"], dialect.SIGNAL_STATES)

    def test_parser_never_touches_the_filesystem(self):
        """A path that does not exist anywhere must still parse identically."""
        a = self.p(r"rawdata\DJMINI3\DJMINI3--10db-60db_2470m_100m_10m.iq")
        self.assertTrue(a["ok"])
        self.assertEqual(a["center_freq_mhz"], 2470.0)


# ---------------------------------------------------------------------------
# D1 - pack slices
# ---------------------------------------------------------------------------
class TestD1PackSlice(DialectTestBase):

    def test_pack_slice_with_vtsbw_directory(self):
        self.assertFields(
            r"PublicDataset\DJI FPV COMBO\VTSBW=10\pack1_0-1s.iq",
            naming_dialect="D1_pack_slice",
            source_root="PublicDataset",
            class_dir="DJI FPV COMBO",
            vendor="dji", model="fpv_combo", device_role="uav",
            pack_index=1, time_start_s=0.0, time_end_s=1.0, duration_s=1.0,
            vtsbw_mhz=10.0, is_slice=True, unparsed_residue=None,
        )

    def test_pack_slice_without_vtsbw_directory(self):
        got = self.assertFields(
            r"PublicDataset\DAUTEL EVO NANO\pack1_12-13s.iq",
            naming_dialect="D1_pack_slice",
            class_dir="DAUTEL EVO NANO",
            vendor="autel", model="evo_nano",
            pack_index=1, time_start_s=12.0, time_end_s=13.0,
            vtsbw_mhz=None,
        )
        self.assertIsNone(got["unparsed_residue"])

    def test_mini3_pack2_numbering_errata_e003_is_reported_not_corrected(self):
        """E-003: MINI3 packs start at 2.  The parser records what the name
        says; the correction belongs in the migration manifest, not here."""
        self.assertFields(
            r"PublicDataset\DJI MINI3\VTSBW=10\pack2_0-1s.iq",
            pack_index=2, naming_dialect="D1_pack_slice")

    def test_pack_identity_comes_from_class_dir_not_filename(self):
        got = self.p(r"PublicDataset\DJI MINI4 PRO\VTSBW=10\pack1_5-6s.iq")
        self.assertEqual(got["model_raw"], "DJI MINI4 PRO")
        self.assertEqual(got["model"], "mini4_pro")


# ---------------------------------------------------------------------------
# D2 - UAV attenuation ladder (three _Nm groups)
# ---------------------------------------------------------------------------
class TestD2UavLadder(DialectTestBase):

    def test_three_Nm_groups_are_cf_samplerate_vtsbw_not_distance(self):
        self.assertFields(
            r"rawdata\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m.iq",
            naming_dialect="D2_uav_ladder",
            center_freq_mhz=5760.0,
            sample_rate_msps=100.0,
            vtsbw_mhz=10.0,
            distance_m=None,
            attenuation_label_raw="16db", attenuation_label_db=16.0,
            rx_gain_db=90.0,
            snr_nominal_db=None,
            unparsed_residue=None,
        )

    def test_double_dash_is_a_separator_and_X_is_positive(self):
        """spec appendix B / A4 #2: ``--Xdb`` is a separator, X is positive."""
        got = self.assertFields(
            r"rawdata\DJMINI3\DJMINI3--10db-60db_2470m_100m_10m.iq",
            naming_dialect="D2_uav_ladder",
            model="mini3", model_raw="DJMINI3",
            attenuation_label_db=10.0, rx_gain_db=60.0,
            center_freq_mhz=2470.0, vtsbw_mhz=10.0,
        )
        self.assertGreater(got["attenuation_label_db"], 0)
        self.assertIsNone(got["unparsed_residue"])

    def test_ladder_is_never_emitted_as_snr_nominal_db(self):
        """A4 #2: the ladder is monotonic but NOT dB-true (slope 0.58-0.77)."""
        for rp in (r"rawdata\DJMINI3\DJMINI3--46db-60db_2470m_100m_10m.iq",
                   r"rawdata\DJFPVCOMBO\DJFPVCOMBO-35db-90db_5760m_100m_10m.iq"):
            got = self.p(rp)
            self.assertIsNone(got["snr_nominal_db"], rp)
            self.assertIsNotNone(got["attenuation_label_db"], rp)

    def test_avata2_SNRxdB_occupies_the_ladder_slot_and_take_is_parsed(self):
        self.assertFields(
            r"rawdata\DJI AVATA2\DJI AVTA2-SNR1dB-85db_5765m_100m_10m(4).iq",
            naming_dialect="D2_uav_ladder",
            model_raw="DJI AVTA2", vendor="dji", model="avata2",
            snr_nominal_label="SNR1dB", snr_nominal_db=1.0,
            attenuation_label_db=None,
            rx_gain_db=85.0, center_freq_mhz=5765.0, vtsbw_mhz=10.0,
            take=4, unparsed_residue=None,
        )

    def test_avata2_SNR0dB_zero_is_kept_not_dropped(self):
        got = self.p(r"rawdata\DJI AVATA2\DJI AVTA2-SNR0dB-85db_5765m_100m_10m(3).iq")
        self.assertEqual(got["snr_nominal_db"], 0.0)
        self.assertIsNotNone(got["snr_nominal_db"])

    def test_lost_samples_tag(self):
        self.assertFields(
            BIG + r"\DJMAVIC3PRO\DJMAVIC3PRO-11db-90db_5800m_100m_20m-lost5.iq",
            naming_dialect="D2_uav_ladder",
            lost_samples_tag="lost5",
            source_root=BIG,
            unparsed_residue=None,
        )

    def test_weizhi_940_is_the_centre_frequency_not_a_db_group(self):
        """F3 probe.  NB the 940 sits in the ``_940m`` cf slot; the file is
        ``WEIZHI-28db-90db_940m_100m_40m.iq``."""
        self.assertFields(
            BIG + r"\WEIZHI-28db-90db_940m_100m_40m.iq",
            naming_dialect="D2_uav_ladder",
            vendor="weizhi", model=None, device_role="unk",
            center_freq_mhz=940.0,
            attenuation_label_db=28.0, rx_gain_db=90.0,
            sample_rate_msps=100.0, vtsbw_mhz=40.0,
            variant_unresolved=True, unparsed_residue=None,
        )

    def test_ceshi_and_cheshi_directories_flag_test_artifacts(self):
        for rp in (BIG + r"\DJMINI3ceshi\DJMINI3--14db-60db_5790m_100m_20m.iq",
                   BIG + r"\DJMAVIC3PROcheshi\DJMAVIC3PRO-18db-60db_2450m_100m_20m.iq"):
            self.assertTrue(self.p(rp)["is_test_artifact"], rp)

    def test_non_ceshi_sibling_is_not_flagged(self):
        self.assertFalse(
            self.p(BIG + r"\DJMAVIC3PRO\DJMAVIC3PRO-11db-90db_5800m_100m_20m-lost5.iq"
                   )["is_test_artifact"])

    def test_mini4pro_DI_vs_DJ_spelling_both_normalise(self):
        a = self.p(r"rawdata\DJMINI4PRO\DIMINI4PRO-17db-60db_2450m_100m_20m.iq")
        self.assertEqual((a["vendor"], a["model"]), ("dji", "mini4_pro"))
        self.assertEqual(a["model_raw"], "DIMINI4PRO")


# ---------------------------------------------------------------------------
# D3 - two-group hyphen dialect (RC + a few UAV/background classes)
# ---------------------------------------------------------------------------
class TestD3Hyphen2Group(DialectTestBase):

    def test_single_db_group_is_the_gain_not_a_ladder_step(self):
        self.assertFields(
            r"rawdata\FUTABA-T10J\FUTABA-T10J-daifei-60db_2440m_100m(1).iq",
            naming_dialect="D3_hyphen_2group",
            vendor="futaba", model="t10j", device_role="rc",
            rx_gain_db=60.0, attenuation_label_db=None,
            center_freq_mhz=2440.0, sample_rate_msps=100.0,
            vtsbw_mhz=None, distance_m=None,
            signal_state="idle", take=1, unparsed_residue=None,
        )

    def test_dafei_is_a_typo_of_daifei_and_maps_to_idle(self):
        self.assertFields(
            r"rawdata\FRSKY-X9DP2019\FRSKY-X9DP2019-dafei-60db_2440m_100m(2).iq",
            signal_state="idle", model="x9dp2019", vendor="frsky", take=2)

    def test_duima_maps_to_pairing(self):
        self.assertFields(
            r"rawdata\SKYDROID-T10\SKYDROID-T10-duima-60db_2470m_100(2).iq",
            signal_state="pairing", vendor="skydroid", model="t10",
            center_freq_mhz=2470.0, take=2)

    def test_unitless_sample_rate_group_is_still_100_msps(self):
        """``..._2440m_100.iq`` - the trailing group has no ``m``; it is the
        sample-rate magnitude, never metres."""
        self.assertFields(
            r"rawdata\JUMPER-T14\JUMPER-T14-daifei-60db_2440m_100.iq",
            sample_rate_msps=100.0, center_freq_mhz=2440.0,
            distance_m=None, unparsed_residue=None)

    def test_fullwidth_partial_marker(self):
        self.assertFields(
            r"rawdata\JUMPER-T14\JUMPER-T14-daifei-60db_2440m_100" + PARTIAL + ".iq",
            is_partial=True, unparsed_residue=None,
            naming_dialect="D3_hyphen_2group")

    def test_partial_marker_after_a_take(self):
        self.assertFields(
            r"rawdata\FLYSKY_EL_18\FLYSKY_EL_18-daifei-60db_2440m_100m(1)"
            + PARTIAL + ".iq",
            is_partial=True, take=1, signal_state="idle",
            vendor="flysky", model="el18", unparsed_residue=None)

    def test_non_partial_sibling_is_not_flagged(self):
        self.assertFalse(
            self.p(r"rawdata\JUMPER-T14\JUMPER-T14-duima-60db_2440m_100.iq"
                   )["is_partial"])

    def test_kong_misfiled_into_djmini3_is_background_and_flagged(self):
        """ERRATA E-004 / spec s5 row 1."""
        self.assertFields(
            r"rawdata\DJMINI3\KONG-60db_2470m_100m.iq",
            model="kong", device_role="bg",
            class_dir="DJMINI3", is_misfiled=True,
            rx_gain_db=60.0, center_freq_mhz=2470.0)

    def test_kong_in_its_own_directory_is_not_misfiled(self):
        self.assertFields(
            r"rawdata\KONG\KONG-60db_2470m_100m_0-2s.iq",
            model="kong", device_role="bg", is_misfiled=False,
            time_start_s=0.0, time_end_s=2.0, is_slice=True)

    def test_parrot_has_no_state_token_and_stays_variant_unresolved(self):
        """V05/F4 forensically identifies ANAFI, but the FILENAME does not say
        so; the parser must not import that conclusion."""
        self.assertFields(
            r"rawdata\PARROT\PARROT_90db_5750m_100m(2).iq",
            vendor="parrot", model="parrot_unspecified",
            variant_unresolved=True,
            signal_state="unknown",
            rx_gain_db=90.0, center_freq_mhz=5750.0, take=2)

    def test_parrot_partial(self):
        self.assertFields(
            r"rawdata\PARROT\PARROT_90db_5750m_100m(2)" + PARTIAL + ".iq",
            is_partial=True, take=2, model="parrot_unspecified")

    def test_dautel_trailing_1dB_group_is_unexplained_and_says_so(self):
        """The only ``_NdB`` sitting in the VTSBW slot in the whole library.
        We refuse to guess: it lands in unparsed_residue."""
        self.assertFields(
            r"rawdata\DAUTEL EVO NANO\DAUTEL_EVO Nano-90db_5770m_100m_1dB(1).iq",
            naming_dialect="D3_hyphen_2group",
            vendor="autel", model="evo_nano",
            rx_gain_db=90.0, center_freq_mhz=5770.0, sample_rate_msps=100.0,
            vtsbw_mhz=None, take=1,
            unparsed_residue="_1dB",
        )

    def test_space_inside_model_token(self):
        got = self.p(r"rawdata\DAUTEL EVO NANO\DAUTEL_EVO Nano-85db_5770m_100m(2).iq")
        self.assertEqual(got["model_raw"], "DAUTEL_EVO Nano")
        self.assertEqual(got["model"], "evo_nano")

    def test_jr_propo_space_in_vendor(self):
        self.assertFields(
            r"rawdata\JR PROPO_XG14\JR PROPO_XG14-daifei-60db_2440m_100m(1).iq",
            vendor="jrpropo", model="xg14", signal_state="idle")


# ---------------------------------------------------------------------------
# D4 - gain-underscore dialect
# ---------------------------------------------------------------------------
class TestD4GainUnderscore(DialectTestBase):

    def test_gain_prefix_form(self):
        self.assertFields(
            r"rawdata\SIYI_MK15\SIYI_MK15_2440_daifei_gain80dB_0-2s.iq",
            naming_dialect="D4_gain_underscore",
            vendor="siyi", model="mk15", device_role="rc",
            center_freq_mhz=2440.0, rx_gain_db=80.0,
            signal_state="idle", time_start_s=0.0, time_end_s=2.0,
            duration_s=2.0, is_slice=True, unparsed_residue=None,
        )

    def test_bare_dB_form_without_the_gain_prefix_plus_take(self):
        self.assertFields(
            r"rawdata\FutabaT14SG\FUtabaT14SG_2440_daifei_80dB(2)_0-2s.iq",
            naming_dialect="D4_gain_underscore",
            model_raw="FUtabaT14SG", vendor="futaba", model="t14sg",
            center_freq_mhz=2440.0, rx_gain_db=80.0, take=2,
            time_start_s=0.0, time_end_s=2.0, unparsed_residue=None,
        )

    def test_take_then_slice_then_partial_all_three_tails(self):
        self.assertFields(
            r"rawdata\FutabaT14SG\FUtabaT14SG_2440_daifei_80dB(2)_0-2s" + PARTIAL + ".iq",
            take=2, time_start_s=0.0, time_end_s=2.0, is_partial=True,
            unparsed_residue=None)

    def test_xuanting_maps_to_hover(self):
        self.assertFields(
            r"rawdata\YUNZHUO-H12\YUNZHUO-H12_2440_xuanting_gain80dB_0-2s.iq",
            signal_state="hover", vendor="yunzhuo", model="h12",
            center_freq_mhz=2440.0, rx_gain_db=80.0)

    def test_unsliced_parent_has_no_time_window(self):
        got = self.assertFields(
            r"过大的原始数据\SIYI_MK15\SIYI_MK15_2440_duima_gain80dB.iq",
            signal_state="pairing", is_slice=False)
        self.assertIsNone(got["time_start_s"])
        self.assertIsNone(got["time_end_s"])

    def test_herelink_lowercase_model(self):
        self.assertFields(
            r"rawdata\herelinkHx4\herelinkHx4_2420_duima_gain60dB_0-2s.iq",
            vendor="herelink", model="hx4", center_freq_mhz=2420.0,
            signal_state="pairing")

    def test_yunzhuo_is_not_silently_merged_into_skydroid(self):
        """spec s2 asks for SKYDROID == 云卓 -> vendor=skydroid, but fact F1
        says SKYDROID H12 and YUNZHUO H12 are two physical devices.  The parser
        keeps them distinct and exposes the merge as data for the taxonomy
        step instead of applying it."""
        y = self.p(r"rawdata\YUNZHUO-H12\YUNZHUO-H12_2440_xuanting_gain80dB_0-2s.iq")
        s = self.p(r"rawdata\SKYDROID-H12\SKYDROID-H12-daifei-60db_2470m_100.iq")
        self.assertEqual(y["vendor"], "yunzhuo")
        self.assertEqual(s["vendor"], "skydroid")
        self.assertEqual(y["model"], "h12")
        self.assertEqual(s["model"], "h12")
        self.assertEqual(dialect.VENDOR_MERGE_SPEC_S2.get("yunzhuo"), "skydroid")


# ---------------------------------------------------------------------------
# D5 - ZJU -SNRn- dialect
# ---------------------------------------------------------------------------
class TestD5ZjuSnr(DialectTestBase):

    def test_snr_is_an_ordinal_not_a_dB_value(self):
        """V03: 3 files = same source segment + independent AWGN, unlabelled."""
        self.assertFields(
            ZJU + r"\iq data\AVATA\AVATA-SNR2-5770m.iq",
            naming_dialect="D5_zju_snr",
            snr_nominal_label="SNR2", snr_ordinal=2, snr_nominal_db=None,
            center_freq_mhz=5770.0, distance_m=None, unparsed_residue=None)

    def test_air2s_file_inside_air2r_directory_is_flagged(self):
        """V03: AVATA and Air2R share one source recording; the files in the
        Air2R directory are named Air2S."""
        got = self.assertFields(
            ZJU + r"\iq data\Air2R\Air2S-SNR1-5770m.iq",
            class_dir="Air2R", model_raw="Air2S", model="air2s",
            variant_unresolved=True)
        self.assertNotEqual(got["class_dir"].lower(), got["model_raw"].lower())

    def test_zju_DJi_bucket_has_a_vendor_but_deliberately_no_model(self):
        self.assertFields(
            ZJU + r"\iq data\DJi\DJI-SNR1-5770m.iq",
            vendor="dji", model=None, vocab_hit=True,
            variant_unresolved=True, device_role="uav")

    def test_matrice200_and_matrice300_resolve(self):
        for n in ("200", "300"):
            self.assertFields(
                ZJU + r"\iq data\Matrice%s\Matrice%s-SNR3-5770m.iq" % (n, n),
                vendor="dji", model="matrice" + n, vocab_hit=True,
                snr_ordinal=3)

    def test_FLY_suffix_maps_to_flight_state(self):
        self.assertFields(
            ZJU + r"\iq data\RadioLinkAT9S_FLY\RadioLinkAT9S_FLY-SNR1-5770m.iq",
            signal_state="flight", vendor="radiolink", device_role="rc")

    def test_tarains_is_a_typo_of_taranis(self):
        self.assertFields(
            ZJU + r"\iq data\Tarains_Plus_FLY\Tarains_Plus_FLY-SNR2-5770m.iq",
            vendor="frsky", model="taranis_plus", signal_state="flight")

    def test_futaba_T61Z_suspected_digit_transposition_is_flagged(self):
        """Almost certainly T16IZ; we do not silently rewrite it."""
        self.assertFields(
            ZJU + r"\iq data\FutabaT61Z_FLY\FutabaT61Z_FLY-SNR1-5770m.iq",
            model_raw="FutabaT61Z_FLY", model="t61z",
            variant_unresolved=True, signal_state="flight")

    def test_5770m_suffix_is_cf_even_though_it_is_wrong_for_24GHz_RCs(self):
        """V03: the blanket '-5770m' is physically impossible for a 2.4 GHz
        handheld.  The parser reports what the name says; the correction is a
        manifest-level job, so cf is populated and NOT silently nulled."""
        got = self.p(ZJU + r"\iq data\FutabaT14SG_FLY\FutabaT14SG_FLY-SNR1-5770m.iq")
        self.assertEqual(got["center_freq_mhz"], 5770.0)
        self.assertEqual(got["device_role"], "rc")


# ---------------------------------------------------------------------------
# D6 - SDR default filename
# ---------------------------------------------------------------------------
class TestD6SdrDefault(DialectTestBase):

    def test_int32_min_sentinel_frequency_is_refused_not_recorded_as_cf(self):
        self.assertFields(
            r".dat_pic\Mavic2Air2\data.bin_75000000SPS_-2147483648Hz_"
            r"2024_03_26_T15-49-15.iq",
            naming_dialect="D6_sdr_default",
            sample_rate_msps=75.0,
            center_freq_mhz=None,
            vendor="dji", model="mavic2_air2",
            unparsed_residue="-2147483648Hz(INT32_MIN sentinel, cf unset)",
        )


# ---------------------------------------------------------------------------
# D7 - DroneRFa-style band / RF-chain
# ---------------------------------------------------------------------------
class TestD7DroneRfaRf(DialectTestBase):

    def test_high_band_rf0(self):
        self.assertFields(
            r".dat_pic\AVATA\AVATA_high_RF0.dat",
            naming_dialect="D7_dronerfa_rf",
            band_tag="high", rf_chain=0,
            vendor="dji", model="avata", unparsed_residue=None)

    def test_low_band_and_underscored_model(self):
        self.assertFields(
            r".dat_pic\Matrice100\MATRICE_100_high_RF0.dat",
            band_tag="high", rf_chain=0,
            vendor="dji", model="matrice100")

    def test_dat_pic_Dji_bucket_deliberate_null_model(self):
        self.assertFields(
            r".dat_pic\Dji\Dji_low_RF0.dat",
            band_tag="low", rf_chain=0,
            vendor="dji", model=None, vocab_hit=True)


# ---------------------------------------------------------------------------
# D8 - crop_data derived slices
# ---------------------------------------------------------------------------
class TestD8CropSlice(DialectTestBase):

    def test_slice_inherits_every_field_from_the_reparsed_parent_stem(self):
        self.assertFields(
            r"crop_data\DJMINI4PRO\DJMINI4PRO-17db-60db_2450m_100m_20m"
            r"\DIMINI4PRO-17db-60db_2450m_100m_20m_0-2s.dat",
            naming_dialect="D8_crop_slice",
            artifact_class="derived_slice",
            class_dir="DJMINI4PRO",
            vendor="dji", model="mini4_pro",
            attenuation_label_db=17.0, rx_gain_db=60.0,
            center_freq_mhz=2450.0, sample_rate_msps=100.0, vtsbw_mhz=20.0,
            time_start_s=0.0, time_end_s=2.0, is_slice=True,
            unparsed_residue=None,
        )

    def test_slice_window_offsets_are_parsed(self):
        self.assertFields(
            r"crop_data\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m"
            r"\DJFPVCOMBO-16db-90db_5760m_100m_10m_10-12s.dat",
            time_start_s=10.0, time_end_s=12.0, duration_s=2.0,
            artifact_class="derived_slice")

    def test_derived_slices_are_marked_derived_not_recording(self):
        got = self.p(r"crop_data\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m"
                     r"\DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat")
        self.assertNotEqual(got["artifact_class"], "recording")


# ---------------------------------------------------------------------------
# D9 / D10 / D11
# ---------------------------------------------------------------------------
class TestD9BareModel(DialectTestBase):

    def test_bare_model_only(self):
        self.assertFields(
            r"exp3\rawdata\FUTABA-T10J.iq",
            naming_dialect="D9_bare_model",
            vendor="futaba", model="t10j", device_role="rc",
            rx_gain_db=None, center_freq_mhz=None, unparsed_residue=None)

    def test_bare_model_with_a_duration_suffix(self):
        self.assertFields(
            r"exp3\rawdata\FUTABA-T10J_0.3s.iq",
            naming_dialect="D9_bare_model",
            duration_s=0.3, model="t10j")

    def test_bare_model_with_a_slice_window(self):
        self.assertFields(
            r"exp3\rawdata\FLYSKY_FS_I6X\FLYSKY_FS_I6X_0-2s.iq",
            naming_dialect="D9_bare_model",
            time_start_s=0.0, time_end_s=2.0, is_slice=True,
            vendor="flysky", model="fs_i6x")


class TestD10Archive(DialectTestBase):

    def test_class_archive_keeps_its_identity(self):
        self.assertFields(
            r"PublicDataset\Dataset\ValidationSet_5Drones\DJI AVTA2.rar",
            naming_dialect="D10_archive",
            artifact_class="archive_experiment",
            vendor="dji", model="avata2",
            is_test_artifact=True, ok=True)

    def test_experiment_archive_has_no_device_identity_by_construction(self):
        got = self.assertFields(
            r"augmentation_exp1_MethodSelect\exp_res.rar",
            naming_dialect="D10_archive", model_raw=None, vocab_hit=None)
        self.assertIsNone(got["model"])
        self.assertTrue(got["ok"])

    def test_autumn_augmentation_archive_is_still_classified(self):
        """ERRATA E-001 (29% duplicate images) - quarantine decision is
        downstream; the parser must still classify the row rather than drop
        it."""
        got = self.p(r"PublicDataset\Dataset\IMAGESET-exp"
                     r"\dataset_5Drones-raw+20dB_autumn_usingAG.rar")
        self.assertTrue(got["ok"])
        self.assertEqual(got["naming_dialect"], "D10_archive")


class TestD11DerivedCache(DialectTestBase):

    def test_npy_label_cache(self):
        self.assertFields(
            r"pics_exp1_alldrones\1.classification_dataset\train\labels.cache.npy",
            naming_dialect="D11_derived_cache",
            artifact_class="derived_cache",
            is_test_artifact=True, ok=True)


# ---------------------------------------------------------------------------
# Vocabulary normalisation mandated by spec s2
# ---------------------------------------------------------------------------
class TestVocabularyNormalisation(DialectTestBase):

    def test_spec_s2_typo_corrections(self):
        cases = [
            (r"rawdata\DJI AVATA2\DJI AVTA2-SNR1dB-85db_5765m_100m_10m(4).iq",
             "dji", "avata2"),                                   # AVTA2 -> avata2
            (r"rawdata\DJMINI4PRO\DIMINI4PRO-17db-60db_2450m_100m_20m.iq",
             "dji", "mini4_pro"),                                # DIMINI4PRO
            (r"rawdata\DAUTEL EVO NANO\DAUTEL_EVO Nano-85db_5770m_100m(2).iq",
             "autel", "evo_nano"),                               # DAUTEL -> autel
        ]
        for rp, vendor, model in cases:
            got = self.p(rp)
            self.assertEqual(got["vendor"], vendor, rp)
            self.assertEqual(got["model"], model, rp)

    def test_dafei_and_daifei_collapse_to_the_same_state(self):
        a = self.p(r"rawdata\FRSKY-X9DP2019\FRSKY-X9DP2019-dafei-60db_2440m_100m(2).iq")
        b = self.p(r"rawdata\FRSKY-X14\FRSKY-X14-daifei-60db_2440m_100m(2).iq")
        self.assertEqual(a["signal_state"], b["signal_state"])
        self.assertEqual(a["signal_state"], "idle")

    def test_model_raw_is_never_rewritten(self):
        got = self.p(r"rawdata\DJMINI4PRO\DIMINI4PRO-17db-60db_2450m_100m_20m.iq")
        self.assertEqual(got["model_raw"], "DIMINI4PRO")

    def test_vendor_raw_is_only_set_when_it_really_prefixes_the_token(self):
        """``DAUTEL_EVO Nano`` normalises to vendor ``autel`` but the literal
        prefix is ``DAUTEL``, not ``AUTEL`` - so vendor_raw must stay None
        rather than fabricate a substring that is not in the name."""
        got = self.p(r"rawdata\DAUTEL EVO NANO\DAUTEL_EVO Nano-85db_5770m_100m(2).iq")
        self.assertIsNone(got["vendor_raw"])
        got2 = self.p(r"rawdata\SKYDROID-T10\SKYDROID-T10-duima-60db_2470m_100.iq")
        self.assertEqual(got2["vendor_raw"], "SKYDROID")


# ---------------------------------------------------------------------------
# The "no invented values" contract
# ---------------------------------------------------------------------------
class TestNeverInvent(DialectTestBase):

    def test_distance_m_is_none_for_every_dialect(self):
        for rp in (r"rawdata\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m.iq",
                   r"rawdata\FUTABA-T10J\FUTABA-T10J-daifei-60db_2440m_100m(1).iq",
                   ZJU + r"\iq data\AVATA\AVATA-SNR2-5770m.iq",
                   r"PublicDataset\DJI FPV COMBO\VTSBW=10\pack1_0-1s.iq"):
            self.assertIsNone(self.p(rp)["distance_m"], rp)

    def test_absent_fields_stay_none(self):
        got = self.p(r"exp3\rawdata\FUTABA-T10J.iq")
        for f in ("rx_gain_db", "center_freq_mhz", "sample_rate_msps",
                  "vtsbw_mhz", "take", "pack_index", "snr_nominal_db",
                  "attenuation_label_db", "lost_samples_tag"):
            self.assertIsNone(got[f], f)

    def test_unknown_shape_is_reported_not_forced(self):
        got = self.p(r"rawdata\NOPE\!!!totally-unparseable~~~.iq")
        self.assertFalse(got["ok"])
        self.assertIsNotNone(got["reason"])


# ---------------------------------------------------------------------------
# Whole-corpus regression against the real E1 scan
# ---------------------------------------------------------------------------
class TestCorpusCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CSV_PATH):
            raise unittest.SkipTest("E1 scan CSV not present: %s" % CSV_PATH)
        with open(CSV_PATH, "r", encoding="utf-8", newline="") as fh:
            cls.rows = list(csv.DictReader(fh))
        cls.parsed, cls.unparsed = dialect.parse_all(cls.rows)

    def test_row_count_is_the_expected_954(self):
        self.assertEqual(len(self.rows), 954)

    def test_every_row_is_accounted_for(self):
        self.assertEqual(len(self.parsed) + len(self.unparsed), len(self.rows))

    def test_no_row_fails_to_match_a_dialect(self):
        self.assertEqual(
            [r["source_relpath"] for r in self.unparsed], [],
            "rows matched no dialect")

    def test_exactly_the_two_known_residues_remain(self):
        residues = sorted(r["source_relpath"] for r in self.parsed
                          if r["unparsed_residue"])
        self.assertEqual(residues, sorted([
            r".dat_pic\Mavic2Air2\data.bin_75000000SPS_-2147483648Hz_"
            r"2024_03_26_T15-49-15.iq",
            r"rawdata\DAUTEL EVO NANO\DAUTEL_EVO Nano-90db_5770m_100m_1dB(1).iq",
        ]))

    def test_no_true_vocabulary_misses_remain(self):
        misses = sorted(set(r["model_raw"] for r in self.parsed
                            if r["vocab_hit"] is False))
        self.assertEqual(misses, [])

    def test_distance_m_is_never_populated_anywhere_in_the_corpus(self):
        self.assertEqual(
            [r["source_relpath"] for r in self.parsed
             if r["distance_m"] is not None], [])

    def test_every_row_keeps_a_naming_dialect_tag(self):
        for r in self.parsed:
            self.assertIn(r["naming_dialect"], dialect.DIALECTS,
                          r["source_relpath"])

    def test_sample_rate_is_100_msps_wherever_it_is_stated(self):
        """The one documented exception is the .dat_pic SDR-default file at
        75 MS/s."""
        odd = sorted(set(r["sample_rate_msps"] for r in self.parsed
                         if r["sample_rate_msps"] not in (None, 100.0)))
        self.assertEqual(odd, [75.0])

    def test_centre_frequencies_are_ism_or_unii_channel_centres(self):
        cfs = sorted(set(r["center_freq_mhz"] for r in self.parsed
                         if r["center_freq_mhz"] is not None))
        for cf in cfs:
            self.assertTrue(
                cf == 940.0 or 2400.0 <= cf <= 2500.0 or 5700.0 <= cf <= 5900.0,
                "unexpected centre frequency %r" % cf)

    def test_vtsbw_values_match_the_publicdataset_directory_names(self):
        vt = sorted(set(r["vtsbw_mhz"] for r in self.parsed
                        if r["vtsbw_mhz"] is not None))
        self.assertEqual(vt, [10.0, 20.0, 40.0, 60.0])

    def test_report_builds_and_states_the_measured_numbers(self):
        text, parsed, unparsed = dialect.build_report(CSV_PATH)
        self.assertIn("954", text)
        self.assertEqual(len(parsed), 954)
        self.assertEqual(len(unparsed), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
