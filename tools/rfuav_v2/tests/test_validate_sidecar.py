#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for tools/rfuav_v2/validate_sidecar.py.

One golden fixture that MUST validate with zero findings, then one
deliberately broken mutation per rule.  Every negative test asserts on the
specific rule id, so a test failing tells you which contract broke.

Stdlib only, python >= 3.9:
    python3 tools/rfuav_v2/tests/test_validate_sidecar.py
    python3 -m unittest discover -s tools/rfuav_v2/tests
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLDIR = os.path.dirname(_HERE)
if _TOOLDIR not in sys.path:
    sys.path.insert(0, _TOOLDIR)

import validate_sidecar as V  # noqa: E402

SCHEMA = V.load_schema()

SESSION_ID = "dji-mini3-u01-r01-s01-d20240910-01"
RECORDING_ID = SESSION_ID + "-c00"
META_NAME = RECORDING_ID + V.META_EXT

# 1000 S/s * 8 bytes * 0.064 s = 512 bytes = 64 cf32 samples.
SAMPLE_RATE = 1000.0
NUM_SAMPLES = 64
DURATION_S = 0.064
SIZE_BYTES = 512

PAYLOAD = b"\x00\x01\x02\x03" * (SIZE_BYTES // 4)
PAYLOAD_SHA512 = hashlib.sha512(PAYLOAD).hexdigest()


def golden():
    """A fully compliant document.  Returns a fresh deep copy every call."""
    return {
        "global": {
            "core:datatype": "cf32_le",
            "core:version": "1.2.6",
            "core:sample_rate": SAMPLE_RATE,
            "core:num_channels": 1,
            "core:sha512": PAYLOAD_SHA512,
            "core:hw": "USRP X310 + UBX-160, 60 dB rx gain",
            "core:author": "RFUAV maintainers <rfuav@example.org>",
            "core:license": "https://creativecommons.org/licenses/by/4.0/",
            "core:recorder": "rfuav-migrate 2.0.0 (sigmf-python 1.11.1)",
            "core:description": "DJI Mini 3 downlink, factory default state.",
            "core:extensions": [
                {"name": "rfuav", "version": "0.1.0", "optional": True},
            ],
            "rfuav:taxonomy_id": "uav.dji.mini3",
            "rfuav:recording_id": RECORDING_ID,
            "rfuav:session_id": SESSION_ID,
            "rfuav:model": "Mini 3",
            "rfuav:device_role": "uav",
            "rfuav:device_unit_id": "u01",
            "rfuav:rx_id": "r01",
            "rfuav:site_id": "s01",
            "rfuav:signal_state": "flight",
            "rfuav:link_protocol": "dji.ocusync2",
            "rfuav:band_captured": "2g4",
            "rfuav:bands_capable": ["2g4", "5g8"],
            "rfuav:protocol_mode_at_capture": "factory_default_unverified",
            "rfuav:variant_unresolved": False,
            "rfuav:source_dataset": "rfuav",
            "rfuav:original_license": "CC-BY-4.0",
            "rfuav:redistributable": True,
            "rfuav:attribution": "RFUAV v1 (Wang et al.), CC BY 4.0.",
            "rfuav:naming_dialect": "rfuav_pack",
            "rfuav:continuous": True,
            "rfuav:gap_boundaries": [],
            "rfuav:quarantine_status": "none",
            "rfuav:duration_s": DURATION_S,
            "rfuav:num_samples": NUM_SAMPLES,
            "rfuav:size_bytes": SIZE_BYTES,
            "rfuav:snr_nominal_db": None,
            "rfuav:recording_index": 0,
            "rfuav:capture_date": "2024-09-10",
            "rfuav:errata_refs": ["RFUAV-E-003"],
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": 2440000000.0,
                "core:datetime": "2024-09-10T13:04:05Z",
                "rfuav:vtsbw_hz": 100000000.0,
            },
        ],
        "annotations": [
            {
                "core:sample_start": 0,
                "core:sample_count": NUM_SAMPLES,
                "core:freq_lower_edge": 2430000000.0,
                "core:freq_upper_edge": 2450000000.0,
                "core:label": "uav.dji.mini3",
            },
        ],
    }


class Base(unittest.TestCase):

    maxDiff = None

    def run_doc(self, doc, meta_name=META_NAME, data_path=None):
        return V.validate_document(doc, SCHEMA, meta_name=meta_name,
                                   data_path=data_path)

    def rules(self, findings, level=None):
        return sorted(set(f.rule for f in findings
                          if level is None or f.level == level))

    def assert_clean(self, findings, note=""):
        if findings:
            self.fail("expected no findings %s, got:\n  %s" % (
                note, "\n  ".join(f.format() for f in findings)))

    def assert_rule(self, findings, rule, level=V.ERROR, needle=None):
        hits = [f for f in findings if f.rule == rule and f.level == level]
        if not hits:
            self.fail("expected a %s finding for rule %s, got:\n  %s" % (
                level, rule, "\n  ".join(f.format() for f in findings) or "<none>"))
        if needle is not None:
            joined = " | ".join(f.message for f in hits)
            self.assertIn(needle, joined)
        return hits


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------

class TestGolden(Base):

    def test_golden_document_is_clean(self):
        self.assert_clean(self.run_doc(golden()))

    def test_golden_is_stable_under_deepcopy(self):
        first = golden()
        second = copy.deepcopy(first)
        self.assertEqual(first, second)
        self.assert_clean(self.run_doc(second))

    def test_optional_foreign_namespace_is_allowed_when_declared(self):
        doc = golden()
        doc["global"]["core:extensions"].append(
            {"name": "traceability", "version": "1.0.0", "optional": True})
        doc["global"]["traceability:origin"] = {
            "url": "file:///E:/Drone_dataset/RFUAV/rawdata/DJMINI3/pack1_0-1s.iq"}
        self.assert_clean(self.run_doc(doc))

    def test_snr_annotation_with_method_version_is_allowed(self):
        doc = golden()
        doc["annotations"][0]["rfuav:snr_est_v2_db"] = 21.4
        doc["annotations"][0]["rfuav:snr_est_v2_ci95"] = [20.9, 21.9]
        doc["annotations"][0]["rfuav:method_version"] = "fcme_v2.0"
        self.assert_clean(self.run_doc(doc))


# ---------------------------------------------------------------------------
# negative cases - one rule each
# ---------------------------------------------------------------------------

class TestSchemaRules(Base):
    """Rules that the JSON Schema itself carries."""

    def test_N01_bare_cf32_datatype_rejected(self):
        # sigmf-python's core:datatype pattern is unanchored: bare "cf32" passes
        # upstream.  The profile pins the exact string.
        doc = golden()
        doc["global"]["core:datatype"] = "cf32"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="cf32_le")

    def test_N02_datatype_with_trailing_garbage_rejected(self):
        doc = golden()
        doc["global"]["core:datatype"] = "cf32_lefoo"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="cf32_le")

    def test_N03_wrong_sigmf_version_rejected(self):
        doc = golden()
        doc["global"]["core:version"] = "1.0.0"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="1.2.6")

    def test_N04_uppercase_sha512_rejected(self):
        doc = golden()
        doc["global"]["core:sha512"] = PAYLOAD_SHA512.upper()
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="pattern")

    def test_N05_short_sha512_rejected(self):
        doc = golden()
        doc["global"]["core:sha512"] = PAYLOAD_SHA512[:127]
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="pattern")

    def test_N06_missing_recorder_rejected(self):
        doc = golden()
        del doc["global"]["core:recorder"]
        self.assert_rule(self.run_doc(doc), "SCHEMA",
                         needle="missing required key 'core:recorder'")

    def test_N07_license_must_be_a_url(self):
        doc = golden()
        doc["global"]["core:license"] = "CC BY 4.0"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="pattern")

    def test_N08_typo_in_core_key_rejected(self):
        # THE gap this layer exists for: sigmf-python validates this document.
        doc = golden()
        doc["global"]["core:sample_rte"] = 100000000.0
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="typo-guard")

    def test_N09_typo_in_rfuav_key_rejected(self):
        doc = golden()
        doc["global"]["rfuav:signl_state"] = "flight"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="typo-guard")

    def test_N10_unknown_signal_state_rejected(self):
        doc = golden()
        doc["global"]["rfuav:signal_state"] = "garbage"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="must be one of")

    def test_N11_zero_sample_rate_rejected(self):
        doc = golden()
        doc["global"]["core:sample_rate"] = 0
        self.assert_rule(self.run_doc(doc), "SCHEMA",
                         needle="exclusiveMinimum")

    def test_N12_naive_local_datetime_rejected(self):
        doc = golden()
        doc["captures"][0]["core:datetime"] = "2024-09-10T13:04:05"
        findings = self.run_doc(doc)
        self.assert_rule(findings, "SCHEMA", needle="pattern")
        self.assert_rule(findings, "X10", needle="trailing Z")

    def test_N13_offset_datetime_rejected(self):
        doc = golden()
        doc["captures"][0]["core:datetime"] = "2024-09-10T21:04:05+08:00"
        self.assert_rule(self.run_doc(doc), "X10", needle="offsets are rejected")

    def test_N14_malformed_session_id_rejected(self):
        doc = golden()
        broken = "dji-mini3-u01-r01-d20240910-01"  # no -sNN
        doc["global"]["rfuav:session_id"] = broken
        doc["global"]["rfuav:recording_id"] = broken + "-c00"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="pattern")

    def test_N15_single_digit_config_index_rejected(self):
        doc = golden()
        doc["global"]["rfuav:recording_id"] = SESSION_ID + "-c0"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="pattern")

    def test_N16_unknown_quarantine_status_rejected(self):
        doc = golden()
        doc["global"]["rfuav:quarantine_status"] = "deleted"
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="must be one of")

    def test_N17_top_level_extra_key_rejected(self):
        doc = golden()
        doc["collection"] = []
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="unknown key")

    def test_N18_snr_estimate_without_method_version_rejected(self):
        doc = golden()
        doc["annotations"][0]["rfuav:snr_est_v2_db"] = 21.4
        self.assert_rule(self.run_doc(doc), "SCHEMA",
                         needle="missing required key 'rfuav:method_version'")

    def test_N19_empty_captures_rejected(self):
        doc = golden()
        doc["captures"] = []
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="minItems")

    def test_N20_num_channels_must_be_one(self):
        doc = golden()
        doc["global"]["core:num_channels"] = 2
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="maximum")

    def test_N21_boolean_is_not_an_integer(self):
        # python bool is an int subclass; the evaluator must not be fooled.
        doc = golden()
        doc["global"]["rfuav:num_samples"] = True
        self.assert_rule(self.run_doc(doc), "SCHEMA", needle="expected type")


class TestCrossFieldRules(Base):
    """Rules JSON Schema structurally cannot express."""

    def test_X01_filename_must_match_recording_id(self):
        findings = self.run_doc(golden(), meta_name="pack1_0-1s.sigmf-meta")
        self.assert_rule(findings, "X01", needle="spec S2 requires")

    def test_X02_recording_id_must_extend_session_id(self):
        doc = golden()
        doc["global"]["rfuav:recording_id"] = \
            "dji-mini3-u01-r01-s01-d20240911-01-c00"
        findings = self.run_doc(doc, meta_name=None)
        self.assert_rule(findings, "X02", needle="session_id + '-cNN'")

    def test_X03_session_segments_must_match_scalar_fields(self):
        doc = golden()
        doc["global"]["rfuav:rx_id"] = "r07"
        self.assert_rule(self.run_doc(doc), "X03", needle="contradicts session_id")

    def test_X04_taxonomy_short_must_come_from_taxonomy_id(self):
        doc = golden()
        doc["global"]["rfuav:taxonomy_id"] = "uav.dji.avata2"
        doc["global"]["rfuav:device_role"] = "uav"
        self.assert_rule(self.run_doc(doc), "X04", needle="not segments")

    def test_X05_device_role_must_match_taxonomy_role(self):
        doc = golden()
        doc["global"]["rfuav:device_role"] = "rc"
        self.assert_rule(self.run_doc(doc), "X05", needle="role segment")

    def test_X06_missing_rfuav_extension_declaration(self):
        doc = golden()
        doc["global"]["core:extensions"] = [
            {"name": "traceability", "version": "1.0.0", "optional": True}]
        doc["global"]["traceability:origin"] = "x"
        self.assert_rule(self.run_doc(doc), "X06", needle="mandatory entry")

    def test_X06_extension_version_must_be_exactly_0_1_0(self):
        # verificacion-empirica-02 used "v0.1.0"; the profile says "0.1.0".
        doc = golden()
        doc["global"]["core:extensions"][0]["version"] = "v0.1.0"
        self.assert_rule(self.run_doc(doc), "X06", needle="version='v0.1.0'")

    def test_X06_extension_must_be_optional_true(self):
        doc = golden()
        doc["global"]["core:extensions"][0]["optional"] = False
        self.assert_rule(self.run_doc(doc), "X06", needle="optional=False")

    def test_X06_duplicate_namespace_declaration(self):
        doc = golden()
        doc["global"]["core:extensions"].append(
            {"name": "rfuav", "version": "0.1.0", "optional": True})
        self.assert_rule(self.run_doc(doc), "X06", needle="more than once")

    def test_X07_undeclared_namespace_is_an_error_not_a_warning(self):
        doc = golden()
        doc["global"]["signal:emitter"] = {"seid": 1}
        self.assert_rule(self.run_doc(doc), "X07", needle="not declared")

    def test_X08_first_capture_must_start_at_sample_zero(self):
        doc = golden()
        doc["captures"][0]["core:sample_start"] = 1024
        self.assert_rule(self.run_doc(doc), "X08", needle="MUST start at sample 0")

    def test_X08_captures_must_be_strictly_increasing(self):
        doc = golden()
        doc["captures"].append({"core:sample_start": 0,
                                "core:datetime": "2024-09-10T13:04:06Z"})
        self.assert_rule(self.run_doc(doc), "X08", needle="strictly")

    def test_X09_first_capture_must_carry_frequency(self):
        doc = golden()
        del doc["captures"][0]["core:frequency"]
        self.assert_rule(self.run_doc(doc), "X09", needle="core:frequency")

    def test_X10_impossible_calendar_date_rejected(self):
        doc = golden()
        doc["captures"][0]["core:datetime"] = "2024-02-30T00:00:00Z"
        self.assert_rule(self.run_doc(doc), "X10", needle="not a real calendar date")

    def test_X10_capture_timestamps_must_not_go_backwards(self):
        doc = golden()
        doc["captures"].append({"core:sample_start": 32,
                                "core:datetime": "2024-09-10T13:04:00Z"})
        self.assert_rule(self.run_doc(doc), "X10", needle="backwards")

    def test_X11_slicing_date_masquerading_as_capture_date(self):
        # audit A: the 11-26 mtimes are SLICING dates, not capture dates.
        doc = golden()
        doc["captures"][0]["core:datetime"] = "2024-11-26T09:00:00Z"
        self.assert_rule(self.run_doc(doc), "X11", needle="SLICING dates")

    def test_X11_midnight_crossing_is_only_a_warning(self):
        doc = golden()
        doc["captures"][0]["core:datetime"] = "2024-09-11T00:12:00Z"
        findings = self.run_doc(doc)
        self.assert_rule(findings, "X11", level=V.WARN, needle="UTC midnight")
        self.assertEqual([], [f for f in findings if f.level == V.ERROR])

    def test_X12_size_identity_violation(self):
        doc = golden()
        doc["global"]["rfuav:duration_s"] = 0.128  # implies 1024 bytes
        findings = self.run_doc(doc)
        self.assert_rule(findings, "X12", needle="size identity violated")

    def test_X13_num_samples_incoherent_with_size(self):
        doc = golden()
        doc["global"]["rfuav:num_samples"] = 65
        findings = self.run_doc(doc)
        self.assert_rule(findings, "X13", needle="payload bytes")

    def test_X14_annotations_must_be_sorted(self):
        doc = golden()
        doc["annotations"] = [
            {"core:sample_start": 32, "core:sample_count": 8},
            {"core:sample_start": 0, "core:sample_count": 8},
        ]
        self.assert_rule(self.run_doc(doc), "X14", needle="MUST be sorted")

    def test_X14_annotation_beyond_end_of_recording(self):
        doc = golden()
        doc["annotations"] = [{"core:sample_start": 60, "core:sample_count": 32}]
        self.assert_rule(self.run_doc(doc), "X14", needle="only has 64 samples")

    def test_X14_freq_edges_must_be_paired(self):
        doc = golden()
        del doc["annotations"][0]["core:freq_upper_edge"]
        self.assert_rule(self.run_doc(doc), "X14", needle="MUST be given")

    def test_X14_freq_edges_must_be_ordered(self):
        doc = golden()
        doc["annotations"][0]["core:freq_lower_edge"] = 2450000000.0
        doc["annotations"][0]["core:freq_upper_edge"] = 2430000000.0
        self.assert_rule(self.run_doc(doc), "X14", needle="must be <")

    def test_X15_discontinuous_recording_needs_gap_boundaries(self):
        doc = golden()
        doc["global"]["rfuav:continuous"] = False
        self.assert_rule(self.run_doc(doc), "X15", needle="MUST list the")

    def test_X15_gap_boundaries_must_increase(self):
        doc = golden()
        doc["global"]["rfuav:continuous"] = False
        doc["global"]["rfuav:gap_boundaries"] = [16, 16]
        self.assert_rule(self.run_doc(doc), "X15", needle="strictly increasing")

    def test_X19_annotation_count_over_the_operational_cap_warns(self):
        # V02 test 3: add_annotation is O(n^2) and 1e5 rows already cost 497 s.
        # Dense per-frame labels belong in frames.parquet (spec S3).
        # _check_annotations is called directly: running the whole schema over
        # 100001 objects would only make the test slow, not more informative.
        doc = golden()
        doc["annotations"] = [{"core:sample_start": 0}] * (
            V.ANNOTATION_SOFT_CAP + 1)
        findings = []
        V._check_annotations(doc, doc["global"], findings)
        self.assert_rule(findings, "X19", level=V.WARN, needle="frames.parquet")

    def test_X19_realistic_burst_count_does_not_warn(self):
        doc = golden()
        doc["annotations"] = [
            {"core:sample_start": i, "core:sample_count": 1}
            for i in range(NUM_SAMPLES)]
        findings = []
        V._check_annotations(doc, doc["global"], findings)
        self.assert_clean(findings)

    def test_X20_nc_source_cannot_be_redistributable(self):
        doc = golden()
        doc["global"]["rfuav:source_dataset"] = "dronerfb-dir"
        doc["global"]["rfuav:original_license"] = "CC-BY-NC-SA-4.0"
        doc["global"]["rfuav:redistributable"] = True
        self.assert_rule(self.run_doc(doc), "X20", needle="recipes/")

    def test_X20_unknown_spdx_is_a_warning(self):
        doc = golden()
        doc["global"]["rfuav:original_license"] = "CC-BY-4"
        findings = self.run_doc(doc)
        self.assert_rule(findings, "X20", level=V.WARN, needle="allowlist")
        self.assertEqual([], [f for f in findings if f.level == V.ERROR])

    def test_X21_non_default_protocol_mode_on_rfuav_warns(self):
        doc = golden()
        doc["global"]["rfuav:protocol_mode_at_capture"] = "non_default_documented"
        self.assert_rule(self.run_doc(doc), "X21", level=V.WARN, needle="F2")

    def test_X22_quarantined_record_is_flagged(self):
        doc = golden()
        doc["global"]["rfuav:quarantine_status"] = "duplicates"
        findings = self.run_doc(doc)
        self.assert_rule(findings, "X22", level=V.WARN, needle="train/test split")
        self.assertEqual([], [f for f in findings if f.level == V.ERROR])

    def test_X24_truncated_dataset_is_not_a_whole_number_of_samples(self):
        doc = golden()
        doc["global"]["rfuav:size_bytes"] = 510
        self.assert_rule(self.run_doc(doc), "X24", needle="whole number")


class TestRawTextRules(Base):
    """Rules that need the bytes on disk, not the parsed object."""

    def test_X18_duplicate_json_key_detected(self):
        raw = (b'{"global": {"core:datatype": "cf32_le", '
               b'"core:datatype": "ci16_le"}, "captures": [], '
               b'"annotations": []}')
        findings = []
        doc = V.load_meta_text(raw, findings)
        self.assertIsNotNone(doc)
        self.assert_rule(findings, "X18", needle="core:datatype")
        # json silently kept the LAST one, which is exactly the danger.
        self.assertEqual("ci16_le", doc["global"]["core:datatype"])

    def test_X23_bom_detected_but_still_parsed(self):
        raw = b"\xef\xbb\xbf" + json.dumps(golden()).encode("utf-8")
        findings = []
        doc = V.load_meta_text(raw, findings)
        self.assertIsNotNone(doc)
        self.assert_rule(findings, "X23", needle="BOM")

    def test_X23_invalid_json_reported_not_raised(self):
        findings = []
        self.assertIsNone(V.load_meta_text(b"{not json", findings))
        self.assert_rule(findings, "X23", needle="not valid JSON")

    def test_X23_non_object_top_level(self):
        findings = []
        self.assertIsNone(V.load_meta_text(b"[]", findings))
        self.assert_rule(findings, "X23", needle="must be a JSON object")


# ---------------------------------------------------------------------------
# filesystem behaviour
# ---------------------------------------------------------------------------

class TestFilesystem(Base):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rfuav_validate_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write_pair(self, doc=None, name=RECORDING_ID, payload=PAYLOAD,
                   write_data=True):
        doc = golden() if doc is None else doc
        meta = os.path.join(self.tmp, name + V.META_EXT)
        with open(meta, "wb") as handle:
            handle.write(json.dumps(doc, indent=2).encode("utf-8"))
        if write_data:
            with open(os.path.join(self.tmp, name + V.DATA_EXT), "wb") as handle:
                handle.write(payload)
        return meta

    def test_valid_pair_on_disk_is_clean(self):
        meta = self.write_pair()
        self.assert_clean(V.validate_meta_file(meta, SCHEMA))

    def test_check_hash_passes_on_matching_payload(self):
        meta = self.write_pair()
        self.assert_clean(V.validate_meta_file(meta, SCHEMA, check_hash=True))

    def test_X16_check_hash_catches_a_modified_dataset(self):
        meta = self.write_pair(payload=b"\xff" * SIZE_BYTES)
        findings = V.validate_meta_file(meta, SCHEMA, check_hash=True)
        self.assert_rule(findings, "X16", needle="hashes to")

    def test_X16_is_off_by_default(self):
        meta = self.write_pair(payload=b"\xff" * SIZE_BYTES)
        self.assert_clean(V.validate_meta_file(meta, SCHEMA))

    def test_X12_real_file_size_beats_declared_size_bytes(self):
        doc = golden()
        doc["global"]["rfuav:size_bytes"] = SIZE_BYTES  # a lie vs the real file
        meta = self.write_pair(doc=doc, payload=PAYLOAD + b"\x00" * 8)
        findings = V.validate_meta_file(meta, SCHEMA)
        self.assert_rule(findings, "X12", needle="the .sigmf-data file on disk")

    def test_X17_orphan_meta_detected_in_directory_mode(self):
        self.write_pair(write_data=False)
        results = V.validate_path(self.tmp, SCHEMA)
        flat = [f for findings in results.values() for f in findings]
        self.assert_rule(flat, "X17", needle="no %s" % (RECORDING_ID + V.DATA_EXT))

    def test_X17_orphan_data_detected_in_directory_mode(self):
        self.write_pair()
        with open(os.path.join(self.tmp, "stray.sigmf-data"), "wb") as handle:
            handle.write(b"\x00" * 8)
        results = V.validate_path(self.tmp, SCHEMA)
        stray = results[os.path.join(self.tmp, "stray.sigmf-data")]
        self.assert_rule(stray, "X17", needle="invisible to every SigMF tool")

    def test_X17_orphan_meta_detected_in_single_file_mode(self):
        meta = self.write_pair(write_data=False)
        results = V.validate_path(meta, SCHEMA)
        self.assert_rule(results[meta], "X17", needle="orphan")

    def test_X01_filename_mismatch_on_disk(self):
        meta = self.write_pair(name="pack1_0-1s")
        findings = V.validate_meta_file(meta, SCHEMA)
        self.assert_rule(findings, "X01", needle="pack1_0-1s.sigmf-meta")

    def test_core_dataset_redirects_the_data_file_lookup(self):
        doc = golden()
        doc["global"]["core:dataset"] = "elsewhere.bin"
        meta = self.write_pair(doc=doc, name=RECORDING_ID, write_data=False)
        with open(os.path.join(self.tmp, "elsewhere.bin"), "wb") as handle:
            handle.write(PAYLOAD)
        findings = V.validate_meta_file(meta, SCHEMA, check_hash=True)
        self.assert_clean(findings)

    def test_directory_walk_finds_nested_sidecars(self):
        nested = os.path.join(self.tmp, "bundled", "rfuav", "uav.dji.mini3")
        os.makedirs(nested)
        # Distinct recording_ids on purpose: reusing one id in two directories
        # is a real X25 violation, not a fixture detail.
        for target, cfg in ((self.tmp, "c00"), (nested, "c01")):
            doc = golden()
            recording_id = SESSION_ID + "-" + cfg
            doc["global"]["rfuav:recording_id"] = recording_id
            path = os.path.join(target, recording_id + V.META_EXT)
            with open(path, "wb") as handle:
                handle.write(json.dumps(doc).encode("utf-8"))
            with open(path[:-len(V.META_EXT)] + V.DATA_EXT, "wb") as handle:
                handle.write(PAYLOAD)
        results = V.validate_path(self.tmp, SCHEMA)
        self.assertEqual(2, len(results))
        for findings in results.values():
            self.assert_clean(findings)


class TestCli(Base):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rfuav_validate_cli_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.script = os.path.join(_TOOLDIR, "validate_sidecar.py")

    def write(self, doc, name=RECORDING_ID, data=True):
        meta = os.path.join(self.tmp, name + V.META_EXT)
        with open(meta, "wb") as handle:
            handle.write(json.dumps(doc, indent=2).encode("utf-8"))
        if data:
            with open(os.path.join(self.tmp, name + V.DATA_EXT), "wb") as handle:
                handle.write(PAYLOAD)
        return meta

    def run_cli(self, *args):
        proc = subprocess.Popen(
            [sys.executable, self.script] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        return proc.returncode, out.decode("utf-8"), err.decode("utf-8")

    def test_cli_exit_zero_on_valid_tree(self):
        self.write(golden())
        code, out, err = self.run_cli(self.tmp)
        self.assertEqual(0, code, out + err)
        self.assertIn("OK", out)
        self.assertIn("0 error(s)", out)

    def test_cli_exit_one_on_broken_sidecar(self):
        doc = golden()
        doc["global"]["core:sample_rte"] = 1
        self.write(doc)
        code, out, err = self.run_cli(self.tmp)
        self.assertEqual(1, code, out + err)
        self.assertIn("FAIL", out)
        self.assertIn("typo-guard", out)

    def test_cli_warnings_do_not_fail_unless_asked(self):
        doc = golden()
        doc["global"]["rfuav:quarantine_status"] = "ceshi"
        self.write(doc)
        code, out, _ = self.run_cli(self.tmp)
        self.assertEqual(0, code, out)
        self.assertIn("1 warning(s)", out)
        code, out, _ = self.run_cli(self.tmp, "--warn-as-error")
        self.assertEqual(1, code, out)

    def test_cli_json_report_is_machine_readable(self):
        doc = golden()
        doc["global"]["rfuav:signal_state"] = "idle"  # SPEC-GAP-01, see report
        self.write(doc)
        code, out, err = self.run_cli(self.tmp, "--json")
        self.assertEqual(1, code, out + err)
        payload = json.loads(out)
        self.assertEqual(1, payload["summary"]["files"])
        self.assertEqual(1, payload["summary"]["failed"])
        rules = [f["rule"] for f in payload["files"][0]["findings"]]
        self.assertIn("SCHEMA", rules)

    def test_cli_metadata_only_suppresses_orphan_error(self):
        doc = golden()
        doc["global"]["core:metadata_only"] = True
        self.write(doc, data=False)
        code, out, err = self.run_cli(self.tmp)
        self.assertEqual(0, code, out + err)

    def test_cli_missing_target_exits_two(self):
        code, _, err = self.run_cli(os.path.join(self.tmp, "nope"))
        self.assertEqual(2, code)
        self.assertIn("no such file", err)

    def test_cli_no_orphan_check_flag(self):
        self.write(golden(), data=False)
        code, out, _ = self.run_cli(self.tmp)
        self.assertEqual(1, code, out)
        code, out, _ = self.run_cli(self.tmp, "--no-orphan-check")
        self.assertEqual(0, code, out)


class TestSchemaEvaluator(Base):
    """Direct unit tests for the hand-rolled JSON Schema subset."""

    def check(self, instance, schema):
        return V.validate_schema(instance, schema, schema, "")

    def test_type_integer_rejects_bool(self):
        self.assertTrue(self.check(True, {"type": "integer"}))
        self.assertFalse(self.check(3, {"type": "integer"}))

    def test_type_number_accepts_int_and_float(self):
        self.assertFalse(self.check(3, {"type": "number"}))
        self.assertFalse(self.check(3.5, {"type": "number"}))
        self.assertTrue(self.check("3", {"type": "number"}))

    def test_nullable_union_type(self):
        schema = {"type": ["number", "null"]}
        self.assertFalse(self.check(None, schema))
        self.assertFalse(self.check(1.5, schema))
        self.assertTrue(self.check("x", schema))

    def test_pattern_properties_beat_additional_properties(self):
        schema = {
            "type": "object",
            "patternProperties": {"^x:": {"type": "string"}},
            "additionalProperties": False,
        }
        self.assertFalse(self.check({"x:a": "ok"}, schema))
        self.assertTrue(self.check({"x:a": 1}, schema))
        self.assertTrue(self.check({"y:a": "ok"}, schema))

    def test_negative_lookahead_pattern_compiles_and_matches(self):
        schema = {
            "type": "object",
            "properties": {"core:known": {"type": "string"}},
            "patternProperties": {
                "^(?!core:|rfuav:)[A-Za-z][A-Za-z0-9_-]*:[A-Za-z][A-Za-z0-9_]*$":
                    {"type": "string"},
            },
            "additionalProperties": False,
        }
        self.assertFalse(self.check({"signal:emitter": "a"}, schema))
        self.assertTrue(self.check({"core:unknown": "a"}, schema))

    def test_uniqueitems(self):
        self.assertTrue(self.check([1, 1], {"uniqueItems": True}))
        self.assertFalse(self.check([1, 2], {"uniqueItems": True}))

    def test_if_then_else(self):
        schema = {
            "if": {"required": ["a"]},
            "then": {"required": ["b"]},
            "else": {"required": ["c"]},
        }
        self.assertTrue(self.check({"a": 1}, schema))
        self.assertFalse(self.check({"a": 1, "b": 2}, schema))
        self.assertFalse(self.check({"c": 3}, schema))

    def test_local_ref_resolution(self):
        schema = {"$defs": {"n": {"type": "integer"}},
                  "properties": {"v": {"$ref": "#/$defs/n"}}}
        self.assertFalse(self.check({"v": 1}, schema))
        self.assertTrue(self.check({"v": "1"}, schema))

    def test_error_path_points_at_the_offending_key(self):
        schema = {"properties": {"global": {"properties": {
            "core:x": {"type": "integer"}}}}}
        findings = self.check({"global": {"core:x": "no"}}, schema)
        self.assertEqual(["global/core:x"], [f.path for f in findings])

    def test_schema_file_itself_is_wellformed(self):
        self.assertEqual("object", SCHEMA["type"])
        self.assertEqual(["global", "captures", "annotations"],
                         SCHEMA["required"])
        self.assertIs(False, SCHEMA["additionalProperties"])


class TestDateHelpers(Base):

    def test_days_from_civil_matches_known_epoch(self):
        self.assertEqual(0, V._days_from_civil(1970, 1, 1))
        self.assertEqual(19976, V._days_from_civil(2024, 9, 10))

    def test_leap_day_accepted_and_rejected(self):
        self.assertTrue(V._is_valid_civil_date(2024, 2, 29))
        self.assertFalse(V._is_valid_civil_date(2023, 2, 29))
        self.assertFalse(V._is_valid_civil_date(1900, 2, 29))
        self.assertTrue(V._is_valid_civil_date(2000, 2, 29))

    def test_fractional_seconds_allowed(self):
        V._parse_utc_z("2024-09-10T13:04:05.123456789Z")

    def test_lowercase_z_rejected(self):
        self.assertRaises(ValueError, V._parse_utc_z, "2024-09-10T13:04:05z")


if __name__ == "__main__":
    unittest.main(verbosity=2)
