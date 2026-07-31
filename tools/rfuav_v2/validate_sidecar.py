#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RFUAV v2.0 second-layer CI validator for ``.sigmf-meta`` sidecars.

Why this exists
---------------
``sigmf-python`` 1.11.1 is the FIRST validation layer and it is not enough
(verificacion-empirica-02):

* ``core:*`` typos pass silently -- the global object is ``additionalProperties:
  true`` and the ``core`` namespace counts as "declared", so
  ``core:not_a_real_field`` validates OK.
* the ``core:datatype`` pattern is not anchored, so ``cf32_lefoo`` and bare
  ``cf32`` both pass.
* nothing in the ``rfuav`` namespace is validated at all (``tx_state:
  "garbage"`` passes).
* an undeclared extension namespace is only a ``warnings.warn`` and never
  reaches the CLI exit code.
* a ``.sigmf-meta`` with no ``.sigmf-data`` next to it exits 0.
* the CLI itself crashes on macOS (``os.sched_getaffinity``).

This module is the SECOND layer: a hand-rolled JSON-Schema subset evaluator
driven by ``spec/rfuav-compliance-schema.json`` plus the cross-field and
filesystem rules that JSON Schema structurally cannot express.

Hard constraint: STDLIB ONLY, python >= 3.9.  The same file runs unchanged on
the Windows data server (python 3.11.4) and on macOS system python (3.9.6).
No ``jsonschema``, no ``pandas``, no ``numpy``.

Usage
-----
    python3 tools/rfuav_v2/validate_sidecar.py <file-or-dir> [options]

Exit code 0 = no ERROR findings, 1 = at least one ERROR (or, with
``--warn-as-error``, at least one WARN).

Rule inventory
--------------
Schema-expressible rules are reported as ``[SCHEMA]``.  Everything else has an
explicit ``Xnn`` id:

    X01  meta filename must be {rfuav:recording_id}.sigmf-meta      (spec S2)
    X02  rfuav:recording_id == rfuav:session_id + "-cNN"            (spec S2)
    X03  session_id u/r/s segments agree with the scalar fields     (spec S2/D6)
    X04  session_id taxonomy_short is consistent with taxonomy_id   (spec S2)
    X05  rfuav:device_role == role segment of rfuav:taxonomy_id     (spec S2)
    X06  core:extensions declares {rfuav, 0.1.0, optional:true}     (V02 impl.1)
    X07  every namespace used in the document is declared           (V02 gap b)
    X08  captures[0].core:sample_start == 0, strictly increasing    (SigMF MUST)
    X09  captures[0] carries core:frequency and core:datetime       (spec S3)
    X10  core:datetime is a real calendar instant in UTC 'Z'        (spec S3)
    X11  captures[0] date agrees with the session_id d-segment      (spec S2/A)
    X12  size identity  size_bytes == sample_rate * 2 * 4 * dur     (spec S8)
    X13  num_samples / duration_s / size_bytes are mutually coherent
    X14  annotations sorted, in range, freq edges paired            (SigMF MUST)
    X15  gap_boundaries strictly increasing, non-empty iff !continuous
    X16  core:sha512 matches the real file           (opt-in --check-hash, D4)
    X17  every .sigmf-meta has a .sigmf-data and vice versa         (V02 gap a)
    X18  no duplicate JSON keys in the raw text (json keeps the last silently)
    X19  annotation count <= 1e5 operational cap             (WARN, V02 test 3)
    X20  licence coherence: NC / NOASSERTION cannot be redistributable   (D7)
    X21  source_dataset rfuav implies protocol_mode factory_default_unverified
    X22  quarantined records are flagged so they never reach a split (WARN)
    X23  file is UTF-8 without BOM and decodes to a JSON object
    X24  size_bytes is a whole number of cf32 samples (multiple of 8)
    X25  rfuav:recording_id is unique across the tree     (canonical ID, spec S2)
    X26  one session_id maps to exactly one taxonomy_id / device (spec S2, F1)
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

__all__ = [
    "Finding",
    "ERROR",
    "WARN",
    "load_schema",
    "validate_schema",
    "validate_document",
    "validate_meta_file",
    "validate_path",
    "main",
]

ERROR = "ERROR"
WARN = "WARN"

DEFAULT_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spec",
    "rfuav-compliance-schema.json",
)

META_EXT = ".sigmf-meta"
DATA_EXT = ".sigmf-data"

#: The extension declaration the profile demands, verbatim.
REQUIRED_EXTENSION = {"name": "rfuav", "version": "0.1.0", "optional": True}

#: Namespaces that never need a ``core:extensions`` entry.
IMPLICIT_NAMESPACES = frozenset(["core"])

KNOWN_SPDX = frozenset([
    "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC0-1.0",
    "Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "GPL-3.0-only",
    "GPL-2.0-only", "LGPL-3.0-only", "ODbL-1.0", "NOASSERTION", "NONE",
])

#: Licences that forbid physical redistribution inside ``bundled/`` (D7).
NON_REDISTRIBUTABLE_SPDX = frozenset(["NOASSERTION", "NONE"])

BYTES_PER_SAMPLE_CF32 = 8  # 2 components * 4 bytes, complex float32 interleaved

ANNOTATION_SOFT_CAP = 100000  # V02 test 3: meta stays <= ~20 MB, validate ~3 s

_SESSION_RE = re.compile(
    r"^(?P<tax>[a-z0-9]+(?:[._-][a-z0-9]+)*)"
    r"-u(?P<unit>\d{2})-r(?P<rx>\d{2})-s(?P<site>\d{2})"
    r"-d(?P<date>\d{8})-(?P<seq>\d{2})$"
)
_RECORDING_RE = re.compile(r"^(?P<session>.+)-c(?P<cfg>\d{2})$")
_NAMESPACED_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):")
_DATETIME_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T"
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})(?:\.(?P<frac>\d{1,9}))?Z$"
)


class Finding(object):
    """One validation result.  ``level`` is :data:`ERROR` or :data:`WARN`."""

    __slots__ = ("level", "rule", "path", "message")

    def __init__(self, level, rule, path, message):
        self.level = level
        self.rule = rule
        self.path = path
        self.message = message

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Finding(%s, %s, %s, %s)" % (
            self.level, self.rule, self.path, self.message)

    def format(self):
        where = self.path or "<document>"
        return "%-5s [%s] %s: %s" % (self.level, self.rule, where, self.message)


def _as_int(value):
    """Return ``value`` as a python int if it is a JSON *integer*, else None.

    JSON Schema (and :func:`_json_type_of`) treat ``1.0`` as an integer, and
    ``json.load`` turns the literal ``512.0`` into a python ``float``.  Every
    cross-field rule below used to guard with a bare ``isinstance(x, int)``,
    so writing ``"rfuav:size_bytes": 512.0`` passed the schema layer and then
    SILENTLY disabled X08/X12/X13/X14/X15/X24 -- a wrong size identity with
    zero findings.  Funnel every integer read through this helper instead.

    ``bool`` is rejected on purpose: python ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_number(value):
    """Return ``value`` as a float if it is a JSON number, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _err(rule, path, message):
    return Finding(ERROR, rule, path, message)


def _warn(rule, path, message):
    return Finding(WARN, rule, path, message)


# --------------------------------------------------------------------------
# Minimal JSON Schema (draft 2020-12 subset) evaluator
# --------------------------------------------------------------------------
# Supported: $ref (local), type, const, enum, required, properties,
# patternProperties, additionalProperties, propertyNames, items, prefixItems,
# minItems, maxItems, uniqueItems, minLength, maxLength, pattern, minimum,
# maximum, exclusiveMinimum, exclusiveMaximum, multipleOf, format (uri/uuid),
# allOf, anyOf, oneOf, not, if/then/else.
# Deliberately unsupported (unused by our schema): remote $ref, $dynamicRef,
# dependentSchemas, contains, unevaluated*.


def _json_type_of(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "integer" if value.is_integer() else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _matches_type(value, wanted):
    actual = _json_type_of(value)
    if wanted == "number":
        return actual in ("integer", "number")
    if wanted == "integer":
        return actual == "integer"
    return actual == wanted


_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _check_format(value, fmt):
    """Return an error string or None.  Only the formats we actually use."""
    if not isinstance(value, str):
        return None
    if fmt == "uri":
        return None if _URI_RE.match(value) else "is not an absolute URI"
    if fmt == "uuid":
        return None if _UUID_RE.match(value) else "is not a UUID"
    return None


def _ecma_anchor(pattern):
    """Rewrite unescaped ``$`` to ``\\Z`` so ``$`` means ECMA-262 ``$``.

    JSON Schema regexes are ECMA-262, where ``$`` (no ``m`` flag) matches ONLY
    at the very end of the input.  Python's ``$`` also matches just before a
    single trailing newline, so ``re.search("^[0-9a-f]{128}$", "a"*128 + "\\n")``
    succeeds.  That let a ``core:sha512`` (or session_id, or capture_date, ...)
    carrying a trailing newline pass every anchored pattern in the profile --
    a value that is then NOT byte-comparable across manifests (D4).
    """
    out = []
    index = 0
    in_class = False
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "\\" and index + 1 < length:
            out.append(pattern[index:index + 2])
            index += 2
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        if char == "$" and not in_class:
            out.append("\\Z")
        else:
            out.append(char)
        index += 1
    return "".join(out)


_PATTERN_CACHE = {}


def _compile_pattern(pattern):
    compiled = _PATTERN_CACHE.get(pattern)
    if compiled is None:
        compiled = re.compile(_ecma_anchor(pattern))
        _PATTERN_CACHE[pattern] = compiled
    return compiled


def _join(path, part):
    return part if not path else "%s/%s" % (path, part)


def _resolve_ref(root, ref):
    if not ref.startswith("#/"):
        raise ValueError("only local $ref is supported, got %r" % (ref,))
    node = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        node = node[token]
    return node


def validate_schema(instance, schema, root=None, path=""):
    """Validate ``instance`` against ``schema``; return a list of Findings."""
    if root is None:
        root = schema
    out = []
    if schema is True or schema == {}:
        return out
    if schema is False:
        return [_err("SCHEMA", path, "no value is allowed here")]
    if not isinstance(schema, dict):
        raise TypeError("schema node must be a dict/bool, got %r" % (schema,))

    if "$ref" in schema:
        out.extend(validate_schema(
            instance, _resolve_ref(root, schema["$ref"]), root, path))

    if "type" in schema:
        wanted = schema["type"]
        wanted_list = wanted if isinstance(wanted, list) else [wanted]
        if not any(_matches_type(instance, w) for w in wanted_list):
            out.append(_err("SCHEMA", path, "expected type %s, got %s (%s)" % (
                "/".join(wanted_list), _json_type_of(instance),
                _truncate(instance))))
            return out  # further keywords would be noise

    if "const" in schema and instance != schema["const"]:
        out.append(_err("SCHEMA", path, "must be exactly %r, got %s" % (
            schema["const"], _truncate(instance))))
    if "enum" in schema and instance not in schema["enum"]:
        out.append(_err("SCHEMA", path, "must be one of %s, got %s" % (
            schema["enum"], _truncate(instance))))

    if isinstance(instance, str):
        out.extend(_validate_string(instance, schema, path))
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        out.extend(_validate_number(instance, schema, path))
    if isinstance(instance, list):
        out.extend(_validate_array(instance, schema, root, path))
    if isinstance(instance, dict):
        out.extend(_validate_object(instance, schema, root, path))

    for key, combiner in (("allOf", "all"), ("anyOf", "any"), ("oneOf", "one")):
        if key not in schema:
            continue
        results = [validate_schema(instance, sub, root, path)
                   for sub in schema[key]]
        passed = [i for i, r in enumerate(results) if not r]
        if combiner == "all":
            for r in results:
                out.extend(r)
        elif combiner == "any" and not passed:
            out.append(_err("SCHEMA", path, "matches none of the anyOf branches"))
        elif combiner == "one" and len(passed) != 1:
            out.append(_err("SCHEMA", path,
                            "must match exactly one oneOf branch, matched %d"
                            % len(passed)))

    if "not" in schema and not validate_schema(instance, schema["not"], root, path):
        out.append(_err("SCHEMA", path, "must NOT match the 'not' subschema"))

    if "if" in schema:
        cond_ok = not validate_schema(instance, schema["if"], root, path)
        branch = schema.get("then") if cond_ok else schema.get("else")
        if branch is not None:
            out.extend(validate_schema(instance, branch, root, path))

    return out


def _truncate(value, limit=80):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _validate_string(instance, schema, path):
    out = []
    if "minLength" in schema and len(instance) < schema["minLength"]:
        out.append(_err("SCHEMA", path, "shorter than minLength=%d" %
                        schema["minLength"]))
    if "maxLength" in schema and len(instance) > schema["maxLength"]:
        out.append(_err("SCHEMA", path, "longer than maxLength=%d" %
                        schema["maxLength"]))
    if ("pattern" in schema
            and _compile_pattern(schema["pattern"]).search(instance) is None):
        out.append(_err("SCHEMA", path, "%s does not match pattern %s" % (
            _truncate(instance), schema["pattern"])))
    if "format" in schema:
        problem = _check_format(instance, schema["format"])
        if problem:
            out.append(_err("SCHEMA", path, "%s %s" % (
                _truncate(instance), problem)))
    return out


def _validate_number(instance, schema, path):
    out = []
    if "minimum" in schema and instance < schema["minimum"]:
        out.append(_err("SCHEMA", path, "%r < minimum %r" % (
            instance, schema["minimum"])))
    if "maximum" in schema and instance > schema["maximum"]:
        out.append(_err("SCHEMA", path, "%r > maximum %r" % (
            instance, schema["maximum"])))
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        out.append(_err("SCHEMA", path, "%r <= exclusiveMinimum %r" % (
            instance, schema["exclusiveMinimum"])))
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        out.append(_err("SCHEMA", path, "%r >= exclusiveMaximum %r" % (
            instance, schema["exclusiveMaximum"])))
    if "multipleOf" in schema:
        step = schema["multipleOf"]
        if step and (instance / float(step)) != int(instance / float(step)):
            out.append(_err("SCHEMA", path, "%r is not a multiple of %r" % (
                instance, step)))
    return out


def _validate_array(instance, schema, root, path):
    out = []
    if "minItems" in schema and len(instance) < schema["minItems"]:
        out.append(_err("SCHEMA", path, "has %d items, minItems=%d" % (
            len(instance), schema["minItems"])))
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        out.append(_err("SCHEMA", path, "has %d items, maxItems=%d" % (
            len(instance), schema["maxItems"])))
    if schema.get("uniqueItems"):
        seen = []
        for item in instance:
            if item in seen:
                out.append(_err("SCHEMA", path,
                                "duplicate item %s (uniqueItems)" %
                                _truncate(item)))
                break
            seen.append(item)
    offset = 0
    if "prefixItems" in schema:
        for index, sub in enumerate(schema["prefixItems"]):
            if index < len(instance):
                out.extend(validate_schema(
                    instance[index], sub, root, "%s[%d]" % (path, index)))
        offset = len(schema["prefixItems"])
    if "items" in schema:
        for index in range(offset, len(instance)):
            out.extend(validate_schema(
                instance[index], schema["items"], root,
                "%s[%d]" % (path, index)))
    return out


def _validate_object(instance, schema, root, path):
    out = []
    for name in schema.get("required", []):
        if name not in instance:
            out.append(_err("SCHEMA", path, "missing required key %r" % name))

    props = schema.get("properties", {})
    pattern_props = schema.get("patternProperties", {})
    additional = schema.get("additionalProperties", True)
    property_names = schema.get("propertyNames")

    compiled = [(_compile_pattern(p), sub) for p, sub in pattern_props.items()]

    for key in instance:
        if property_names is not None:
            out.extend(validate_schema(
                key, property_names, root, _join(path, key)))
        handled = False
        if key in props:
            out.extend(validate_schema(
                instance[key], props[key], root, _join(path, key)))
            handled = True
        for rx, sub in compiled:
            if rx.search(key):
                out.extend(validate_schema(
                    instance[key], sub, root, _join(path, key)))
                handled = True
        if handled:
            continue
        if additional is False:
            out.append(_err("SCHEMA", _join(path, key),
                            "unknown key %r is not allowed here "
                            "(typo-guard; sigmf-python accepts these silently)"
                            % key))
        elif isinstance(additional, dict):
            out.extend(validate_schema(
                instance[key], additional, root, _join(path, key)))
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_schema(path=None):
    with open(path or DEFAULT_SCHEMA_PATH, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _pairs_hook_factory(duplicates):
    def hook(pairs):
        mapping = OrderedDict()
        for key, value in pairs:
            if key in mapping:
                duplicates.append(key)
            mapping[key] = value
        return mapping
    return hook


def load_meta_text(raw_bytes, findings):
    """Decode + parse, appending X18/X23 findings.  Returns doc or None."""
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        findings.append(_err(
            "X23", "", "file starts with a UTF-8 BOM; SigMF metadata MUST be "
            "plain UTF-8 (a BOM breaks strict JSON parsers)"))
        raw_bytes = raw_bytes[3:]
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        findings.append(_err("X23", "", "not valid UTF-8: %s" % exc))
        return None
    duplicates = []
    try:
        doc = json.loads(text, object_pairs_hook=_pairs_hook_factory(duplicates))
    except ValueError as exc:
        findings.append(_err("X23", "", "not valid JSON: %s" % exc))
        return None
    for key in sorted(set(duplicates)):
        findings.append(_err(
            "X18", "", "duplicate JSON key %r: json.load() silently keeps the "
            "LAST occurrence, so the file does not mean what it looks like"
            % key))
    if not isinstance(doc, dict):
        findings.append(_err("X23", "", "top level must be a JSON object, got %s"
                             % _json_type_of(doc)))
        return None
    return doc


# --------------------------------------------------------------------------
# Cross-field / semantic rules that JSON Schema cannot express
# --------------------------------------------------------------------------

def _days_from_civil(year, month, day):
    """Howard Hinnant's civil-from-days inverse; no datetime import needed."""
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    yoe = year - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _is_valid_civil_date(year, month, day):
    if not 1 <= month <= 12 or day < 1:
        return False
    leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    lengths = [31, 29 if leap else 28, 31, 30, 31, 30,
               31, 31, 30, 31, 30, 31]
    return day <= lengths[month - 1]


def _parse_utc_z(text):
    """Return (days_since_epoch, seconds_of_day) or raise ValueError."""
    match = _DATETIME_RE.match(text)
    if not match:
        raise ValueError(
            "must be strict ISO-8601 UTC 'YYYY-MM-DDThh:mm:ss[.fff]Z' with a "
            "literal trailing Z (naive local times and +hh:mm offsets are "
            "rejected)")
    year = int(match.group("y"))
    month = int(match.group("mo"))
    day = int(match.group("d"))
    hour = int(match.group("h"))
    minute = int(match.group("mi"))
    second = int(match.group("s"))
    if not _is_valid_civil_date(year, month, day):
        raise ValueError("%04d-%02d-%02d is not a real calendar date"
                         % (year, month, day))
    if hour > 23 or minute > 59 or second > 60:
        raise ValueError("time %02d:%02d:%02d is out of range"
                         % (hour, minute, second))
    return _days_from_civil(year, month, day), hour * 3600 + minute * 60 + second


def _namespaces_used(node, acc):
    if isinstance(node, dict):
        for key, value in node.items():
            match = _NAMESPACED_KEY_RE.match(key)
            if match:
                acc.add(match.group(1))
            _namespaces_used(value, acc)
    elif isinstance(node, list):
        for item in node:
            _namespaces_used(item, acc)
    return acc


def _check_identifiers(glob, meta_name, out):
    recording_id = glob.get("rfuav:recording_id")
    session_id = glob.get("rfuav:session_id")
    taxonomy_id = glob.get("rfuav:taxonomy_id")

    session_match = _SESSION_RE.match(session_id) if isinstance(session_id, str) else None

    # X02 recording_id == session_id + "-cNN"
    if isinstance(recording_id, str) and isinstance(session_id, str):
        rec_match = _RECORDING_RE.match(recording_id)
        if rec_match is None or rec_match.group("session") != session_id:
            out.append(_err(
                "X02", "global/rfuav:recording_id",
                "%r must be exactly rfuav:session_id + '-cNN' (session_id=%r)"
                % (recording_id, session_id)))

    # X01 filename == {recording_id}.sigmf-meta
    if meta_name is not None and isinstance(recording_id, str):
        expected = recording_id + META_EXT
        if meta_name != expected:
            out.append(_err(
                "X01", "global/rfuav:recording_id",
                "file is named %r but spec S2 requires %r" % (meta_name, expected)))

    if session_match is None:
        return

    # X03 session_id segments vs the scalar fields
    for field, group, prefix in (("rfuav:device_unit_id", "unit", "u"),
                                 ("rfuav:rx_id", "rx", "r"),
                                 ("rfuav:site_id", "site", "s")):
        value = glob.get(field)
        if isinstance(value, str):
            expected = prefix + session_match.group(group)
            if value != expected:
                out.append(_err(
                    "X03", "global/" + field,
                    "%r contradicts session_id which says %r" % (value, expected)))

    # X04 taxonomy_short vs taxonomy_id
    if isinstance(taxonomy_id, str) and "." in taxonomy_id:
        segments = taxonomy_id.split(".")
        short_tokens = [t for t in re.split(r"[._-]", session_match.group("tax")) if t]
        unknown = [t for t in short_tokens if t not in segments]
        model = segments[2] if len(segments) > 2 else None
        if unknown:
            out.append(_err(
                "X04", "global/rfuav:session_id",
                "taxonomy_short %r contains token(s) %s that are not segments "
                "of rfuav:taxonomy_id %r (spec S2 never pins the exact "
                "taxonomy_short join, so this check only demands that every "
                "token be a taxonomy segment)"
                % (session_match.group("tax"), unknown, taxonomy_id)))
        elif model is not None and model not in short_tokens:
            out.append(_err(
                "X04", "global/rfuav:session_id",
                "taxonomy_short %r does not contain the model segment %r of "
                "rfuav:taxonomy_id %r"
                % (session_match.group("tax"), model, taxonomy_id)))

    # X05 device_role vs taxonomy role
    role = glob.get("rfuav:device_role")
    if isinstance(taxonomy_id, str) and isinstance(role, str):
        tax_role = taxonomy_id.split(".")[0]
        if tax_role != role:
            out.append(_err(
                "X05", "global/rfuav:device_role",
                "%r contradicts the role segment of rfuav:taxonomy_id (%r)"
                % (role, tax_role)))


def _check_extensions(doc, glob, out):
    extensions = glob.get("core:extensions")
    if not isinstance(extensions, list):
        return
    names = []
    found_rfuav = None
    for item in extensions:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        names.append(name)
        if name == "rfuav":
            found_rfuav = item
    # ``name`` is only schema-checked to be a string INSIDE each entry; a
    # malformed list can mix ``None`` and ``str``, and ``sorted()`` on that
    # mixture raises TypeError and takes the whole CLI down with a traceback.
    # Sort by repr so the report stays deterministic without ever comparing
    # across types.
    duplicates = sorted(set(n for n in names if names.count(n) > 1), key=repr)
    for name in duplicates:
        out.append(_err("X06", "global/core:extensions",
                        "namespace %r declared more than once" % name))
    if found_rfuav is None:
        out.append(_err(
            "X06", "global/core:extensions",
            "missing the mandatory entry %r; sigmf-python only warns today but "
            "upstream announces a future hard ValidationError" % (REQUIRED_EXTENSION,)))
    else:
        for key, want in REQUIRED_EXTENSION.items():
            got = found_rfuav.get(key)
            if got != want:
                out.append(_err(
                    "X06", "global/core:extensions",
                    "rfuav extension entry has %s=%r, the profile requires %r"
                    % (key, got, want)))

    declared = set(n for n in names if isinstance(n, str)) | set(IMPLICIT_NAMESPACES)
    used = _namespaces_used(doc, set())
    for namespace in sorted(used - declared):
        out.append(_err(
            "X07", "global/core:extensions",
            "namespace %r is used in the document but not declared in "
            "core:extensions (sigmf-python only emits a python warning that "
            "never reaches the CLI exit code)" % namespace))


def _check_captures(doc, glob, out):
    captures = doc.get("captures")
    if not isinstance(captures, list) or not captures:
        return
    first = captures[0]
    if isinstance(first, dict):
        if first.get("core:sample_start") != 0:
            out.append(_err(
                "X08", "captures[0]/core:sample_start",
                "the first capture segment MUST start at sample 0, got %r"
                % first.get("core:sample_start")))
        for field in ("core:frequency", "core:datetime"):
            if field not in first:
                out.append(_err("X09", "captures[0]",
                                "the RFUAV profile requires %s on captures[0]"
                                % field))
    previous = None
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            continue
        start = _as_int(capture.get("core:sample_start"))
        if start is not None:
            if previous is not None and start <= previous:
                out.append(_err(
                    "X08", "captures[%d]/core:sample_start" % index,
                    "captures MUST be sorted by sample_start and be strictly "
                    "increasing (%r follows %r)" % (start, previous)))
            previous = start

    stamps = []
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            continue
        text = capture.get("core:datetime")
        if text is None:
            continue
        if not isinstance(text, str):
            continue
        try:
            stamps.append((index, _parse_utc_z(text)))
        except ValueError as exc:
            out.append(_err("X10", "captures[%d]/core:datetime" % index,
                            "%r %s" % (text, exc)))
    for i in range(1, len(stamps)):
        if stamps[i][1] < stamps[i - 1][1]:
            out.append(_err(
                "X10", "captures[%d]/core:datetime" % stamps[i][0],
                "capture timestamps go backwards relative to captures[%d]"
                % stamps[i - 1][0]))

    # X11 capture date vs the session_id d-segment
    session_id = glob.get("rfuav:session_id")
    match = _SESSION_RE.match(session_id) if isinstance(session_id, str) else None
    if match is None or not stamps or stamps[0][0] != 0:
        return
    ymd = match.group("date")
    try:
        session_days = _days_from_civil(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
    except Exception:
        return
    if not _is_valid_civil_date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])):
        out.append(_err("X11", "global/rfuav:session_id",
                        "d-segment %r is not a real calendar date" % ymd))
        return
    delta = stamps[0][1][0] - session_days
    if delta == 0:
        return
    if abs(delta) == 1:
        out.append(_warn(
            "X11", "captures[0]/core:datetime",
            "capture date differs by one day from the session_id d-segment "
            "(%s); acceptable only if the session crossed UTC midnight" % ymd))
    else:
        out.append(_err(
            "X11", "captures[0]/core:datetime",
            "capture date is %d days away from the session_id d-segment (%s); "
            "spec S2 takes the d-segment from the real capture calendar and "
            "audit A warns that the 11-26 mtimes are SLICING dates, not "
            "capture dates" % (delta, ymd)))


def _resolve_size_bytes(glob, data_path):
    """Return (size_bytes, source) or (None, None)."""
    if data_path is not None and os.path.isfile(data_path):
        return os.path.getsize(data_path), "the .sigmf-data file on disk"
    declared = _as_int(glob.get("rfuav:size_bytes"))
    if declared is not None:
        return declared, "rfuav:size_bytes"
    return None, None


def _check_size_identity(glob, data_path, out):
    sample_rate = _as_number(glob.get("core:sample_rate"))
    duration = _as_number(glob.get("rfuav:duration_s"))
    num_samples = _as_int(glob.get("rfuav:num_samples"))
    trailing = _as_int(glob.get("core:trailing_bytes"))
    if trailing is None:
        trailing = 0
    channels = _as_int(glob.get("core:num_channels"))
    if channels is None or channels < 1:
        channels = 1

    size_bytes, source = _resolve_size_bytes(glob, data_path)

    # X24 whole number of cf32 samples
    if size_bytes is not None:
        payload = size_bytes - trailing
        if payload <= 0:
            out.append(_err("X24", "global/core:trailing_bytes",
                            "trailing_bytes=%d leaves no payload in a %d byte "
                            "dataset" % (trailing, size_bytes)))
            return
        if payload % (BYTES_PER_SAMPLE_CF32 * channels):
            out.append(_err(
                "X24", "global/core:sha512",
                "dataset payload is %d bytes (%s), not a whole number of "
                "cf32_le samples (%d bytes each) - the file is truncated or "
                "the datatype is wrong"
                % (payload, source, BYTES_PER_SAMPLE_CF32 * channels)))

    # X13 internal coherence of num_samples / duration_s / sample_rate
    if (num_samples is not None and duration is not None
            and sample_rate is not None and sample_rate > 0):
        implied = num_samples / float(sample_rate)
        if abs(implied - duration) > max(1e-9, 0.5 / float(sample_rate)):
            out.append(_err(
                "X13", "global/rfuav:duration_s",
                "duration_s=%r but num_samples/sample_rate = %.12g s" % (
                    duration, implied)))

    if size_bytes is None:
        return
    payload = size_bytes - trailing

    if num_samples is not None:
        expected = num_samples * BYTES_PER_SAMPLE_CF32 * channels
        if expected != payload:
            out.append(_err(
                "X13", "global/rfuav:num_samples",
                "num_samples=%d implies %d payload bytes but %s reports %d"
                % (num_samples, expected, source, payload)))

    # X12 the spec S8 size identity proper
    if sample_rate is not None and duration is not None:
        exact = sample_rate * 2 * 4 * duration * channels
        expected = int(round(exact))
        if abs(exact - expected) > 1e-6:
            out.append(_warn(
                "X12", "global/rfuav:duration_s",
                "sample_rate * 2 * 4 * duration_s = %.6f is not an integral "
                "number of bytes" % exact))
        if expected != payload:
            out.append(_err(
                "X12", "global/rfuav:duration_s",
                "size identity violated (spec S8): sample_rate(%r) * 2 * 4 * "
                "duration_s(%r) = %d bytes, but %s reports %d payload bytes "
                "(delta %+d)"
                % (sample_rate, duration, expected, source, payload,
                   payload - expected)))


def _check_annotations(doc, glob, out):
    annotations = doc.get("annotations")
    if not isinstance(annotations, list):
        return
    if len(annotations) > ANNOTATION_SOFT_CAP:
        out.append(_warn(
            "X19", "annotations",
            "%d annotations exceeds the operational cap of %d; per-frame dense "
            "labels belong in frames.parquet, not in SigMF (spec S3)"
            % (len(annotations), ANNOTATION_SOFT_CAP)))
    num_samples = _as_int(glob.get("rfuav:num_samples"))
    previous = None
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            continue
        where = "annotations[%d]" % index
        start = _as_int(annotation.get("core:sample_start"))
        count = _as_int(annotation.get("core:sample_count"))
        if start is not None:
            if previous is not None and start < previous:
                out.append(_err(
                    "X14", where + "/core:sample_start",
                    "annotations MUST be sorted by sample_start (%r follows %r)"
                    % (start, previous)))
            previous = start
            if num_samples is not None:
                end = start + (count if count is not None else 0)
                if start >= num_samples or end > num_samples:
                    out.append(_err(
                        "X14", where,
                        "annotation spans samples [%d, %d) but the recording "
                        "only has %d samples" % (start, end, num_samples)))
        lower = annotation.get("core:freq_lower_edge")
        upper = annotation.get("core:freq_upper_edge")
        if (lower is None) != (upper is None):
            out.append(_err(
                "X14", where,
                "core:freq_lower_edge and core:freq_upper_edge MUST be given "
                "together (SigMF requirement)"))
        elif (_as_number(lower) is not None and _as_number(upper) is not None
              and _as_number(lower) >= _as_number(upper)):
            out.append(_err("X14", where,
                            "freq_lower_edge %r must be < freq_upper_edge %r"
                            % (lower, upper)))


def _check_gaps(glob, out):
    continuous = glob.get("rfuav:continuous")
    gaps = glob.get("rfuav:gap_boundaries")
    if continuous is False and not gaps:
        out.append(_err(
            "X15", "global/rfuav:gap_boundaries",
            "rfuav:continuous is false so gap_boundaries MUST list the "
            "discontinuities (-lostN files and missing slices, spec S5)"))
    if continuous is True and gaps:
        out.append(_err(
            "X15", "global/rfuav:gap_boundaries",
            "rfuav:continuous is true but %d gap boundaries are declared"
            % len(gaps)))
    if isinstance(gaps, list):
        num_samples = _as_int(glob.get("rfuav:num_samples"))
        previous = None
        for index, raw_value in enumerate(gaps):
            value = _as_int(raw_value)
            if value is None:
                continue
            if previous is not None and value <= previous:
                out.append(_err(
                    "X15", "global/rfuav:gap_boundaries[%d]" % index,
                    "gap boundaries MUST be strictly increasing (%r follows %r)"
                    % (value, previous)))
            previous = value
            if num_samples is not None and value >= num_samples:
                out.append(_err(
                    "X15", "global/rfuav:gap_boundaries[%d]" % index,
                    "gap boundary %d is beyond the recording length (%d samples)"
                    % (value, num_samples)))


def _check_policy(glob, out):
    spdx = glob.get("rfuav:original_license")
    redistributable = glob.get("rfuav:redistributable")
    if isinstance(spdx, str):
        if spdx not in KNOWN_SPDX:
            out.append(_warn(
                "X20", "global/rfuav:original_license",
                "%r is not in the known SPDX allowlist; confirm the identifier "
                "before it reaches ATTRIBUTION.md" % spdx))
        forbidden = spdx in NON_REDISTRIBUTABLE_SPDX or "-NC" in spdx.upper()
        if forbidden and redistributable is True:
            out.append(_err(
                "X20", "global/rfuav:redistributable",
                "original_license %r cannot be physically redistributed in "
                "bundled/ (D7: NC / unlicensed sources go to recipes/)" % spdx))

    if (glob.get("rfuav:source_dataset") == "rfuav"
            and glob.get("rfuav:protocol_mode_at_capture") not in
            (None, "factory_default_unverified")):
        out.append(_warn(
            "X21", "global/rfuav:protocol_mode_at_capture",
            "F2 records that every RFUAV radio was captured in factory default "
            "state with no traceability, so %r needs written evidence"
            % glob.get("rfuav:protocol_mode_at_capture")))

    status = glob.get("rfuav:quarantine_status")
    if isinstance(status, str) and status != "none":
        out.append(_warn(
            "X22", "global/rfuav:quarantine_status",
            "record is quarantined as %r; it MUST NOT appear in any train/test "
            "split and MUST have a _quarantine/LEDGER.csv row" % status))


def _check_hash(glob, data_path, out):
    declared = glob.get("core:sha512")
    if not isinstance(declared, str) or data_path is None:
        return
    if not os.path.isfile(data_path):
        return
    digest = hashlib.sha512()
    with open(data_path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != declared.lower():
        out.append(_err("X16", "global/core:sha512",
                        "declared %s... but the file hashes to %s..."
                        % (declared[:16], actual[:16])))


def validate_document(doc, schema, meta_name=None, data_path=None,
                      check_hash=False):
    """Validate an already-parsed metadata document.  Returns [Finding]."""
    out = validate_schema(doc, schema, schema, "")
    glob = doc.get("global")
    if not isinstance(glob, dict):
        return out
    _check_identifiers(glob, meta_name, out)
    _check_extensions(doc, glob, out)
    _check_captures(doc, glob, out)
    _check_size_identity(glob, data_path, out)
    _check_annotations(doc, glob, out)
    _check_gaps(glob, out)
    _check_policy(glob, out)
    if check_hash:
        _check_hash(glob, data_path, out)
    return out


def validate_meta_file(meta_path, schema, check_hash=False):
    findings = []
    try:
        with open(meta_path, "rb") as handle:
            raw = handle.read()
    except (IOError, OSError) as exc:
        return [_err("X23", "", "cannot read file: %s" % exc)]
    doc = load_meta_text(raw, findings)
    if doc is None:
        return findings
    data_path = meta_path[:-len(META_EXT)] + DATA_EXT
    glob = doc.get("global")
    if isinstance(glob, dict) and isinstance(glob.get("core:dataset"), str):
        data_path = os.path.join(os.path.dirname(meta_path), glob["core:dataset"])
    findings.extend(validate_document(
        doc,
        schema,
        meta_name=os.path.basename(meta_path),
        data_path=data_path,
        check_hash=check_hash,
    ))
    return findings


# --------------------------------------------------------------------------
# Directory walk + orphan pairing
# --------------------------------------------------------------------------

def collect_meta_files(root):
    if os.path.isfile(root):
        # A non-".sigmf-meta" target used to be accepted here, and every
        # downstream consumer then computed ``root[:-len(META_EXT)] + DATA_EXT``
        # on it: "/tree/foo.iq" became "/tre.sigmf-data", a path OUTSIDE the
        # tree that --check-hash would happily stat and hash.  Refuse instead.
        return ([root], []) if root.endswith(META_EXT) else ([], [])
    metas = []
    datas = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(META_EXT):
                metas.append(os.path.join(dirpath, name))
            elif name.endswith(DATA_EXT):
                datas.append(os.path.join(dirpath, name))
    return metas, datas


def _declared_dataset(meta_path):
    """Return the dataset path a sidecar points at, honouring ``core:dataset``.

    Spec open decision (b) explicitly allows keeping ``.iq`` payloads plus a
    ``core:dataset`` pointer.  :func:`validate_meta_file` already honoured it
    for the size identity, but X17 did not, so a perfectly paired
    ``rec.sigmf-meta`` + ``rec.iq`` was reported CLEAN by the document rules
    and ORPHAN by the tree rule in the same run.
    """
    try:
        with open(meta_path, "rb") as handle:
            doc = json.loads(handle.read().decode("utf-8", "replace"))
        declared = doc.get("global", {}).get("core:dataset")
    except Exception:
        declared = None
    if isinstance(declared, str) and declared:
        return os.path.normpath(
            os.path.join(os.path.dirname(meta_path), declared))
    return os.path.normpath(meta_path[:-len(META_EXT)] + DATA_EXT)


def orphan_findings(metas, datas):
    """X17: .sigmf-meta <-> .sigmf-data pairing (upstream never checks this)."""
    out = {}
    paired_data = set()
    for path in metas:
        target = _declared_dataset(path)
        if os.path.isfile(target):
            paired_data.add(target)
            continue
        out.setdefault(path, []).append(_err(
            "X17", "",
            "orphan: no %s next to this metadata file (sigmf_validate exits 0 "
            "for orphan metas even with checksums on)"
            % os.path.basename(target)))
    for path in sorted(datas):
        if os.path.normpath(path) in paired_data:
            continue
        stem = path[:-len(DATA_EXT)]
        out.setdefault(path, []).append(_err(
            "X17", "",
            "orphan: dataset file has no %s sidecar, so it is invisible to "
            "every SigMF tool" % (os.path.basename(stem) + META_EXT)))
    return out


def cross_file_findings(metas):
    """X25/X26: rules that only exist ACROSS files.

    X25  ``rfuav:recording_id`` is the canonical identity of a recording and
         is also the filename stem; two files claiming the same one make the
         manifest join ambiguous.
    X26  ``session_id`` is the Shulman leakage grouping key (spec S2/S4).  If
         two DIFFERENT devices share one session_id, GroupKFold puts them in
         the same fold and the split gate silently stops protecting anything.
         This is exactly the locked fact F1 hazard: SKYDROID H12 and YUNZHUO
         H12 are two physically distinct radios whose taxonomy_short is the
         same token ``h12``, so per-file X04 cannot catch the collision.
    """
    out = {}
    by_recording = OrderedDict()
    by_session = OrderedDict()
    for path in metas:
        try:
            with open(path, "rb") as handle:
                doc = json.loads(handle.read().decode("utf-8", "replace"))
            glob = doc.get("global") or {}
        except Exception:
            continue
        if not isinstance(glob, dict):
            continue
        recording_id = glob.get("rfuav:recording_id")
        session_id = glob.get("rfuav:session_id")
        if isinstance(recording_id, str):
            by_recording.setdefault(recording_id, []).append(path)
        if isinstance(session_id, str):
            by_session.setdefault(session_id, []).append(
                (path, glob.get("rfuav:taxonomy_id")))

    for recording_id, paths in by_recording.items():
        if len(paths) < 2:
            continue
        for path in paths:
            others = [os.path.basename(p) for p in paths if p != path]
            out.setdefault(path, []).append(_err(
                "X25", "global/rfuav:recording_id",
                "recording_id %r is also claimed by %s; the canonical ID must "
                "be unique across the tree (spec S2)"
                % (recording_id, ", ".join(sorted(others)))))

    for session_id, entries in by_session.items():
        taxonomies = sorted(set(
            t for _, t in entries if isinstance(t, str)))
        if len(taxonomies) < 2:
            continue
        for path, _taxonomy in entries:
            out.setdefault(path, []).append(_err(
                "X26", "global/rfuav:session_id",
                "session_id %r is shared by %d different taxonomy_id values "
                "(%s); session_id is the leakage grouping key, so two distinct "
                "devices under one key defeat GroupKFold (spec S2/S4, F1: "
                "SKYDROID H12 != YUNZHUO H12)"
                % (session_id, len(taxonomies), ", ".join(taxonomies))))
    return out


def validate_path(root, schema, check_hash=False, orphan_check=True):
    """Validate a file or a tree.  Returns an ordered list of (path, findings)."""
    metas, datas = collect_meta_files(root)
    results = OrderedDict()
    if os.path.isfile(root) and not metas:
        return OrderedDict([(root, [_err(
            "X17", "",
            "target is not a %s file; this validator only reads SigMF "
            "metadata (point it at the sidecar or at the directory)"
            % META_EXT)])])
    for path in metas:
        results[path] = validate_meta_file(path, schema, check_hash=check_hash)
    if orphan_check:
        # The single-file branch used to inline its own, DIFFERENT pairing
        # check that ignored core:dataset.  One implementation only.
        for path, findings in orphan_findings(metas, datas).items():
            results.setdefault(path, []).extend(findings)
    for path, findings in cross_file_findings(metas).items():
        results.setdefault(path, []).extend(findings)
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _metadata_only(path):
    try:
        with open(path, "rb") as handle:
            doc = json.loads(handle.read().decode("utf-8", "replace"))
        return bool(doc.get("global", {}).get("core:metadata_only"))
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="validate_sidecar.py",
        description="RFUAV v2.0 compliance validator for SigMF sidecars "
                    "(stdlib only, python >= 3.9).")
    parser.add_argument("target", help="a .sigmf-meta file or a directory tree")
    parser.add_argument("--schema", default=None,
                        help="path to rfuav-compliance-schema.json")
    parser.add_argument("--check-hash", action="store_true",
                        help="recompute core:sha512 from the dataset file "
                             "(~3.9 s/GB single-thread; off by default)")
    parser.add_argument("--no-orphan-check", action="store_true",
                        help="skip the .sigmf-meta <-> .sigmf-data pairing check")
    parser.add_argument("--warn-as-error", action="store_true",
                        help="exit 1 when there are warnings but no errors")
    parser.add_argument("--quiet", action="store_true",
                        help="only print files that have findings")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="machine readable report on stdout")
    args = parser.parse_args(argv)

    try:
        schema = load_schema(args.schema)
    except (IOError, OSError, ValueError) as exc:
        sys.stderr.write("cannot load schema: %s\n" % exc)
        return 2

    if not os.path.exists(args.target):
        sys.stderr.write("no such file or directory: %s\n" % args.target)
        return 2

    results = validate_path(args.target, schema,
                            check_hash=args.check_hash,
                            orphan_check=not args.no_orphan_check)

    # metadata_only recordings legitimately have no dataset file
    for path, findings in results.items():
        if path.endswith(META_EXT) and _metadata_only(path):
            results[path] = [f for f in findings if f.rule != "X17"]

    n_error = 0
    n_warn = 0
    n_bad_files = 0

    if args.as_json:
        payload = {"target": args.target, "files": []}
        for path, findings in results.items():
            errors = [f for f in findings if f.level == ERROR]
            warns = [f for f in findings if f.level == WARN]
            n_error += len(errors)
            n_warn += len(warns)
            if errors:
                n_bad_files += 1
            payload["files"].append({
                "path": path,
                "status": "FAIL" if errors else "OK",
                "findings": [{"level": f.level, "rule": f.rule,
                              "path": f.path, "message": f.message}
                             for f in findings],
            })
        payload["summary"] = {"files": len(results), "failed": n_bad_files,
                              "errors": n_error, "warnings": n_warn}
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    else:
        for path, findings in results.items():
            errors = [f for f in findings if f.level == ERROR]
            warns = [f for f in findings if f.level == WARN]
            n_error += len(errors)
            n_warn += len(warns)
            if errors:
                n_bad_files += 1
            if not findings:
                if not args.quiet:
                    sys.stdout.write("OK    %s\n" % path)
                continue
            sys.stdout.write("%s  %s\n" % ("FAIL" if errors else "WARN", path))
            for finding in findings:
                sys.stdout.write("      %s\n" % finding.format())
        sys.stdout.write(
            "\n%d file(s) checked, %d failed, %d error(s), %d warning(s)\n"
            % (len(results), n_bad_files, n_error, n_warn))

    if n_error:
        return 1
    if args.warn_as_error and n_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
