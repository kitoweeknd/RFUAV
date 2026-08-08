# -*- coding: utf-8 -*-
"""
iq_dtype.py -- the single source of truth for "how wide is one IQ sample".

WHY THIS MODULE EXISTS (verificacion-empirica-06, defect class D9)
------------------------------------------------------------------------------
The library holds 93 files / 53.43 GB of ci16_le (interleaved int16) IQ, in two
foreign roots that RFUAV never captured itself:

    <ROOT_ZJU>        81 files, sample_rate_msps BLANK
    .dat_pic          12 files, 75 MS/s

Every reader in this repository was written as if float32 were the only thing
that can appear on disk.  The producer side already knew better --
build_manifests.py has carried a correct dtype vocabulary and a correct
path->dtype rule table since the manifests were first built -- but *no consumer
ever read either of them*.  That asymmetry IS the defect: the knowledge exists
and is not consulted.  Four failure sites followed from it:

    D9-1  pipeline_runner.load_iq_file      `sample_size = 8` hard-coded.
                                            SILENT: halves the reported sample
                                            count and the derived duration.
    D9-2  pipeline_runner._load_memmap      `.view(np.float32)` on an int16
                                            buffer -> noisy ValueError.
    D9-3  ChannelSimulation.core.io_utils   `raw[0::2] + 1j*raw[1::2]` promotes
                                            int16 to complex128 -- SILENT, and
                                            a breach of the complex64 invariant.
    D9-4  SNREstimation CLI                 --dtype cannot even spell int16.

Today all 93 files are >= 457.8 MB, so all of them take the memmap branch and
the failure is 100 % noisy.  D9-1 is latent, not fixed: it wakes up the moment
anything slices one of those files below the 100 MB memmap threshold, which
this project does routinely.  No published RFUAV number is affected -- every
released figure came from cf32_le data -- so the damage is entirely
prospective.  What it blocks is G1: DroneRFa is natively int16.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT DO
------------------------------------------------------------------------------
Does:   own the token vocabulary; convert a manifest token (or a legacy numpy
        dtype name) into the on-disk component dtype, the in-memory complex
        dtype, and the bytes-per-complex-sample; resolve a file path to its
        recorded datatype via recordings.csv.

Does NOT: read IQ files.  Three readers already exist (graphic/io_utils.py,
        ChannelSimulation/core/io_utils.py, pipeline_runner._load_memmap).  A
        fourth would be a fourth thing to keep in sync, which is the disease
        this module is treating.  Library code deep inside ChannelSimulation
        must NOT import from tools/ either -- tools depend on the application,
        never the reverse -- so those two call sites are fixed locally with
        itemsize arithmetic instead.  Only CLIs import this module.

THERE IS NO DEFAULT
------------------------------------------------------------------------------
`from_manifest_token` and `DtypeResolver.for_path` raise on anything they
cannot establish.  A silent fallback to float32 is not a safety net, it is
precisely D9-1: the default IS the defect.  A caller who genuinely wants to
assume something must say so at the call site, and the assumption is then
recorded in the returned evidence field so it survives into a log.

NAMING
------------------------------------------------------------------------------
Canonical tokens are SigMF's: cf32_le, cf64_le, ci16_le, ci8.  Earlier drafts of
spec section 8 wrote `sc16_le` for the int16 case, which is not a SigMF token;
that prose was migrated to ci16_le on 2026-07-31, and `sc16_le` stays here as a
deprecated alias so older documents and any copies outside this repo keep
resolving rather than becoming hard errors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DtypeError",
    "IQDtype",
    "DtypeAssignment",
    "DtypeResolver",
    "VOCAB",
    "ALIASES",
    "BYTES_PER_SAMPLE",
    "from_manifest_token",
    "from_numpy_component",
    "bytes_per_complex_sample",
    "complex_dtype_for",
    "n_complex_samples",
]


class DtypeError(ValueError):
    """A sample dtype could not be established with certainty.

    Deliberately a subclass of ValueError so existing `except ValueError`
    handlers around file loading keep working, but distinguishable for callers
    that want to tell "this file is malformed" from "I do not know what this
    file is".
    """


# ---------------------------------------------------------------------------
# dtype arithmetic -- derived from itemsize, never hard-coded
# ---------------------------------------------------------------------------
def bytes_per_complex_sample(dtype) -> int:
    """Bytes occupied on disk by ONE complex sample (I and Q together).

    Accepts either the component dtype of an interleaved file (the normal
    case: float32 -> 8) or an already-complex dtype (complex64 -> 8).  This is
    the arithmetic that D9-1 hard-coded to 8.
    """
    d = np.dtype(dtype)
    if d.kind == "c":
        return int(d.itemsize)
    return int(2 * d.itemsize)


def complex_dtype_for(dtype) -> np.dtype:
    """The complex dtype to decode an interleaved file of `dtype` into.

    <=4-byte components (int8, int16, float32) decode into complex64; wider
    ones into complex128.  int16 decoding into complex64 is the whole point:
    the old `raw[0::2] + 1j*raw[1::2]` idiom produced complex128 for integer
    input, quadrupling memory and breaching the complex64 invariant that
    channels/normalize_power depend on.
    """
    d = np.dtype(dtype)
    if d.kind == "c":
        return d
    return np.dtype(np.complex64) if d.itemsize <= 4 else np.dtype(np.complex128)


def n_complex_samples(file_size_bytes: int, dtype) -> int:
    """How many whole complex samples fit in a file of `file_size_bytes`.

    Named because getting it wrong is D9-1, and a named function can be
    tested; an inline `size // 8` cannot.
    """
    if file_size_bytes < 0:
        raise DtypeError(f"negative file size: {file_size_bytes}")
    return int(file_size_bytes) // bytes_per_complex_sample(dtype)


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IQDtype:
    """One entry of the interleaved-IQ datatype vocabulary."""

    token: str          # canonical SigMF token, e.g. "ci16_le"
    component: np.dtype  # on-disk dtype of a single component (I or Q)
    complex_dtype: np.dtype  # complex dtype to decode into
    is_integer: bool    # integer components carry no absolute dB scale

    @property
    def bytes_per_sample(self) -> int:
        """Bytes per COMPLEX sample -- the manifest's bytes_per_sample column."""
        return bytes_per_complex_sample(self.component)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.token


def _entry(token: str, component: str) -> IQDtype:
    comp = np.dtype(component)
    return IQDtype(
        token=token,
        component=comp,
        complex_dtype=complex_dtype_for(comp),
        is_integer=comp.kind in "iu",
    )


VOCAB: dict[str, IQDtype] = {
    e.token: e
    for e in (
        _entry("cf32_le", "<f4"),
        _entry("cf64_le", "<f8"),
        _entry("ci16_le", "<i2"),
        _entry("ci8", "i1"),
    )
}

# Accepted spellings that are NOT canonical.  Kept small and explicit: every
# entry here is a spelling that already exists somewhere in this repo's prose
# or CLIs, not a convenience invented for its own sake.
ALIASES: dict[str, str] = {
    # SDR-vendor spelling of interleaved int16; early spec/§8 prose used it
    # before the 2026-07-31 migration to SigMF's ci16_le.  Kept so those older
    # documents keep resolving.
    "sc16_le": "ci16_le",
    "sc8": "ci8",
    # legacy numpy component/complex names, as the old --dtype flags spelled them
    "float32": "cf32_le",
    "f4": "cf32_le",
    "complex64": "cf32_le",
    "float64": "cf64_le",
    "f8": "cf64_le",
    "complex128": "cf64_le",
    "int16": "ci16_le",
    "i2": "ci16_le",
    "int8": "ci8",
    "i1": "ci8",
}

# The manifest's bytes_per_sample lookup, derived rather than typed out.
# build_manifests.py imports this instead of redeclaring it, so the producer
# and the consumers cannot drift apart -- drift is what D9 was.
BYTES_PER_SAMPLE: dict[str, int] = {
    token: e.bytes_per_sample for token, e in VOCAB.items()
}


def _vocab_help() -> str:
    canon = ", ".join(sorted(VOCAB))
    alias = ", ".join(f"{a}->{c}" for a, c in sorted(ALIASES.items()))
    return (
        f"canonical tokens: {canon}; accepted aliases: {alias}. "
        "There is deliberately no default -- assuming float32 is defect D9."
    )


def from_manifest_token(token: str | None) -> IQDtype:
    """Resolve a manifest `datatype` token (or a legacy alias) to an IQDtype.

    Raises DtypeError on anything unrecognised, blank, or None.
    """
    if token is None:
        raise DtypeError(f"no IQ datatype given ({_vocab_help()})")
    key = str(token).strip().lower()
    if not key:
        raise DtypeError(f"blank IQ datatype ({_vocab_help()})")
    key = ALIASES.get(key, key)
    entry = VOCAB.get(key)
    if entry is None:
        raise DtypeError(f"unknown IQ datatype {token!r} ({_vocab_help()})")
    return entry


def from_numpy_component(dtype) -> IQDtype:
    """Resolve a numpy dtype to an IQDtype.

    For legacy call sites that already hold a numpy dtype (the old
    `--dtype float32` path, and readers whose signature takes `dtype=np.float32`).
    An already-complex dtype resolves through its component width, so
    complex64 -> cf32_le.
    """
    d = np.dtype(dtype)
    if d.kind == "c":
        if d.itemsize % 2:
            raise DtypeError(f"complex dtype with odd itemsize: {d!r}")
        d = np.dtype(f"<f{d.itemsize // 2}")
    for entry in VOCAB.values():
        if entry.component == d:
            return entry
    raise DtypeError(f"no IQ datatype token for numpy dtype {d!r} ({_vocab_help()})")


# ---------------------------------------------------------------------------
# path -> datatype, out of the manifest
# ---------------------------------------------------------------------------
_MEASURED = "measured"
_CALLER_DEFAULT = "CALLER_DEFAULT"


@dataclass(frozen=True)
class DtypeAssignment:
    """What the manifest says about one file's dtype, and how sure it is."""

    dtype: IQDtype
    evidence: str
    source_relpath: str

    @property
    def measured(self) -> bool:
        """True only for dtypes actually probed off the bytes.

        Anything else -- ASSUMED_UNPROBED from the manifest's path rules, or
        CALLER_DEFAULT injected at a call site -- is a guess, and a number
        derived from a guessed dtype must not be published as measured.
        """
        return self.evidence.strip().lower() == _MEASURED


def _norm(path) -> str:
    """Normalise a path for keying: forward slashes, lowercase, no leading ./"""
    s = str(path).strip().replace("\\", "/").lstrip("/")
    while s.startswith("./"):
        s = s[2:]
    return s.lower()


class DtypeResolver:
    """Answers "what dtype is this file?" from a recordings.csv manifest.

    Lookup order, most specific first:
      1. the exact normalised relative path
      2. progressively shorter path suffixes (so an absolute path such as
         E:/Drone_dataset/RFUAV/rawdata/x.iq matches the manifest's
         rawdata/x.iq)
      3. the basename, but ONLY when that basename is unique across the whole
         manifest.  Non-unique basenames raise rather than pick one: the
         library is full of same-named files under different roots (E-011),
         and silently choosing is how you attribute int16 bytes to a float32
         recording.
    """

    def __init__(self, entries: dict[str, DtypeAssignment],
                 basenames: dict[str, DtypeAssignment],
                 ambiguous: set[str]) -> None:
        self._by_path = entries
        self._by_base = basenames
        self._ambiguous = ambiguous

    # -- construction -------------------------------------------------------
    @classmethod
    def from_recordings_csv(cls, csv_path, *, strict: bool = True) -> "DtypeResolver":
        """Build a resolver from a manifests/recordings.csv.

        `strict` (default) cross-checks each row's bytes_per_sample column
        against the vocabulary and raises on disagreement.  A manifest that
        disagrees with itself must stop the run: silently trusting one of the
        two columns is how a 53 GB dtype error becomes a published number.
        """
        csv_path = Path(csv_path)
        by_path: dict[str, DtypeAssignment] = {}
        by_base: dict[str, DtypeAssignment] = {}
        ambiguous: set[str] = set()

        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"source_relpath", "datatype"} - set(reader.fieldnames or ())
            if missing:
                raise DtypeError(
                    f"{csv_path}: recordings manifest lacks column(s) "
                    f"{sorted(missing)}; found {list(reader.fieldnames or ())}"
                )
            for lineno, row in enumerate(reader, start=2):
                relpath = (row.get("source_relpath") or "").strip()
                if not relpath:
                    continue
                dtype = from_manifest_token(row.get("datatype"))
                if strict:
                    declared = (row.get("bytes_per_sample") or "").strip()
                    if declared and int(declared) != dtype.bytes_per_sample:
                        raise DtypeError(
                            f"{csv_path}:{lineno}: manifest disagrees with itself "
                            f"for {relpath!r}: datatype={dtype.token} implies "
                            f"{dtype.bytes_per_sample} bytes/sample but the "
                            f"bytes_per_sample column says {declared}"
                        )
                assignment = DtypeAssignment(
                    dtype=dtype,
                    evidence=(row.get("dtype_evidence") or "").strip() or "UNRECORDED",
                    source_relpath=relpath,
                )
                key = _norm(relpath)
                by_path[key] = assignment
                base = key.rsplit("/", 1)[-1]
                if base in by_base and by_base[base].source_relpath != relpath:
                    ambiguous.add(base)
                else:
                    by_base[base] = assignment

        if not by_path:
            raise DtypeError(f"{csv_path}: no usable rows (empty manifest?)")
        for base in ambiguous:
            by_base.pop(base, None)
        return cls(by_path, by_base, ambiguous)

    # -- query --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_path)

    def for_path(self, path, *, require_measured: bool = False,
                 default: str | None = None) -> DtypeAssignment:
        """Resolve one file path to its recorded dtype.

        require_measured  refuse an ASSUMED_UNPROBED row.  Set this for any
                          computation whose output will be published.
        default           a token to fall back on when the manifest has no row
                          for this path.  Passing it is an explicit, recorded
                          assumption (evidence=CALLER_DEFAULT), not a silent
                          one; omitting it makes an unknown path an error.
        """
        key = _norm(path)
        found = self._by_path.get(key)
        if found is None:
            # progressively drop leading components: absolute -> relative
            parts = key.split("/")
            for i in range(1, len(parts)):
                found = self._by_path.get("/".join(parts[i:]))
                if found is not None:
                    break
        if found is None:
            base = key.rsplit("/", 1)[-1]
            if base in self._ambiguous:
                raise DtypeError(
                    f"{path}: basename {base!r} appears under more than one "
                    "manifest root, so a basename match would be a guess; "
                    "pass a path that matches the manifest's source_relpath"
                )
            found = self._by_base.get(base)
        if found is None:
            if default is not None:
                return DtypeAssignment(
                    dtype=from_manifest_token(default),
                    evidence=_CALLER_DEFAULT,
                    source_relpath=str(path),
                )
            raise DtypeError(
                f"{path}: not in the recordings manifest ({len(self)} rows). "
                "Refusing to guess a dtype -- pass default= if an assumption "
                "is genuinely acceptable here, and it will be recorded as one."
            )
        if require_measured and not found.measured:
            raise DtypeError(
                f"{found.source_relpath}: dtype {found.dtype.token} has "
                f"evidence={found.evidence!r}, not {_MEASURED!r}; refusing to "
                "produce a publishable number from an unprobed dtype"
            )
        return found
