#!/usr/bin/env python3
"""Build the harmonized permit table: one Parquet file per source contribution, with the raw record preserved."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


SCRIPT_VERSION = "v103"
BATCH_SIZE = 65_536
SUPPORTED_SUFFIXES = {".parquet", ".csv", ".tsv", ".jsonl", ".ndjson"}


CANONICAL_FIELDS: list[tuple[str, pa.DataType]] = [
    ("record_id", pa.string()),
    ("source_contribution_id", pa.string()),
    ("source_materialization", pa.string()),
    ("source_registry", pa.string()),
    ("source_path", pa.string()),
    ("source_file", pa.string()),
    ("source_row_ordinal", pa.int64()),
    ("native_record_id", pa.string()),
    ("native_source_key", pa.string()),
    ("permit_number", pa.string()),
    ("description", pa.string()),
    ("record_type_native", pa.string()),
    ("status_native", pa.string()),
    ("application_datetime_utc", pa.string()),
    ("issue_datetime_utc", pa.string()),
    ("source_datetime_utc", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("iso6709", pa.string()),
    ("coordinate_reference_system", pa.string()),
    ("coordinate_provenance", pa.string()),
    ("raw_json", pa.string()),
]
CANONICAL_SCHEMA = pa.schema([pa.field(n, t) for n, t in CANONICAL_FIELDS])


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slug(value: str) -> str:
    value = re.sub(r"\.json$", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "source"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def as_string_array(table: pa.Table, candidates: Sequence[str]) -> pa.Array:
    """Return the first matching column cast to string, or a null array."""
    by_norm = {norm_name(name): name for name in table.column_names}
    for candidate in candidates:
        name = by_norm.get(norm_name(candidate))
        if name is None:
            continue
        arr = table[name]
        try:
            return pc.cast(arr, pa.string(), safe=False)
        except Exception:
            return pa.array(
                [None if x is None else str(x) for x in arr.to_pylist()],
                type=pa.string(),
            )
    return pa.nulls(table.num_rows, type=pa.string())


def matched_column(table: pa.Table, candidates: Sequence[str]) -> str | None:
    by_norm = {norm_name(name): name for name in table.column_names}
    for candidate in candidates:
        name = by_norm.get(norm_name(candidate))
        if name is not None:
            return name
    return None


def json_rows(table: pa.Table) -> list[str]:
    """Serialize complete source rows for files without a native raw field."""
    return [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=json_default)
        for row in table.to_pylist()
    ]


def raw_array(table: pa.Table) -> pa.Array:
    name = matched_column(table, ["raw", "raw_json", "source_record", "record_json"])
    if name is None:
        return pa.array(json_rows(table), type=pa.string())
    try:
        arr = pc.cast(table[name], pa.string(), safe=False)
        # Rebuild raw_json from native columns when the payload is null.
        if arr.null_count:
            values = arr.to_pylist()
            rows = table.to_pylist()
            values = [
                value
                if value is not None
                else json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=json_default)
                for value, row in zip(values, rows)
            ]
            return pa.array(values, type=pa.string())
        return arr
    except Exception:
        return pa.array(json_rows(table), type=pa.string())


def normalize_datetime(values: pa.Array) -> pa.Array:
    raw = values.to_pylist()
    series = pd.Series(raw, dtype="object")
    # Mixed date formats; values without a timezone are treated as UTC.
    # Microsecond precision keeps pre-1677 dates representable.
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[us, UTC]")
    numeric = pd.to_numeric(series, errors="coerce")
    # Infer the epoch unit (s, ms, us or ns) from magnitude.
    units = ((numeric.abs() >= 1e17, "ns"), (numeric.abs() >= 1e14, "us"),
             (numeric.abs() >= 1e11, "ms"), (numeric.abs() >= 1e8, "s"))
    numeric_mask = numeric.notna()
    yyyymmdd = numeric_mask & numeric.between(19000101, 20991231) & (numeric % 1 == 0)
    if yyyymmdd.any():
        parsed.loc[yyyymmdd] = pd.to_datetime(
            numeric.loc[yyyymmdd].astype("Int64").astype(str),
            format="%Y%m%d", errors="coerce", utc=True,
        ).dt.as_unit('us')
    for mask, unit in units:
        mask = mask & numeric_mask & parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(numeric.loc[mask], unit=unit, errors="coerce", utc=True).dt.as_unit('us')
    remaining = parsed.isna()
    if remaining.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[remaining] = pd.to_datetime(
                series.loc[remaining], format="mixed", errors="coerce", utc=True
            ).dt.as_unit('us')
    formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return pa.array([None if pd.isna(x) else str(x) for x in formatted], type=pa.string())


def extract_raw_values(raw_values: Sequence[str | None], aliases: Sequence[str]) -> list[str | None]:
    """Extract simple JSON scalar values without changing the original raw text."""
    key_pattern = "|".join(re.escape(x) for x in aliases)
    pattern = re.compile(
        rf'"(?:{key_pattern})"\s*:\s*(?:"((?:\\.|[^"\\])*)"|([-+0-9.eE]+)|null)',
        re.IGNORECASE,
    )
    out: list[str | None] = []
    for raw in raw_values:
        if not raw:
            out.append(None)
            continue
        match = pattern.search(raw)
        if not match:
            out.append(None)
        elif match.group(1) is not None:
            try:
                out.append(json.loads('"' + match.group(1) + '"'))
            except Exception:
                out.append(match.group(1))
        else:
            out.append(match.group(2))
    return out


def fill_from_raw(primary: pa.Array, raw: pa.Array, aliases: Sequence[str], raw_values: Sequence[str | None] | None = None) -> pa.Array:
    values = primary.to_pylist()
    if not any(value is None for value in values):
        return primary
    if raw_values is None:
        raw_values = raw.to_pylist()
    # Skip the regex when no requested key occurs in the batch.
    tokens = tuple(f'"{alias}"' for alias in aliases)
    if tokens and not any(token in text for text in raw_values if text for token in tokens):
        return primary
    extracted = extract_raw_values(raw_values, aliases)
    merged = [a if a not in (None, "") else b for a, b in zip(values, extracted)]
    return pa.array(merged, type=pa.string())


def normalize_float(values: pa.Array) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values.to_pylist(), dtype="object"), errors="coerce")
    return series.to_numpy(dtype="float64", na_value=np.nan)


def coordinate_arrays(table: pa.Table, raw: pa.Array | None = None, raw_values: Sequence[str | None] | None = None) -> tuple[pa.Array, pa.Array, pa.Array, pa.Array]:
    lat_name = matched_column(table, [
        "latitude", "lat", "y", "point_latitude", "geometry_latitude", "geo_lat",
    ])
    lon_name = matched_column(table, [
        "longitude", "lon", "lng", "long", "x", "point_longitude", "geometry_longitude", "geo_lon",
    ])
    n = table.num_rows
    if lat_name is None:
        lat_values = pa.nulls(n, pa.string())
    else:
        lat_values = as_string_array(table, [lat_name])
    if lon_name is None:
        lon_values = pa.nulls(n, pa.string())
    else:
        lon_values = as_string_array(table, [lon_name])
    if raw is not None:
        lat_values = fill_from_raw(lat_values, raw, ["latitude", "lat", "y"], raw_values)
        lon_values = fill_from_raw(lon_values, raw, ["longitude", "lon", "lng", "long", "x"], raw_values)
    lat = normalize_float(lat_values)
    lon = normalize_float(lon_values)
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
    iso: list[str | None] = []
    provenance: list[str] = []
    for la, lo, ok in zip(lat, lon, valid):
        if ok:
            # ISO 6709 signed decimal, WGS 84.
            iso.append(f"{la:+.6f}{lo:+.6f}/")
            provenance.append("direct_field")
        else:
            iso.append(None)
            provenance.append("invalid")
    lat_values = [None if not ok else float(la) for la, ok in zip(lat, valid)]
    lon_values = [None if not ok else float(lo) for lo, ok in zip(lon, valid)]
    return (pa.array(lat_values, type=pa.float64()), pa.array(lon_values, type=pa.float64()),
            pa.array(iso, type=pa.string()), pa.array(provenance, type=pa.string()))


def read_batches(path: Path, batch_size: int = BATCH_SIZE) -> Iterator[tuple[pa.RecordBatch, Path]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, use_threads=True):
            yield batch, path
    elif suffix in {".csv", ".tsv"}:
        parse = pacsv.ParseOptions(delimiter="\t" if suffix == ".tsv" else ",")
        reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=8 * 1024 * 1024), parse_options=parse)
        while True:
            try:
                yield reader.read_next_batch(), path
            except StopIteration:
                break
    elif suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= batch_size:
                    yield pa.RecordBatch.from_pylist(rows), path
                    rows = []
        if rows:
            yield pa.RecordBatch.from_pylist(rows), path
    else:
        raise ValueError(f"Unsupported source file: {path}")


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    parquet = [p for p in files if p.suffix.lower() == ".parquet"]
    return parquet or files


def resolve_entry(entry: dict[str, Any], roots: Sequence[Path]) -> list[Path]:
    raw_path = str(entry["path"])
    # Accept Windows-style ledger paths on Linux.
    normalized_path = raw_path.replace("\\", "/")
    original = Path(normalized_path)
    ledger_absolute = normalized_path.startswith("/") or bool(re.match(r"^[A-Za-z]:/", normalized_path))
    candidates: list[Path] = []
    if ledger_absolute and original.exists():
        candidates.append(original)
    for root in roots:
        root = root.resolve()
        if not ledger_absolute:
            candidates.append(root / original)
            continue
        parts = original.parts
        # Map absolute ledger paths onto the supplied project root.
        for marker in ("candy_permit150m_20260908", "permit150m_20260908"):
            if marker in parts:
                idx = parts.index(marker)
                candidates.append(root.joinpath(*parts[idx + 1 :]))
        candidates.extend([root / "data" / original.name, root / original.name])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        files = discover_files(candidate)
        if files:
            return files
    return []


def source_signatures(files: Sequence[Path], hash_sources: bool = False) -> list[dict[str, Any]]:
    """Return change-detection metadata for source files, optionally with SHA-256."""
    result: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        item: dict[str, Any] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_sources:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        result.append(item)
    return result


def source_file_label(source_file: Path, roots: Sequence[Path]) -> str:
    """Return a portable relative label for a source file."""
    for root in roots:
        try:
            return str(source_file.resolve().relative_to(root.resolve()))
        except ValueError:
            continue
    return str(source_file)


def select_column_array(table: pa.Table, candidates: Sequence[str]) -> pa.Array:
    return as_string_array(table, candidates)


def build_batch(
    batch: pa.RecordBatch,
    entry: dict[str, Any],
    source_file: Path,
    source_roots: Sequence[Path],
    row_start: int,
) -> pa.Table:
    table = pa.Table.from_batches([batch])
    n = table.num_rows
    contribution_id = slug(entry["materialization"])
    contribution_arr = pa.array([contribution_id] * n, type=pa.string())
    ord_arr = pa.array(range(row_start, row_start + n), type=pa.int64())
    record_id = pc.binary_join_element_wise(
        contribution_arr, pc.cast(ord_arr, pa.string()), "|"
    )
    raw = raw_array(table)
    raw_values = raw.to_pylist()

    source_key = select_column_array(table, ["source_key", "source_id", "source_dataset", "dataset_id"])
    native_id = select_column_array(table, [
        "source_system_row_id", "source_object_id", "native_record_id", "record_id",
        "source_row_ordinal", "source_record_ordinal", "permit_number", "permit_no",
        "record_number", "record_no", "record #", "original_stored_permit_id", "original_file_row_number",
    ])
    source_key = fill_from_raw(source_key, raw, ["source_key", "source_id", "source_dataset", "dataset_id"], raw_values)
    native_id = fill_from_raw(native_id, raw, ["source_system_row_id", "source_object_id", "record_id", "permit_number", "record_number", "record_no"], raw_values)
    permit_number = select_column_array(table, [
        "permit_number", "permit_no", "permit_num", "permit", "record_number", "record_no",
        "record #", "NUM", "number", "original_stored_permit_id",
    ])
    description = select_column_array(table, [
        "description", "work_description", "work_desc", "project_description", "permit_description",
        "description_extracted", "permit_desc", "scope_of_work", "approvedscopeofwork",
        "original_description", "WORK_DESC", "short_notes", "project_name",
    ])
    record_type = select_column_array(table, [
        "record_type", "permit_type", "permit_subtype", "work_type", "sub_type",
        "application_type", "type", "original_native_label", "WORK_TYPE",
    ])
    status = select_column_array(table, ["status", "permit_status", "application_status", "STATUS"])
    permit_number = fill_from_raw(permit_number, raw, ["permit_number", "permit_no", "permit", "record_number", "record_no", "record #", "number", "NUM"], raw_values)
    description = fill_from_raw(description, raw, ["description", "work_description", "work_desc", "project_description", "permit_description", "permit_desc", "scope_of_work", "approvedscopeofwork", "project_name"], raw_values)
    record_type = fill_from_raw(record_type, raw, ["record_type", "permit_type", "work_type", "sub_type", "application_type", "type"], raw_values)
    status = fill_from_raw(status, raw, ["status", "permit_status", "application_status"], raw_values)

    app_raw = select_column_array(table, [
        "source_applied_date", "application_date", "applicationdate", "applied_date", "date_applied",
        "applydate", "applied",
    ])
    issue_raw = select_column_array(table, [
        "source_approved_date", "issue_date", "issued_date", "permit_date", "date_issued",
        "issuedate", "approved_date", "approveddate", "issued",
    ])
    source_raw = select_column_array(table, [
        "source_date", "issued_date", "issue_date", "issuedate", "date", "PERMDT", "issued_date_raw",
    ])
    app_raw = fill_from_raw(app_raw, raw, ["source_applied_date", "application_date", "applicationdate", "applied_date", "applieddate", "date_applied", "applydate", "applied"], raw_values)
    issue_raw = fill_from_raw(issue_raw, raw, ["source_approved_date", "approved_date", "approveddate", "issue_date", "issued_date", "permit_date", "date_issued", "issuedate", "issued"], raw_values)
    source_raw = fill_from_raw(source_raw, raw, ["source_date", "issued_date", "issue_date", "issuedate", "date", "PERMDT"], raw_values)
    lat, lon, iso, coord_provenance = coordinate_arrays(table, raw, raw_values)
    source_file_value = source_file_label(source_file, source_roots)

    arrays = [
        record_id,
        contribution_arr,
        pa.array([entry["materialization"]] * n, type=pa.string()),
        pa.array([entry.get("registry")] * n, type=pa.string()),
        pa.array([entry["path"]] * n, type=pa.string()),
        pa.array([source_file_value] * n, type=pa.string()),
        ord_arr,
        native_id,
        source_key,
        permit_number,
        description,
        record_type,
        status,
        normalize_datetime(app_raw),
        normalize_datetime(issue_raw),
        normalize_datetime(source_raw),
        lat,
        lon,
        iso,
        pa.array(["EPSG:4326/WGS84"] * n, type=pa.string()),
        coord_provenance,
        raw,
    ]
    return pa.Table.from_arrays(arrays, schema=CANONICAL_SCHEMA)


@dataclasses.dataclass
class ProcessResult:
    contribution_id: str
    expected_rows: int
    output_rows: int
    output_path: str | None
    status: str
    source_files: list[str]
    error: str | None = None
    elapsed_seconds: float = 0.0


def process_entry(payload: dict[str, Any]) -> dict[str, Any]:
    entry = payload["entry"]
    roots = [Path(x) for x in payload["roots"]]
    output_root = Path(payload["output_root"])
    batch_size = int(payload["batch_size"])
    force = bool(payload["force"])
    hash_sources = bool(payload.get("hash_sources", False))
    contribution_id = slug(entry["materialization"])
    # Plain directory names avoid Hive partition inference.
    out_dir = output_root / "permit_master" / contribution_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part-00000.parquet"
    marker = out_dir / "_SUCCESS.json"
    config_hash = payload["config_hash"]
    files = resolve_entry(entry, roots)
    signatures = source_signatures(files, hash_sources=hash_sources) if files else []
    if not force and marker.exists() and out_path.exists() and files:
        try:
            old = json.loads(marker.read_text())
            if old.get("config_hash") == config_hash and old.get("source_signatures") == signatures:
                return dataclasses.asdict(ProcessResult(contribution_id, int(entry["records"]), int(old["output_rows"]), str(out_path), "skipped", old.get("source_files", []), elapsed_seconds=0.0))
        except Exception:
            pass

    if not files:
        return dataclasses.asdict(ProcessResult(contribution_id, int(entry["records"]), 0, None, "missing_source", [], "source path could not be resolved"))

    started = time.time()
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    writer: pq.ParquetWriter | None = None
    total = 0
    source_files: list[str] = []
    try:
        writer = pq.ParquetWriter(tmp_path, CANONICAL_SCHEMA, compression="zstd", use_dictionary=True)
        for source_file in files:
            source_files.append(str(source_file))
            offset = total
            for batch, actual_file in read_batches(source_file, batch_size=batch_size):
                derived = build_batch(batch, entry, actual_file, roots, offset)
                writer.write_table(derived, row_group_size=batch_size)
                total += derived.num_rows
                offset += derived.num_rows
        writer.close()
        writer = None
        if total != int(entry["records"]):
            raise ValueError(f"row count mismatch: expected {entry['records']}, produced {total}")
        os.replace(tmp_path, out_path)
        marker.write_text(json.dumps({
            "script_version": SCRIPT_VERSION,
            "config_hash": config_hash,
            "contribution_id": contribution_id,
            "expected_rows": int(entry["records"]),
            "output_rows": total,
            "source_files": source_files,
            "source_signatures": signatures,
            "output_path": str(out_path),
            "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }, indent=2) + "\n")
        return dataclasses.asdict(ProcessResult(contribution_id, int(entry["records"]), total, str(out_path), "written", source_files, elapsed_seconds=time.time() - started))
    except Exception as exc:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        return dataclasses.asdict(ProcessResult(contribution_id, int(entry["records"]), total, None, "error", source_files, repr(exc), time.time() - started))


def load_entries(ledger_path: Path) -> list[dict[str, Any]]:
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = data.get("contributions")
    if not isinstance(entries, list):
        raise ValueError("ledger does not contain a contributions list")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-root", action="append", required=True, type=Path, help="Repeat for each source snapshot root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--only", nargs="*", help="Optional contribution IDs/materialization names")
    parser.add_argument("--allow-missing", action="store_true", help="Record unresolved contributions and continue")
    parser.add_argument("--force", action="store_true", help="Rebuild existing derived outputs")
    parser.add_argument("--hash-sources", action="store_true", help="Hash source files for strong resume/change detection")
    parser.add_argument("--plan", action="store_true", help="Resolve sources and write a plan without reading records")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing derived outputs without reading sources")
    args = parser.parse_args()

    ledger_path = args.ledger.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entries = load_entries(ledger_path)
    if args.only:
        requested = {slug(x) for x in args.only}
        entries = [x for x in entries if slug(x["materialization"]) in requested or x["materialization"] in args.only]
    config = {
        "script_version": SCRIPT_VERSION,
        "ledger": str(ledger_path),
        "source_roots": [str(x.resolve()) for x in args.source_root],
        "batch_size": args.batch_size,
        "canonical_fields": [(n, str(t)) for n, t in CANONICAL_FIELDS],
        "datetime_policy": "ISO-8601 UTC with Z; naive source values interpreted as UTC and preserved in raw_json",
        "coordinate_policy": "ISO 6709 signed decimal; WGS84/EPSG:4326",
        "hash_sources": args.hash_sources,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.plan:
        plan = []
        for entry in entries:
            files = resolve_entry(entry, args.source_root)
            plan.append({"contribution_id": slug(entry["materialization"]), "expected_rows": int(entry["records"]), "resolved_files": [str(x) for x in files], "status": "ready" if files else "missing_source"})
        plan_path = output_root / "pipeline_plan.json"
        plan_path.write_text(json.dumps({"script_version": SCRIPT_VERSION, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "config": config, "entries": plan}, indent=2) + "\n")
        print(json.dumps({"status": "planned", "entries": len(plan), "ready": sum(x["status"] == "ready" for x in plan), "missing": sum(x["status"] != "ready" for x in plan), "plan": str(plan_path)}))
        return 0 if all(x["status"] == "ready" for x in plan) or args.allow_missing else 2
    if args.verify_only:
        manifest = {"script_version": SCRIPT_VERSION, "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "output_root": str(output_root), "entries": []}
        for entry in entries:
            contribution_id = slug(entry["materialization"])
            out_path = output_root / "permit_master" / contribution_id / "part-00000.parquet"
            marker = out_path.parent / "_SUCCESS.json"
            item = {"contribution_id": contribution_id, "expected_rows": int(entry["records"]), "output_path": str(out_path), "status": "missing"}
            if out_path.exists() and marker.exists():
                try:
                    item["output_rows"] = pq.ParquetFile(out_path).metadata.num_rows
                    item["status"] = "verified" if item["output_rows"] == item["expected_rows"] else "row_mismatch"
                except Exception as exc:
                    item["status"] = "error"; item["error"] = repr(exc)
            manifest["entries"].append(item)
        (output_root / "pipeline_verify.json").write_text(json.dumps(manifest, indent=2) + "\n")
        bad = [x for x in manifest["entries"] if x["status"] != "verified"]
        print(json.dumps({"status": "verified" if not bad else "failed", "entries": len(manifest["entries"]), "bad": len(bad), "report": str(output_root / "pipeline_verify.json")}))
        return 0 if not bad else 2
    run_manifest = output_root / "master_build_manifest.json"
    run_state = {
        "script_version": SCRIPT_VERSION,
        "status": "running",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "ledger": str(ledger_path),
        "source_roots": [str(x.resolve()) for x in args.source_root],
        "output_root": str(output_root),
        "config_hash": config_hash,
        "expected_contributions": len(entries),
        "expected_rows": sum(int(x["records"]) for x in entries),
        "results": [],
    }
    run_manifest.write_text(json.dumps(run_state, indent=2) + "\n")

    payloads = [
        {"entry": entry, "roots": [str(x.resolve()) for x in args.source_root], "output_root": str(output_root), "batch_size": args.batch_size, "force": args.force, "hash_sources": args.hash_sources, "config_hash": config_hash}
        for entry in entries
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_entry, payload) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            run_state["results"] = sorted(results, key=lambda x: x["contribution_id"])
            run_state["completed_contributions"] = len(results)
            run_state["completed_rows"] = sum(int(x["output_rows"]) for x in results)
            run_manifest.write_text(json.dumps(run_state, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    missing = [x for x in results if x["status"] == "missing_source"]
    errors = [x for x in results if x["status"] == "error"]
    run_state["finished_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    run_state["status"] = "complete" if not errors and (args.allow_missing or not missing) else "failed"
    run_state["missing_contributions"] = len(missing)
    run_state["error_contributions"] = len(errors)
    run_manifest.write_text(json.dumps(run_state, indent=2) + "\n")
    if errors or (missing and not args.allow_missing):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
