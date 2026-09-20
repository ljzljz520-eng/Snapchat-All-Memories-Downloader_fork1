import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import exif
import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm.asyncio import tqdm

DATE_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
# Snapchat exports use one exact format; strptime alone would also accept
# non-zero-padded dates, so enforce the shape explicitly.
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")
FINGERPRINT_LEN = 16


class Memory(BaseModel):
    date: datetime = Field(alias="Date")
    download_link: str = Field(alias="Download Link")
    location: str = Field(default="", alias="Location")
    latitude: float | None = None
    longitude: float | None = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S UTC")
        return v

    def model_post_init(self, __context):
        if self.location and not self.latitude:
            if match := re.search(r"([-\d.]+),\s*([-\d.]+)", self.location):
                self.latitude = float(match.group(1))
                self.longitude = float(match.group(2))

    @property
    def filename(self) -> str:
        return self.date.strftime("%Y-%m-%d_%H-%M-%S")


class Stats(BaseModel):
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    mb: float = 0


class MemImportError(ValueError):
    """Top-level contract violation of the memories export JSON."""


class RejectedRecord(BaseModel):
    # Zero-based index of the record inside the source "Saved Media" list.
    index: int
    # Name of the offending source field (e.g. "Date", "Location").
    field: str
    reason: str
    # Stable one-way digest; never carries the Download Link itself.
    fingerprint: str


class ImportResult(BaseModel):
    memories: list[Memory]
    rejected: list[RejectedRecord]

    @property
    def total(self) -> int:
        return len(self.memories) + len(self.rejected)


def record_fingerprint(item: object) -> str:
    """Stable, non-sensitive fingerprint of a raw source record.

    SHA-256 over canonical JSON: the same record yields the same fingerprint
    on repeated preflight runs, while the Download Link's query string and CDN
    signature cannot be recovered from the digest.
    """
    canonical = json.dumps(
        item, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]


def validate_record(
    item: object, index: int
) -> tuple[Memory | None, RejectedRecord | None]:
    """Validate one raw record without touching the network or filesystem."""
    fingerprint = record_fingerprint(item)

    def reject(field: str, reason: str) -> RejectedRecord:
        return RejectedRecord(
            index=index, field=field, reason=reason, fingerprint=fingerprint
        )

    if not isinstance(item, dict):
        return None, reject(
            "<record>", f"record must be a JSON object, got {type(item).__name__}"
        )

    if "Date" not in item:
        return None, reject("Date", "missing required field")
    raw_date = item["Date"]
    if not isinstance(raw_date, str):
        return None, reject("Date", f"expected string, got {type(raw_date).__name__}")
    if not DATE_RE.match(raw_date):
        return None, reject(
            "Date",
            f"does not match the required format {DATE_FORMAT!r}; got {raw_date!r}",
        )
    try:
        datetime.strptime(raw_date, DATE_FORMAT)
    except ValueError:
        return None, reject("Date", f"not a valid calendar date; got {raw_date!r}")

    # Validate the link's presence/shape but never put the value (which
    # contains signed query parameters) into a rejection reason.
    if "Download Link" not in item:
        return None, reject("Download Link", "missing required field")
    link = item["Download Link"]
    if not isinstance(link, str):
        return None, reject(
            "Download Link", f"expected a URL string, got {type(link).__name__}"
        )
    if not link.strip():
        return None, reject("Download Link", "URL is empty")

    location = item.get("Location", "")
    if not isinstance(location, str):
        return None, reject(
            "Location", f"expected string, got {type(location).__name__}"
        )
    if location:
        match = re.search(r"([-\d.]+),\s*([-\d.]+)", location)
        if match:
            try:
                float(match.group(1))
                float(match.group(2))
            except ValueError:
                return None, reject(
                    "Location",
                    "coordinate-looking values are not numeric: "
                    f"latitude={match.group(1)!r}, longitude={match.group(2)!r}",
                )

    # Backstop: report any remaining pydantic errors per field instead of
    # letting one bad record abort the whole import.
    try:
        memory = Memory(**item)
    except ValidationError as e:
        first_error = e.errors()[0]
        field = ".".join(str(part) for part in first_error["loc"]) or "<record>"
        return None, reject(field, first_error.get("msg", "validation failed"))
    return memory, None


def load_memories(json_path: Path) -> ImportResult:
    """Parse and preflight the export file. Offline: no network, no mkdir.

    Top-level contract violations raise MemImportError regardless of the
    import mode; individual malformed records are collected into
    ``ImportResult.rejected`` so the caller can choose strict abort or
    partial import. Valid memories keep their original source order.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MemImportError(
            f"top-level contract violation: {json_path} is not valid JSON "
            f"({e.msg} at line {e.lineno} column {e.colno}); expected an object "
            "containing a 'Saved Media' list"
        ) from e
    except OSError as e:
        raise MemImportError(f"cannot read memories file {json_path}: {e}") from e

    if not isinstance(data, dict):
        raise MemImportError(
            "top-level contract violation: expected the JSON document to be an "
            f"object with a 'Saved Media' list, got {type(data).__name__}"
        )
    if "Saved Media" not in data:
        raise MemImportError(
            "top-level contract violation: required key 'Saved Media' is missing; "
            "the Snapchat export must be an object containing a 'Saved Media' list"
        )
    records = data["Saved Media"]
    if not isinstance(records, list):
        raise MemImportError(
            "top-level contract violation: 'Saved Media' must be a list, got "
            f"{type(records).__name__}"
        )

    memories: list[Memory] = []
    rejected: list[RejectedRecord] = []
    for index, item in enumerate(records):
        memory, rejection = validate_record(item, index)
        if rejection is not None:
            rejected.append(rejection)
        if memory is not None:
            memories.append(memory)
    return ImportResult(memories=memories, rejected=rejected)


def format_preflight_report(result: ImportResult, partial: bool) -> str:
    """Render the isolation report.

    The report contains only source indexes, the offending field, a reason
    and a one-way fingerprint. It must never include a Download Link, its
    query string, or a full CDN signature.
    """
    lines = [
        f"Preflight: {result.total} record(s) scanned = "
        f"{len(result.memories)} valid + {len(result.rejected)} rejected",
        (
            "Rejected records (skipped; no download request will be sent):"
            if partial
            else "Rejected records:"
        ),
    ]
    for r in result.rejected:
        lines.append(
            f"  - source index={r.index} (0-based; #{r.index + 1} in file order) "
            f"field='{r.field}' reason={r.reason!r} fingerprint={r.fingerprint}"
        )
    if partial:
        lines.append(
            f"Proceeding with partial import: {len(result.memories)} valid "
            "record(s) keep their original order."
        )
    else:
        lines.append(
            "Aborted in strict mode: 0 download request(s) sent and no output "
            "directory was created. Re-run with --partial to import valid records."
        )
    return "\n".join(lines)


async def get_cdn_url(download_link: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            download_link,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return response.text.strip()


def add_exif_data(image_path: Path, memory: Memory):
    try:
        with open(image_path, "rb") as f:
            img = exif.Image(f)

        dt_str = memory.date.strftime("%Y:%m:%d %H:%M:%S")
        img.datetime_original = dt_str
        img.datetime_digitized = dt_str
        img.datetime = dt_str

        if memory.latitude is not None and memory.longitude is not None:
            # Convert decimal degrees to degrees, minutes, seconds
            def decimal_to_dms(decimal):
                degrees = int(abs(decimal))
                minutes_decimal = (abs(decimal) - degrees) * 60
                minutes = int(minutes_decimal)
                seconds = (minutes_decimal - minutes) * 60
                return (degrees, minutes, seconds)
            
            lat_dms = decimal_to_dms(memory.latitude)
            lon_dms = decimal_to_dms(memory.longitude)
            
            img.gps_latitude = lat_dms
            img.gps_latitude_ref = "N" if memory.latitude >= 0 else "S"
            img.gps_longitude = lon_dms
            img.gps_longitude_ref = "E" if memory.longitude >= 0 else "W"

        with open(image_path, "wb") as f:
            f.write(img.get_file())
    except:
        pass


async def download_memory(
    memory: Memory, output_dir: Path, add_exif: bool, semaphore: asyncio.Semaphore
) -> tuple[bool, int]:
    async with semaphore:
        try:
            cdn_url = await get_cdn_url(memory.download_link)
            ext = Path(cdn_url.split("?")[0]).suffix or ".jpg"
            output_path = output_dir / f"{memory.filename}{ext}"

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(cdn_url)
                response.raise_for_status()

                output_path.write_bytes(response.content)

                timestamp = memory.date.timestamp()
                os.utime(output_path, (timestamp, timestamp))

                if add_exif and ext == ".jpg":
                    add_exif_data(output_path, memory)

                return True, len(response.content)
        except Exception as e:
            print(f"\nError: {e}")
            return False, 0


def print_final_summary(
    stats: Stats,
    start_time: float,
    imported: int,
    rejected: int,
) -> None:
    elapsed = time.time() - start_time
    mb_total = stats.mb
    mb_per_sec = mb_total / elapsed if elapsed > 0 else 0
    print(
        f"\n{'='*50}\n"
        f"Downloaded: {stats.downloaded} ({mb_total:.1f} MB @ {mb_per_sec:.2f} MB/s) "
        f"| Skipped: {stats.skipped} | Failed: {stats.failed} "
        f"| Rejected: {rejected}"
    )
    if rejected:
        total = imported + rejected
        print(f"Accounting: {total} total = {imported} imported + {rejected} rejected")
    print("=" * 50)


async def download_all(
    memories: list[Memory],
    output_dir: Path,
    max_concurrent: int,
    add_exif: bool,
    skip_existing: bool,
    rejected_count: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = Stats()
    start_time = time.time()

    to_download = []
    for memory in memories:
        jpg_path = output_dir / f"{memory.filename}.jpg"
        mp4_path = output_dir / f"{memory.filename}.mp4"
        if skip_existing and (jpg_path.exists() or mp4_path.exists()):
            stats.skipped += 1
        else:
            to_download.append(memory)

    if not to_download:
        print("All files already downloaded!")
        print_final_summary(stats, start_time, len(memories), rejected_count)
        return

    progress_bar = tqdm(
        total=len(to_download),
        desc="Downloading",
        unit="file",
        disable=False,
    )

    async def process_and_update(memory):
        success, bytes_downloaded = await download_memory(
            memory, output_dir, add_exif, semaphore
        )
        if success:
            stats.downloaded += 1
        else:
            stats.failed += 1
        stats.mb += bytes_downloaded / 1024 / 1024

        elapsed = time.time() - start_time
        mb_per_sec = (stats.mb) / elapsed if elapsed > 0 else 0
        progress_bar.set_postfix({"MB/s": f"{mb_per_sec:.2f}"}, refresh=False)
        progress_bar.update(1)

    await asyncio.gather(*[process_and_update(m) for m in to_download])

    progress_bar.close()
    print_final_summary(stats, start_time, len(memories), rejected_count)


async def main():
    parser = argparse.ArgumentParser(
        description="Download Snapchat memories from data export"
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="json/memories_history.json",
        help="Path to memories_history.json",
    )
    parser.add_argument(
        "-o", "--output", default="./downloads", help="Output directory"
    )
    parser.add_argument(
        "-c", "--concurrent", type=int, default=40, help="Max concurrent downloads"
    )
    parser.add_argument("--no-exif", action="store_true", help="Disable EXIF metadata")
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--partial",
        "--partial-import",
        dest="partial",
        action="store_true",
        help=(
            "Partial import: download valid records and skip malformed ones. "
            "By default the command aborts before any network request when a "
            "record fails preflight."
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    output_dir = Path(args.output)

    # Fully offline preflight: happens before download_all(), which is the
    # only place where the output directory is created and requests are sent.
    try:
        result = load_memories(json_path)
    except MemImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)

    if result.rejected:
        print(format_preflight_report(result, partial=args.partial))
        if not args.partial:
            raise SystemExit(1)
        if not result.memories:
            print(
                "No valid records to import: 0 download request(s) sent and no "
                f"output directory '{output_dir}' was created."
            )
            return

    await download_all(
        result.memories,
        output_dir,
        args.concurrent,
        not args.no_exif,
        not args.no_skip_existing,
        rejected_count=len(result.rejected),
    )


if __name__ == "__main__":
    asyncio.run(main())
