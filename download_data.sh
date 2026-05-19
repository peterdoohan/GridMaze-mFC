#!/usr/bin/env bash
# download_data.sh — fetch GridMaze-mFC ephys data and results from Zenodo.
# See README.md "Option 3" for usage. Run with --help for a flag summary.

set -euo pipefail

ZENODO_RECORD="20267467"
ZENODO_BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"
ZENODO_API="https://zenodo.org/api/records/${ZENODO_RECORD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WANT_RESULTS=true
WANT_LFP=true
DATA_DIR="${PARENT_DIR}/data"
RESULTS_DIR="${PARENT_DIR}/results"
VERIFY=true
KEEP_ZIP=false

API_JSON=""

usage() {
    cat <<EOF
Usage: bash download_data.sh [OPTIONS]

Download data.zip (and optionally results.zip) from Zenodo record ${ZENODO_RECORD}
and extract into sibling folders of this repo. Resumable on network drop.

Options:
  --no-results          Skip results.zip (data.zip only, ~73 GB on disk).
  --no-lfp              Skip extracting LFP files (lfp.signal.npy, lfp.times.npy,
                        lfp.metrics.htsv). Saves disk space, NOT bandwidth — the
                        bytes still download since LFP is bundled inside data.zip.
  --data-dir <path>     Destination for data/    (default: ${DATA_DIR})
  --results-dir <path>  Destination for results/ (default: ${RESULTS_DIR})
  --no-verify           Skip MD5 verification against Zenodo's API.
  --keep-zip            Don't delete downloaded zip(s) after extraction.
  -h, --help            Show this message.

Defaults: download both zips (~104 GB on disk), verify MD5s, delete zips after extract.
The download is resumable — if interrupted, just re-run the script.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-results)    WANT_RESULTS=false; shift ;;
            --no-lfp)        WANT_LFP=false; shift ;;
            --data-dir)      DATA_DIR="${2%/}"; shift 2 ;;
            --results-dir)   RESULTS_DIR="${2%/}"; shift 2 ;;
            --no-verify)     VERIFY=false; shift ;;
            --keep-zip)      KEEP_ZIP=true; shift ;;
            -h|--help)       usage; exit 0 ;;
            *)               echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        esac
    done
}

require_tools() {
    local missing=()
    for tool in curl unzip python3; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    if ! command -v md5sum >/dev/null 2>&1 && ! command -v md5 >/dev/null 2>&1; then
        missing+=("md5sum or md5")
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Error: missing required tool(s): ${missing[*]}" >&2
        exit 1
    fi
}

# Portable MD5: Linux uses md5sum, macOS uses md5 -q.
md5_of() {
    local file="$1"
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$file" | awk '{print $1}'
    else
        md5 -q "$file"
    fi
}

fetch_api_json() {
    API_JSON="$(mktemp)"
    trap 'rm -f "$API_JSON"' EXIT
    echo "→ Fetching record metadata from Zenodo API..."
    curl -fsSL "$ZENODO_API" -o "$API_JSON"
}

md5_for_file() {
    # Look up the MD5 for a given filename in the cached Zenodo API JSON.
    local filename="$1"
    FILENAME="$filename" API_JSON_PATH="$API_JSON" python3 - <<'PY'
import json, os, sys
with open(os.environ["API_JSON_PATH"]) as fp:
    record = json.load(fp)
target = os.environ["FILENAME"]
for f in record["files"]:
    if f["key"] == target:
        algo, digest = f["checksum"].split(":", 1)
        if algo != "md5":
            sys.stderr.write(f"Unexpected checksum algo: {algo}\n")
            sys.exit(1)
        print(digest)
        sys.exit(0)
sys.stderr.write(f"No file named {target} in Zenodo record\n")
sys.exit(1)
PY
}

download_and_extract() {
    # Args: zip_name, dest_dir, [unzip_exclude_patterns...]
    local zip_name="$1"; shift
    local dest_dir="${1%/}"; shift
    local exclude_args=("$@")

    local url="${ZENODO_BASE}/${zip_name}"
    local tmpzip="${dest_dir}.partial.zip"

    mkdir -p "$(dirname "$dest_dir")"
    mkdir -p "$dest_dir"

    echo
    echo "→ Downloading ${zip_name}"
    echo "  from: ${url}"
    echo "  to:   ${tmpzip}"
    echo "  (resumable — re-run the script if interrupted)"
    curl -L -C - --fail --progress-bar -o "$tmpzip" "$url"

    if [[ "$VERIFY" == true ]]; then
        local expected
        expected="$(md5_for_file "$zip_name")"
        echo "→ Verifying MD5 of ${zip_name} (this can take a few minutes on large files)"
        echo "  expected: ${expected}"
        local actual
        actual="$(md5_of "$tmpzip")"
        if [[ "$actual" != "$expected" ]]; then
            echo "Error: MD5 mismatch for ${zip_name}" >&2
            echo "  expected: $expected" >&2
            echo "  actual:   $actual" >&2
            echo "  Re-run the script to resume the download. If the mismatch persists," >&2
            echo "  delete ${tmpzip} and start fresh." >&2
            exit 1
        fi
        echo "  ✓ MD5 ok"
    fi

    echo "→ Extracting ${zip_name} to ${dest_dir}"
    if [[ ${#exclude_args[@]} -gt 0 ]]; then
        echo "  excluding patterns: ${exclude_args[*]}"
        unzip -q -o "$tmpzip" -d "$dest_dir" -x "${exclude_args[@]}"
    else
        unzip -q -o "$tmpzip" -d "$dest_dir"
    fi

    if [[ "$KEEP_ZIP" == false ]]; then
        rm "$tmpzip"
        echo "  removed ${tmpzip}"
    else
        echo "  kept ${tmpzip} (--keep-zip)"
    fi
}

main() {
    parse_args "$@"
    require_tools

    echo "GridMaze-mFC ephys data downloader"
    echo "  Zenodo record: ${ZENODO_RECORD}"
    echo "  data dir:      ${DATA_DIR}"
    if [[ "$WANT_RESULTS" == true ]]; then
        echo "  results dir:   ${RESULTS_DIR}"
    else
        echo "  results:       skipped (--no-results)"
    fi
    if [[ "$WANT_LFP" == false ]]; then
        echo "  LFP files:     skipped at extract (--no-lfp; bytes still download)"
    fi
    if [[ "$VERIFY" == false ]]; then
        echo "  checksums:     skipped (--no-verify)"
    fi
    if [[ "$KEEP_ZIP" == true ]]; then
        echo "  zip cleanup:   skipped (--keep-zip)"
    fi

    if [[ "$VERIFY" == true ]]; then
        fetch_api_json
    fi

    local lfp_excludes=()
    if [[ "$WANT_LFP" == false ]]; then
        lfp_excludes=("*/lfp.signal.npy" "*/lfp.times.npy" "*/lfp.metrics.htsv")
    fi

    if [[ ${#lfp_excludes[@]} -gt 0 ]]; then
        download_and_extract "data.zip" "$DATA_DIR" "${lfp_excludes[@]}"
    else
        download_and_extract "data.zip" "$DATA_DIR"
    fi

    if [[ "$WANT_RESULTS" == true ]]; then
        download_and_extract "results.zip" "$RESULTS_DIR"
    fi

    echo
    echo "✓ Done."
    echo "  data:    ${DATA_DIR}"
    if [[ "$WANT_RESULTS" == true ]]; then
        echo "  results: ${RESULTS_DIR}"
    fi
}

main "$@"
