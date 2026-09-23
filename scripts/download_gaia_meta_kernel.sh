#!/usr/bin/env bash
# Download all SPICE kernels listed in the Gaia meta-kernel into data/kernels.
# Usage: ./scripts/download_gaia_meta_kernel.sh [meta-kernel-path] [base-url]

set -euo pipefail

# Resolve the repository root as the parent of this script's directory,
# so the script works regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MK_FILE="${1:-$REPO_ROOT/data/kernels/gaia_ops_v110_20250708_001.tm}"
BASE_URL="${2:-https://spiftp.esac.esa.int/data/SPICE/GAIA/kernels/}"
DEST_DIR="$REPO_ROOT/data/kernels"

# Check if meta-kernel exists
if [[ ! -f "$MK_FILE" ]]; then
    echo "Error: Meta-kernel file not found: $MK_FILE" >&2
    exit 1
fi

# Check for curl
if ! command -v curl &>/dev/null; then
    echo "Error: curl is required but not installed." >&2
    exit 1
fi

# Extract kernel file paths from the meta-kernel
# Lines look like: '$KERNELS/ck/gaia_sc_ssm_...bc'
# We strip the '$KERNELS/' prefix and the surrounding quotes.
mapfile -t KERNEL_PATHS < <(grep -o "\$KERNELS/[^']*" "$MK_FILE" | sed "s/^\$KERNELS\///")

if [[ ${#KERNEL_PATHS[@]} -eq 0 ]]; then
    echo "Error: No kernel paths found in $MK_FILE" >&2
    exit 1
fi

echo "Found ${#KERNEL_PATHS[@]} kernel files to download."
echo "Destination: $DEST_DIR"
echo "Base URL: $BASE_URL"
echo

# Download each kernel
for rel_path in "${KERNEL_PATHS[@]}"; do
    local_path="$DEST_DIR/$rel_path"
    url="$BASE_URL/$rel_path"

    # Create local subdirectory if needed
    mkdir -p "$(dirname "$local_path")"

    # Skip if file already exists (unless FORCE=1 is set)
    if [[ -f "$local_path" && "${FORCE:-0}" -ne 1 ]]; then
        echo "Skipping (already exists): $rel_path"
        continue
    fi

    echo "Downloading: $rel_path"
    if curl -f -L --retry 3 --retry-delay 2 -o "$local_path" "$url"; then
        echo "  -> OK"
    else
        echo "  -> FAILED" >&2
        # Remove partial file
        rm -f "$local_path"
    fi
done

echo
echo "Download complete."