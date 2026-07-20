#!/usr/bin/env bash
# sync_vllm_neuron_docs.sh — Copy vLLM Neuron docs from private-vllm-neuron
# into the staging repo as plain files.
#
# Usage:
#   ./_utilities/sync_vllm_neuron_docs.sh [branch] [source_repo_path]
#
# Arguments:
#   branch           Branch in private-vllm-neuron to sync from (default: neuron-docs)
#   source_repo_path Path to local private-vllm-neuron clone (default: ../private-vllm-neuron)
#
# What it does:
#   1. Fetches the latest from remote and checks out the specified branch in the local clone
#   2. Removes vllm-neuron/docs/ in the staging repo (clean slate)
#   3. Copies docs/ from the local clone into vllm-neuron/docs/
#   4. Does NOT touch files outside vllm-neuron/docs/ (e.g., neuron-inference-overview.rst)
#   5. Prints a summary of what changed
#
# After running:
#   - Review with: git diff --stat
#   - Build locally: make html
#   - Commit: git add vllm-neuron/docs/ && git commit -m "docs: sync vLLM Neuron docs from private-vllm-neuron"
#
# Full workflow for pulling new vLLM Neuron docs content:
#   Prerequisites (one-time): PR #3133 must be merged first — it adds the sidebar
#   navigation, overview page, and toctree entries that reference vllm-neuron/docs/.
#
#   1. Ensure private-vllm-neuron is cloned next to this repo (../private-vllm-neuron)
#   2. Run this script:  ./_utilities/sync_vllm_neuron_docs.sh neuron-docs
#      (or "neuron-staging" once that becomes the primary docs branch)
#   3. Review changes:   git diff --stat
#   4. Build locally:    make html
#   5. Verify output:    open _build/html/vllm-neuron/docs/index.html
#   6. Stage & commit:   git add vllm-neuron/docs/ && git commit -m "docs: sync vLLM Neuron docs from private-vllm-neuron (branch@commit)"
#   7. Push & PR:        push branch, open PR to release-X.Y.Z

set -euo pipefail

BRANCH="${1:-neuron-docs}"
SOURCE_REPO="${2:-$(cd "$(dirname "$0")/.." && cd ../private-vllm-neuron && pwd)}"
STAGING_REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${STAGING_REPO}/vllm-neuron/docs"

echo "=== sync_vllm_neuron_docs ==="
echo "  Source repo : ${SOURCE_REPO}"
echo "  Branch      : ${BRANCH}"
echo "  Destination : ${DEST_DIR}"
echo ""

# Validate source repo exists
if [ ! -d "${SOURCE_REPO}/.git" ]; then
    echo "ERROR: Source repo not found at ${SOURCE_REPO}" >&2
    echo "  Provide path as second argument or clone private-vllm-neuron next to this repo." >&2
    exit 1
fi

# Validate source has a docs/ folder
if [ ! -d "${SOURCE_REPO}/docs" ]; then
    echo "ERROR: No docs/ folder found in ${SOURCE_REPO}" >&2
    exit 1
fi

# Fetch and checkout the branch in source repo
echo ">> Fetching and checking out '${BRANCH}' in source repo..."
(cd "${SOURCE_REPO}" && git fetch origin "${BRANCH}" && git checkout "${BRANCH}" && git pull origin "${BRANCH}")
echo ""

# Get the commit we're syncing from
SOURCE_COMMIT=$(cd "${SOURCE_REPO}" && git rev-parse --short HEAD)
echo ">> Source commit: ${SOURCE_COMMIT} ($(cd "${SOURCE_REPO}" && git log -1 --format='%s' HEAD))"
echo ""

# Remove old docs (clean slate)
if [ -d "${DEST_DIR}" ]; then
    echo ">> Removing existing ${DEST_DIR}..."
    rm -rf "${DEST_DIR}"
fi

# Copy fresh docs
echo ">> Copying docs/ from source..."
mkdir -p "${DEST_DIR}"
cp -R "${SOURCE_REPO}/docs/." "${DEST_DIR}/"

# Count files copied
FILE_COUNT=$(find "${DEST_DIR}" -type f | wc -l | tr -d ' ')
echo ""
echo "=== Done ==="
echo "  Copied ${FILE_COUNT} files from private-vllm-neuron:${BRANCH} (${SOURCE_COMMIT})"
echo "  Destination: ${DEST_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review:  git diff --stat"
echo "  2. Build:   make html"
echo "  3. Commit:  git add vllm-neuron/docs/ && git commit -m \"docs: sync vLLM Neuron docs from private-vllm-neuron (${BRANCH}@${SOURCE_COMMIT})\""
