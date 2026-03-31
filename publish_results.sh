#!/bin/bash

set -uo pipefail

SCRIPT_NAME="$(basename "$0")"
RESULTS_DIR_REL="experiments/results"
DVC_FILE_REL="experiments/results.dvc"
BACKUP_DIR=""
PUBLISH_SUCCEEDED=0

log() {
    printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
    printf '[%s] Error: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 1
}

print_permission_hint() {
    local step="$1"
    local command_name="$2"

    case "$command_name" in
        git)
            cat >&2 <<EOF_HINT
[$SCRIPT_NAME] Hint: Git reported "Permission denied". Check that:
[$SCRIPT_NAME]   - your remote credentials or SSH keys are configured on this machine
[$SCRIPT_NAME]   - you still have push/pull access to the repository remote
EOF_HINT
            ;;
        uv|dvc)
            cat >&2 <<EOF_HINT
[$SCRIPT_NAME] Hint: DVC reported "Permission denied" during "$step". Check that:
[$SCRIPT_NAME]   - this machine has valid DVC remote credentials configured
[$SCRIPT_NAME]   - the configured object store allows read/write access
[$SCRIPT_NAME]   - files under .dvc/cache and experiments/results are owned by your user
EOF_HINT
            ;;
        rsync)
            cat >&2 <<EOF_HINT
[$SCRIPT_NAME] Hint: rsync reported "Permission denied". Check that:
[$SCRIPT_NAME]   - files inside $RESULTS_DIR_REL are readable by your user
[$SCRIPT_NAME]   - the destination directory is writable by your user
EOF_HINT
            ;;
    esac
}

run_step() {
    local step="$1"
    shift

    local output_file
    local status
    output_file="$(mktemp "${TMPDIR:-/tmp}/publish_results_cmd.XXXXXX")"

    log "$step"

    if "$@" >"$output_file" 2>&1; then
        if [ -s "$output_file" ]; then
            cat "$output_file"
        fi
        rm -f "$output_file"
        return 0
    fi

    status=$?
    if [ -s "$output_file" ]; then
        cat "$output_file" >&2
    fi

    printf '[%s] Error: step failed: %s\n' "$SCRIPT_NAME" "$step" >&2
    printf '[%s] Error: command: ' "$SCRIPT_NAME" >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2

    if grep -qi "permission denied" "$output_file"; then
        print_permission_hint "$step" "${1##*/}"
    fi

    rm -f "$output_file"
    exit "$status"
}

usage() {
    cat <<EOF_USAGE
Usage: ./$SCRIPT_NAME [commit message]

Publish the current experiments/results snapshot by:
  1. backing up local experiments/results/
  2. pulling the latest Git pointer + DVC snapshot
  3. merging the backed-up local files over the pulled tree
  4. re-staging the whole directory with DVC
  5. pushing DVC data, committing experiments/results.dvc, and pushing Git

If a commit message is omitted, a timestamped default is used.
EOF_USAGE
}

cleanup() {
    local exit_code=$?

    if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
        exit "$exit_code"
    fi

    if [ "$exit_code" -eq 0 ] && [ "$PUBLISH_SUCCEEDED" -eq 1 ]; then
        rm -rf "$BACKUP_DIR"
    else
        log "Publish did not complete. Local backup preserved at: $BACKUP_DIR/local_results"
    fi

    exit "$exit_code"
}

trap cleanup EXIT

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

COMMIT_MESSAGE="${*:-}"

command -v git >/dev/null 2>&1 || die "git is required but not installed."
command -v uv >/dev/null 2>&1 || die "uv is required but not installed."
command -v rsync >/dev/null 2>&1 || die "rsync is required but not installed."

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || die "Run this script from inside the Git repository."
cd "$REPO_ROOT"

git ls-files --error-unmatch "$DVC_FILE_REL" >/dev/null 2>&1 || die "$DVC_FILE_REL is not tracked by Git."
[ -d "$RESULTS_DIR_REL" ] || die "$RESULTS_DIR_REL does not exist. Run an experiment first."

CURRENT_BRANCH="$(git branch --show-current)"
[ -n "$CURRENT_BRANCH" ] || die "Detached HEAD is not supported. Check out a branch first."

UPSTREAM_REF="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
[ -n "$UPSTREAM_REF" ] || die "Current branch '$CURRENT_BRANCH' has no upstream branch configured."

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Tracked-file changes detected. Commit or stash them before publishing so git pull stays safe."
fi

BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/publish_results.XXXXXX")"
mkdir -p "$BACKUP_DIR/local_results"

run_step "Backing up local $RESULTS_DIR_REL to $BACKUP_DIR/local_results" \
    rsync -a "$RESULTS_DIR_REL/" "$BACKUP_DIR/local_results/"

run_step "Pulling latest Git commits from $UPSTREAM_REF" \
    git pull --rebase

run_step "Pulling latest published $RESULTS_DIR_REL snapshot from DVC" \
    uv run dvc pull "$DVC_FILE_REL"

mkdir -p "$RESULTS_DIR_REL"
run_step "Merging backed-up local files over pulled snapshot (local files win on overlap)" \
    rsync -a "$BACKUP_DIR/local_results/" "$RESULTS_DIR_REL/"

run_step "Re-staging merged $RESULTS_DIR_REL with DVC" \
    uv run dvc add "$RESULTS_DIR_REL"

run_step "Pushing updated DVC data" \
    uv run dvc push "$DVC_FILE_REL"

if git diff --quiet -- "$DVC_FILE_REL"; then
    log "No change detected in $DVC_FILE_REL after merge; nothing to commit."
    PUBLISH_SUCCEEDED=1
    log "Done. Collaborators already have the latest published snapshot."
    exit 0
fi

if [ -z "$COMMIT_MESSAGE" ]; then
    COMMIT_MESSAGE="Publish experiments/results snapshot ($(date '+%Y-%m-%d %H:%M:%S %Z'))"
fi

run_step "Staging updated DVC pointer" \
    git add "$DVC_FILE_REL"
run_step "Committing updated DVC pointer" \
    git commit -m "$COMMIT_MESSAGE" -- "$DVC_FILE_REL"

run_step "Pushing Git commit to origin" \
    git push

PUBLISH_SUCCEEDED=1
log "Publish complete. Teammates can now pull the new pointer with Git and fetch data with DVC."
