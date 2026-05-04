# 1. Create archive directory
mkdir -p archive/runs_before_26192

# 2. Add archive to .gitignore
grep -qxF "archive/" .gitignore 2>/dev/null || echo "archive/" >> .gitignore

# 3. Preview files that will be moved
for f in *; do
    [ -f "$f" ] || continue

    job_id=$(echo "$f" | grep -oE '(^|_)[0-9]{5}(\.|_|$)' | grep -oE '[0-9]{5}' | tail -n 1)

    if [ -n "$job_id" ] && [ "$job_id" -lt 26192 ]; then
        echo "$f"
    fi
done
