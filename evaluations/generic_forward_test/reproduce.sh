#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
REPO=$(cd ../.. && pwd)
PACKET="$REPO/releases/v0.1.0"
HASH=b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f
"$PYTHON" "$REPO/tools/freeze.py" verify "$PACKET" --expected-sha256 "$HASH" > pre_integrity.json
./build.sh > build.stdout.log 2> build.stderr.log
"$PYTHON" ./run_tests.py > final.stdout.log 2> final.stderr.log
"$PYTHON" "$REPO/tools/freeze.py" verify "$PACKET" --expected-sha256 "$HASH" > post_integrity.json
cat final.stdout.log
cat post_integrity.json
