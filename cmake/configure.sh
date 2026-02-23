#!/usr/bin/env bash
set -e

STEP="$1"
THREADS="${2:-4}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$PROJECT_ROOT/bin"

# Detect Python automatically
PYTHON_EXECUTABLE=$(which python3)
echo "[INFO] Using Python interpreter: $PYTHON_EXECUTABLE"

# --------------------------
# Clean
# --------------------------
if [ -z "$STEP" ] || [ "$STEP" = "clean" ]; then
    echo "[CLEAN] Removing build/ and bin/ directories..."
    rm -rf "$BUILD_DIR" "$BIN_DIR"
    [ "$STEP" = "clean" ] && exit 0
fi

# --------------------------
# Build
# --------------------------
if [ -z "$STEP" ] || [ "$STEP" = "build" ]; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "[CMAKE] Configuring project..."
    cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" "$PROJECT_ROOT"

    echo "[CMAKE] Building with $THREADS threads..."
    cmake --build . --parallel "$THREADS"

    echo "[DONE] Build complete."
    exit 0
fi

echo "Usage: $0 [clean|build] [threads]"
exit 1
