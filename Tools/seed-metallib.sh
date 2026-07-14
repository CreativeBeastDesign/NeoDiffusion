#!/bin/bash
# Seed the mlx-swift Cmlx metallib bundle from an Xcode DerivedData build into the
# SwiftPM build products (both executable dirs and the XCTest bundle) — required on
# toolchains where `swift build` does not compile mlx-swift's .metal sources.
# See Tools/diffusion-bench/README.md "Known quirk" + Plans/m8-logbook.md timeline 13.
set -euo pipefail
cd "$(dirname "$0")/.."

D=$(ls -td ~/Library/Developer/Xcode/DerivedData/Diffusion-*/Build/Products/Debug/mlx-swift_Cmlx.bundle 2>/dev/null | head -1)
if [ -z "$D" ]; then
    echo "No mlx-swift_Cmlx.bundle in DerivedData — build the workspace once in Xcode first." >&2
    exit 1
fi
echo "seeding from: $D"

for cfg in debug release; do
    dir=".build/arm64-apple-macosx/$cfg"
    [ -d "$dir" ] || continue
    cp -R "$D" "$dir/" && echo "  -> $dir/"
    xctest="$dir/NeoDiffusionPackageTests.xctest/Contents/Resources"
    if [ -d "$dir/NeoDiffusionPackageTests.xctest" ]; then
        mkdir -p "$xctest" && cp -R "$D" "$xctest/" && echo "  -> $xctest/"
    fi
done

# NOTE (2026-07-14): the FlashBlock.metal -> .metallib compile step was removed here.
# FlashBlock's kernels are compiled at runtime by MLXFast.metalKernel from the inline
# source strings in Packages/DiffusionCore/Sources/FlashBlockRunner.swift — the root
# FlashBlock.metal was a duplicate copy that nothing loaded, and the metallib it produced
# was never opened. Compiling it only proved a *copy* compiled, which is worse than no
# check at all once the two drift. See Plans/gather_qmm_handoff.md §4.1.
