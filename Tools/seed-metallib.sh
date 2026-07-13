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

echo "compiling FlashBlock.metal..."
xcrun -sdk macosx metal -c FlashBlock.metal -o FlashBlock.air
xcrun -sdk macosx metallib FlashBlock.air -o FlashBlock.metallib
rm FlashBlock.air
echo "FlashBlock compiled successfully."
