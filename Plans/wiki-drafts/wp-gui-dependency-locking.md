# Wiki Draft: GUI Toolchain and Transitive Dependency Locking

## Overview
During the implementation of the NeoDiffusion GUI, a critical toolchain compatibility issue arose in Xcode targets involving downstream package updates. This document records the issue, analysis, and solution.

## 1. Problem Description (Sourced)
* **Sourced Fact**: `xcodebuild` tasks (`task-220`, `task-260`) failed with compilation errors in the `Hub` target of the `swift-transformers` package checkouts:
  ```
  Config.swift:456:29: error: initializer 'init(uniqueKeysWithValues:)' requires the types 'Dictionary<String, Value>.Element' and '(ObjectKey, Value)' be equivalent
  Config.swift:466:21: error: cannot convert value of type 'String' to expected dictionary key type 'ObjectKey'
  ```
* **Sourced Fact**: The same package built successfully on the command line via `swift build` in package isolation mode (`task-256`, `task-270`).

## 2. Technical Analysis (Inferred)
* **Inferred Finding**: The error is caused by a breaking change in the `swift-jinja` library (version 2.4.0) changing the signature of `Jinja.Value.object`'s payload dictionary from `String` to a custom wrapper type `ObjectKey`.
* **Inferred Finding**: While `swift build` was constrained by package manifests to resolve `swift-jinja` at version `2.3.6` (which uses normal `String` keys), `xcodebuild` resolved `swift-jinja` independently as a transitive dependency of the remote package `swift-transformers` defined in `project.pbxproj`'s `XCRemoteSwiftPackageReference` list. This bypassed root `Package.swift` rules, resolving `swift-jinja` to the latest version `2.4.0` and breaking compilation.

## 3. Resolution (Sourced)
* **Sourced Fact**: Added `swift-jinja` (exact version `2.3.6`) as a direct dependency in the root `Package.swift`.
* **Sourced Fact**: Added a matching `XCRemoteSwiftPackageReference` configuration inside `project.pbxproj` for `swift-jinja` at version `exact: 2.3.6`:
  ```pbxproj
  CCF271092EABCBD200E2550B /* XCRemoteSwiftPackageReference "swift-jinja" */ = {
      isa = XCRemoteSwiftPackageReference;
      repositoryURL = "https://github.com/huggingface/swift-jinja.git";
      requirement = {
          kind = exactVersion;
          version = 2.3.6;
      };
  };
  ```
* **Result**: `xcodebuild` successfully resolved and downloaded `swift-jinja` 2.3.6, bypassing the API conflict and restoring compiling status (`task-356` **BUILD SUCCEEDED**).
