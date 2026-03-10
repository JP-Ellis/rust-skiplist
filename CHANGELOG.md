# Changelog

All notable changes to this project will be documented in this file.

<!-- markdownlint-disable -->

## [0.6.0] _2025-05-07_

### 🚀 Features

-   [**breaking**] Fix lints in level_generator
-   [**breaking**] Upgrade to rust edition 2024
-   [**breaking**] Refactor level generator and geometric
-   _(geometric)_ Add MaxTooLarge error

### 🐛 Bug Fixes

-   Supplement the key field of the returned result

### 🎨 Styling

-   Fix clippy warnings
-   Run `cargo fmt`
-   Re-organise `Cargo.toml` and format TOML
-   Format lib.rs

### 📚 Documentation

-   Fix GitHub Actions badge

### ⚙️ Miscellaneous Tasks

-   Replace outdated GitHub Actions
-   Pass extra args to tarpaulin
-   Fix skiplist lints
-   Delete pages submodule
-   Update copyright notice
-   Extend lints and formatting
-   _(bench)_ Fix lints
-   Adjust semicolon lint around blocks
-   Add editorconfig
-   Add pre-commit
-   Insert script for docs generation
-   Ignore clippy lints
-   Exclude slow tests from miri
-   _(ci)_ Add renovate config
-   _(ci)_ Remove cargo audit
-   _(ci)_ Brand new test workflow
-   Add nextest config
-   Silence clippy for now
-   Add copilot instructions
-   Add git-cliff configuration
-   _(ci)_ Reduce update noise

### Contributors

-   @JP-Ellis
-   @ByteAlex
-   @Aurora2500
-   @KKMaaaN

## [0.5.0] _2023-03-31_

### 🚀 Features

-   Add Bound API

### ⚙️ Miscellaneous Tasks

-   V0.5.0

### Contributors

-   @JP-Ellis
-   @KKMaaaN

## [0.4.0] _2021-06-28_

### ⚙️ Miscellaneous Tasks

-   :range() done.

### Contributors

-   @JP-Ellis
-   @bstrie
-   @jovenlin0527

## [0.3.0] _2020-02-10_

### Added

-   Changelog.
-   New tests to increase code coverage.
-   New benchmarks to analyze time complexity.
-   GitHub Actions workflow.

### Changed

-   Improved documentation in a number of areas.
-   Fix a bug in `PartialEq::ne` implementations.
-   Fix a bug in `SkipList::contains` which incorrectly stated the end value is
  not included.
-   Make use of `rustfmt`.
-   Updated dependencies.
-   Migrate to Rust Edition 2018 (thanks to chenkun).
-   Migrate benchmarks to [`criterion`](https://criterion.rs) (thanks to Siyun
  Wu).
-   Stabilize the use of `Range` (thanks to Siyun Wu).

### Removed

-   [Travis-CI](https://travis-ci.org).
