# Changelog

## [Unreleased]

## [0.3.0] - 2020-02-10

### Added

-   Changelog.
-   New tests to increase code coverage.
-   New benchmarks to analyze time complexity.
-   Github Actions workflow.

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
