# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- markdownlint-disable -->

## [2.0.1](https://github.com/JP-Ellis/rust-skiplist/compare/v2.0.0...v2.0.1) - _2026-10-28_

### 🐛 Bug Fixes

-   _(node)_ Import Box in release builds

### 🚜 Refactor

-   Keep the concat bookkeeping in one place

-   Share the Equal-as-Less comparison

-   Share the appending extend loop

-   _(node)_ Drop the visitor step method

### ⚡ Performance

-   _(node)_ Lay out node fields hot-first

-   Grow the frontier only for taller towers

-   Trim the head's empty levels on insert

-   Keep the head as tall as the tallest tower

-   Split and append in O(log n)

-   Skip the value test on walk comparisons

-   Read removal precursors from the visitor

-   Keep insert precursors in a caller-owned path

-   Build ordered containers by appending

-   _(set)_ Keep the found node in OccupiedEntry

-   _(list)_ Reach the first and last index directly

-   _(list)_ Clone by chain build and rebuild

-   Pop owning iterators along the base chain

-   _(list)_ Skip the walk for towerless tails

-   _(osl)_ Count long runs of duplicates by rank

-   _(set)_ Replace in a single walk

-   _(map)_ Look up get_mut without precursors

-   Compute range lengths from ranks

-   _(osl)_ Start the take_last scan near the target

-   Find tail precursors in O(log n)

-   _(node)_ Keep the index walk state in locals

-   _(node)_ Keep the ordered walk state in locals

-   _(map)_ Draw the insert level only when needed

-   _(node)_ Use wrapping adds for rank sums

-   _(node)_ Skip the bounds check in LevelArray

### 🧪 Testing

-   Trim slow tests under Miri

-   _(bench)_ Add opt-in large-payload workloads

-   _(bench)_ Add split_off and append workloads

-   _(node)_ Check traverse against the step loop

-   Add hot workloads for the remaining items

-   Add focused hot-path benchmark suite

### ⚙️ Miscellaneous Tasks

-   Add cargo-insta to the mise tools

-   _(ci)_ Build and test with debug assertions off


## [2.0.0](https://github.com/JP-Ellis/rust-skiplist/compare/v1.3.4...v2.0.0) - _2026-10-24_

### ⚠️ Breaking Changes

#### `extract_if` takes a range on `OrderedSkipList` and `SkipMap`

`OrderedSkipList::extract_if` and `SkipMap::extract_if` now take a range as their first argument, matching `SkipSet::extract_if`. The predicate sees only elements (or keys) inside the range, and iteration stops at the upper bound.

Calls written as `extract_if(pred)` no longer compile. Pass `..` to keep the 1.x behaviour of visiting every element:

```rust
// 1.x
let evens: Vec<_> = list.extract_if(|x| *x % 2 == 0).collect();

// 2.0
let evens: Vec<_> = list.extract_if(.., |x| *x % 2 == 0).collect();
```

A narrower range skips the elements outside it without calling the predicate on them. Code that names the `ExtractIf` iterator type must add the range type `R`, which sits between `G` and `F`.

#### `std` is now a default feature

The crate is now `no_std` and requires `alloc`. The new `std` feature is on by default and seeds level generators from `rand::rng()`. Projects that use the default features see no change.

Projects that set `default-features = false` lose every constructor that seeds a `Geometric` from entropy:

-   `Geometric::new` and `impl Default for Geometric`
-   `new`, `with_capacity` and `with_comparator` on the collections
-   `Default`, `FromIterator` and `From<Vec<T>>` / `From<[T; M]>` for the default generator
-   The `SkipSet` operators `|`, `&`, `-` and `^`

Pick one of the following to migrate:

-   Add `features = ["std"]` to restore the 1.x behaviour.
-   Add `features = ["getrandom"]` to seed from the operating system without `std`. In this configuration `Geometric::new` returns `GeometricError::Entropy` when the operating system source fails, and `new()` and `Default` panic. On `wasm32-unknown-unknown`, also enable the `wasm_js` feature of `getrandom` in your own `Cargo.toml`.
-   Seed explicitly with `Geometric::new_with_seed` and pass the generator to `with_level_generator` or `with_comparator_and_level_generator`. `SkipList::with_capacity_and_seed` replaces `SkipList::with_capacity`.

The [`no_std` guide](https://docs.rs/skiplist/2.0.0/skiplist/docs/no_std/index.html) lists the replacement for each constructor.

#### `Geometric` caps `total` at 64

`Geometric::new` and `Geometric::new_with_seed` reject a `total` above `Geometric::MAX_TOTAL` (64) with `GeometricError::MaxTooLarge`. In 1.x the limit was `i32::MAX`. At `q = 0.5`, 64 levels already serve about 2^64 elements, so lowering a larger `total` to 64 loses nothing. `SkipList::with_capacity` clamps its derived level count to 64 and never hits this error.

The `Display` text of `MaxTooLarge` now reads `max must be at most 64.`

### 🚀 Features

-   _(skip_list)_ Add with_capacity_and_seed

-   [**breaking**] Support no_std with alloc

-   _(geometric)_ [**breaking**] Sample from a CDF table

-   _(ordered_skip_list)_ [**breaking**] Take a range in extract_if

-   _(skip_map)_ [**breaking**] Take a range in extract_if

### 🐛 Bug Fixes

-   _(skip_list)_ Document and restore ctor links

-   _(geometric)_ Validate before seeding

### 🚜 Refactor

-   Prefer core and alloc over std

-   _(geometric)_ Store CDF in Sampler::Table

### 📚 Documentation

-   Link gated items to the features section

-   Restore gated constructor links

-   Correct std feature and level clamp

-   Document no_std support

### 🧪 Testing

-   Exercise constructors without entropy

### ⚙️ Miscellaneous Tasks

-   _(ci)_ Check doc links in every feature set

-   _(ci)_ Group additive features in powerset

-   _(ci)_ Lint every library feature set

-   _(ci)_ Run doc-page examples

-   _(ci)_ Build no_std with all features

-   _(ci)_ Build for no_std targets


## [1.3.4](https://github.com/JP-Ellis/rust-skiplist/compare/v1.3.3...v1.3.4) - _2026-10-24_

### 🚜 Refactor

-   _(geometric)_ Guard with bool::ok_or

### 🧪 Testing

-   Shrink chaos runs under Miri

### ⚙️ Miscellaneous Tasks

-   Raise MSRV to 1.98


## [1.3.3](https://github.com/JP-Ellis/rust-skiplist/compare/v1.3.2...v1.3.3) - _2026-10-24_

### ⚡ Performance

-   _(level_generator)_ Bit sampler at q = 1/2

### 🧪 Testing

-   Benchmark Geometric::level across q

### ⚙️ Miscellaneous Tasks

-   Raise MSRV to 1.97


## [1.3.2](https://github.com/JP-Ellis/rust-skiplist/compare/v1.3.1...v1.3.2) - _2026-10-24_

### 📚 Documentation

-   Show core::range types in range doctests

### ⚙️ Miscellaneous Tasks

-   Raise MSRV to 1.96


## [1.3.1](https://github.com/JP-Ellis/rust-skiplist/compare/v1.3.0...v1.3.1) - _2026-10-24_

### 🚜 Refactor

-   _(cursor)_ Use as_mut_unchecked for list ptr

### ⚙️ Miscellaneous Tasks

-   Raise MSRV to 1.95


## [1.3.0](https://github.com/JP-Ellis/rust-skiplist/compare/v1.2.0...v1.3.0) - _2026-10-24_

### 🚀 Features

-   _(skip_list)_ Add insert_mut

-   _(skip_list)_ Add push_{front,back}_mut

-   _(skip_map)_ Add comparator accessor

-   _(skip_list)_ Add pop_front_if and pop_back_if

### 🐛 Bug Fixes

-   Stop IterMut::next_back reading yielded node

-   Keep range_mut back pointer provenance

-   Derive cursor list pointer after search

-   Keep head pointer provenance in splice

-   Derive take_fast precursors structurally

-   Panic when a set entry loses its element

-   Validate precursors before splicing

-   Panic on comparator contradiction

-   _(node)_ Gate fmt::Write import on debug builds

### 🚜 Refactor

-   Take a NonNull in Node::join

-   Drop arrayvec dependency

-   _(node)_ Store links in a LevelArray

-   _(node)_ Add LevelArray

### 📚 Documentation

-   Document comparator panics per method

-   State comparator contract guarantees

-   Note comparator trust in append fast path

-   _(skip_map)_ Drop the summary link to a missing accessor

### 🧪 Testing

-   Silence contract panics in chaos runs

-   Check node height in well-formed checker

-   Seed generators in chaos merge ops

-   Enable inconsistent comparator suites

-   Cover cursor and split paths under chaos

-   Seed chaos generators and report failures

-   Reproduce UB under inconsistent comparators

-   Add structural well-formedness checker

-   Use assert_matches! in unit tests

-   Shrink large-list tests under Miri

### ⚙️ Miscellaneous Tasks

-   _(ci)_ Shard Miri by data structure

-   _(ci)_ Save caches only from main

-   _(renovate)_ Rebase only on conflict

-   _(mise)_ Drop the lockfile

-   _(mise)_ Prefer aqua and github backends

-   Allow 80-char subjects and 100-char bodies

-   _(renovate)_ Group the toolchain pins by dependency name


## [1.2.0](https://github.com/JP-Ellis/rust-skiplist/compare/v1.1.1...v1.2.0) - _2026-09-18_

### 🚀 Features

-   Add unchecked cursor inserts

-   Add cursor API for ordered types

### 🚜 Refactor

-   Make cursor modules private

### ⚙️ Miscellaneous Tasks

-   Add codecov config with tolerance


## [1.1.1](https://github.com/JP-Ellis/rust-skiplist/compare/v1.1.0...v1.1.1) - _2026-09-03_

### 🐛 Bug Fixes

-   Preserve container invariants when user code unwinds

### 🚜 Refactor

-   Satisfy new clippy restriction lints

### 🎨 Styling

-   Apply rumdl formatting

-   Tombi format

### ⚙️ Miscellaneous Tasks

-   Run all gates through mise tasks

-   Tune clippy lints and pin toml version

-   Add biome config using editorconfig

-   Migrate markdown lint config to rumdl

-   Adopt mise for toolchain and tooling

-   _(ci)_ Update step name in docs test


## [1.1.0](https://github.com/JP-Ellis/rust-skiplist/compare/v1.0.0...v1.1.0) - _2026-03-14_

### 🚀 Features

-   Add seeded constructor

### 🐛 Bug Fixes

-   Unable to create upper bound
-   Two off-by-one errors

### 📚 Documentation

-   Fix p/q confusion

### 🧪 Testing

-   Add structural snapshots

### ⚙️ Miscellaneous Tasks

-   Fold cliff config into release-plz
-   Remove output from cliff config
-   _(ci)_ Test doc examples
-   _(ci)_ Run tests on ready for review


## [1.0.0] _2026-03-10_

> [!IMPORTANT]
>
> This release shares little code with the 0.x series. The internal architecture, public API, and crate structure have all changed. If you are upgrading from 0.x, treat this as a new dependency and review the documentation from scratch rather than diffing against the old API.

Version 1.0.0 introduces four skip-list collections, each targeting a familiar stdlib analogue:

- **`SkipList<T>`**: insertion-order positional sequence analogous to `Vec`/`VecDeque`, offering O(log n) insert, remove, and indexed access at any position without element shifting.
- **`OrderedSkipList<T, C>`**: always-sorted bag that, unlike `BTreeSet`, keeps duplicate values adjacent in the list.
- **`SkipSet<T, C>`**: mirrors `BTreeSet` with sorted, unique values and set-algebra operations.
- **`SkipMap<K, V, C>`**: mirrors `BTreeMap` with sorted unique key-value pairs.

See [docs.rs](https://docs.rs/skiplist) for the full API.

All three ordered collections are parameterised by `C: Comparator<T>` rather than requiring `T: Ord` on the struct itself.

- **`OrdComparator`**: zero-sized, delegates to `T: Ord`; the default for all three types.
- **`FnComparator<F>`**: wraps any `Fn(&T, &T) -> Ordering` closure; no newtype needed for custom orderings.
- **`PartialOrdComparator`**: behind the `partial-ord` feature flag; panics if a comparison returns `None` (e.g. NaN). For float keys, `FnComparator(f64::total_cmp)` is the safer alternative.

The node layer has been rewritten to use `NonNull<Node<T, N>>` throughout, replacing the raw `*mut` pointers of the `0.x` series. This aligns the unsafe internals with Rust's pointer-provenance model, allows the full test suite to pass under Miri with Tree Borrows, and substantially reduces the volume of unsafe code.

<!-- generated by git-cliff on 2026-03-10-->

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
