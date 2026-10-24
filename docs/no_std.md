# Using `skiplist` Without `std`

This page explains how the crate builds without the standard library, which constructors each configuration provides, and how to seed the level generator when no operating system supplies entropy.

The crate is `#![no_std]` and requires `alloc`. Every collection allocates its nodes on the heap.

## Feature Matrix

| Configuration | Cargo features | Entropy source |
|---------------|----------------|----------------|
| `std` (default) | `default-features = true` | `rand::rng()` (thread-local RNG) |
| `getrandom` | `default-features = false, features = ["getrandom"]` | `rand::rngs::SysRng` (the operating system, through the `getrandom` crate) |
| neither | `default-features = false` | none |

The `std` feature enables `rand/thread_rng`. The `getrandom` feature enables `rand/sys_rng`. With both features on, `std` wins and seeding goes through `rand::rng()`.

```toml
# std (default)
[dependencies]
skiplist = "1"

# no_std with operating-system entropy
[dependencies]
skiplist = { version = "1", default-features = false, features = ["getrandom"] }

# no_std without entropy
[dependencies]
skiplist = { version = "1", default-features = false }
```

## What Each Configuration Provides

Without `std` or `getrandom`, there is no entropy source, so `Geometric` has no way to pick a seed. The API that needs one is absent. The table below lists each item that is absent and its replacement.

| Removed without entropy | Replacement |
|-------------------------|-------------|
| `new()` | `with_level_generator` |
| `with_comparator` | `with_comparator_and_level_generator` |
| `From<Vec<T>>`, `From<[T; M]>`, `FromIterator` | `with_level_generator` + `Extend` |
| `SkipSet` `\|`, `&`, `-`, `^` | `union` / `intersection` / `difference` / `symmetric_difference` iterators + `Extend` |
| `SkipList::with_capacity` | `SkipList::with_capacity_and_seed(capacity, seed)` |

The same gate removes `Geometric::new` and `impl Default for Geometric`. `Default`, `FromIterator`, `From<Vec<T>>`, `From<[T; M]>` and the `SkipSet` operators need `G: Default`. They are absent for `G = Geometric` without entropy and stay available for any custom generator that implements `Default`.

These items are always available, in every configuration:

-   `Geometric::new_with_seed(total, q, seed)`
-   `with_level_generator` and `with_comparator_and_level_generator`
-   `Extend` on all four collections
-   `SkipList::with_capacity_and_seed(capacity, seed)`

With `getrandom` and without `std`, entropy can fail. `Geometric::new` returns `GeometricError::Entropy` in that case. `Default` and the collection constructors such as `new()` panic.

## Seeding

Without entropy, pass a seed yourself or enable `getrandom`.

-   A fixed seed gives a reproducible list shape. That suits tests and deterministic builds.
-   `getrandom` draws a fresh seed from the operating system on each construction.

The crate has no fallback seed. A seed derived from an address, a clock or a constant would make `Default` appear to work while hiding the problem. On bare metal there is no ASLR, so an address-derived seed is identical on every boot. A predictable seed also lets an attacker who controls the order of inserts and removes find which nodes have tall towers. Removing those nodes pushes the operations from `$O(\log n)$` towards `$O(n)$`. Without an entropy source, `Default` does not exist.

If your target has a hardware random number generator, read a seed from it and pass it to `Geometric::new_with_seed`.

## wasm

On `wasm32-unknown-unknown` the `getrandom` crate has no default backend. Add the following to your own `Cargo.toml`, alongside the `getrandom` feature of this crate:

```toml
[dependencies]
getrandom = { version = "0.4", features = ["wasm_js"] }
```

## Levels

`Geometric::MAX_TOTAL` is 64. A `total` above it returns `GeometricError::MaxTooLarge`. `SkipList::with_capacity` and `with_capacity_and_seed` clamp the level count they derive from the capacity to both `N` and 64.

## Example

The example builds a `SkipList` and an `OrderedSkipList` from fixed seeds. It uses only items that exist in every configuration.

```rust
use skiplist::level_generator::geometric::Geometric;
use skiplist::{OrderedSkipList, SkipList};

// A seeded generator: 16 levels, promotion probability 0.5.
let generator = Geometric::new_with_seed(16, 0.5, 42).unwrap();
let mut list: SkipList<i32> = SkipList::with_level_generator(generator);
list.extend([10, 20, 30]);
assert_eq!(list.len(), 3);

// A sorted list takes its generator the same way and fills through `Extend`.
let generator = Geometric::new_with_seed(16, 0.5, 7).unwrap();
let mut sorted: OrderedSkipList<i32> = OrderedSkipList::with_level_generator(generator);
sorted.extend([30, 10, 20]);
assert_eq!(sorted.get_by_index(0), Some(&10));

// Capacity-based construction with an explicit seed.
let sized: SkipList<i32> = SkipList::with_capacity_and_seed(1_000, 42);
assert!(sized.is_empty());
```
