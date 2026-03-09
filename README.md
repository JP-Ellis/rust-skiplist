# skiplist

<!-- markdownlint-disable no-inline-html -->
<div align="center"><table>
    <tr>
        <td>Package</td>
        <td>
            <a href="https://crates.io/crates/skiplist"><img src="https://img.shields.io/crates/v/skiplist.svg" alt="Version"></a>
            <a href="https://docs.rs/skiplist/latest/skiplist/"><img src="https://img.shields.io/docsrs/skiplist" alt="docs.rs"></a>
            <a href="https://crates.io/crates/skiplist"><img src="https://img.shields.io/crates/d/skiplist.svg" alt="Downloads"></a>
        </td>
    </tr>
    <tr>
        <td>CI/CD</td>
        <td>
            <a href="https://github.com/JP-Ellis/rust-skiplist/actions/workflows/test.yml?query=branch:main"><img
                src="https://img.shields.io/github/actions/workflow/status/JP-Ellis/rust-skiplist/test.yml?branch=main&label=test"
                alt="Test Pipeline"></a>
            <a href="https://github.com/JP-Ellis/rust-skiplist/actions/workflows/release-plz.yml?query=branch:main"><img
                src="https://img.shields.io/github/actions/workflow/status/JP-Ellis/rust-skiplist/release-plz.yml?branch=main&label=release"
                alt="Release Pipeline"></a>
            <a href="https://codecov.io/gh/JP-Ellis/rust-skiplist"><img
                src="https://codecov.io/gh/JP-Ellis/rust-skiplist/graph/badge.svg?branch=main"
                alt="Code Coverage"></a>
        </td>
    </tr>
    <tr>
        <td>Meta</td>
        <td>
            <a href="https://github.com/rust-lang/rust-clippy"><img
                src="https://img.shields.io/badge/linting-clippy-0097A7"
                alt="linting - Clippy"></a>
            <a href="https://github.com/rust-lang/rustfmt"><img
                src="https://img.shields.io/badge/style-rustfmt-0057B8"
                alt="style - rustfmt"></a>
            <a href="https://github.com/rust-lang/miri"><img
                src="https://img.shields.io/badge/testing-Miri-6A0DAD"
                alt="testing - Miri"></a>
            <a href="https://crates.io/crates/skiplist"><img src="https://img.shields.io/crates/l/skiplist.svg" alt="License"></a>
        </td>
    </tr>
</table></div>
<!-- markdownlint-enable no-inline-html -->

Skip list collections with $O(\log n)$ average-case performance for access, insertion, and removal. Four variants cover the common use cases: positional sequences, sorted bags, sorted sets, and sorted maps, all with pluggable ordering via a `Comparator` trait.

> [!NOTE]
>
> **Version 1.0.0 is a complete rewrite.**
>
> The 1.0.0 release shares no code with the 0.x series. The internal architecture, public API, and crate structure have all changed. If you are upgrading from 0.x, treat this as a new dependency and review the documentation from scratch rather than diffing against the old API.

## Adding to your project

```toml
[dependencies]
skiplist = "1"
```

To use the `PartialOrdComparator` (for types that implement `PartialOrd` but not `Ord`, such as `f64`), enable the `partial-ord` feature:

```toml
[dependencies]
skiplist = { version = "1", features = ["partial-ord"] }
```

> [!WARNING]
>
> `PartialOrdComparator` panics at runtime if a comparison returns `None` (e.g. when a `NaN` is inserted or looked up). For floating-point keys, prefer `FnComparator` with `f64::total_cmp`, which provides a true total order with no panics.

## Basic usage

```rust
use skiplist::{SkipList, OrderedSkipList, SkipSet, SkipMap};

// Positional sequence - insertion order is preserved
let mut list: SkipList<i32> = SkipList::new();
list.push_back(10);
list.push_back(20);
list.insert(1, 15); // insert at index 1
assert_eq!(list[1], 15);

// Sorted bag - elements are always kept in order, duplicates allowed
let mut ordered: OrderedSkipList<i32> = OrderedSkipList::new();
ordered.insert(30);
ordered.insert(10);
ordered.insert(10); // duplicate is kept
assert_eq!(ordered.len(), 3);

// Sorted set - like OrderedSkipList but duplicates are rejected
let mut set: SkipSet<i32> = SkipSet::new();
set.insert(3);
set.insert(1);
set.insert(1); // no-op: already present
assert_eq!(set.len(), 2);

// Sorted map - unique keys, sorted by key
let mut map: SkipMap<&str, i32> = SkipMap::new();
map.insert("b", 2);
map.insert("a", 1);
assert_eq!(map.get("a"), Some(&1));
```

### Custom ordering

Ordered collections accept any comparator via the `FnComparator` wrapper - no newtype required:

```rust
use skiplist::{OrderedSkipList, comparator::FnComparator};

// Sort strings by length, then lexicographically
let cmp = FnComparator(|a: &str, b: &str| {
    a.len().cmp(&b.len()).then(a.cmp(b))
});
let mut list = OrderedSkipList::with_comparator(cmp);
list.insert("banana");
list.insert("fig");
list.insert("apple");
// Iteration order: "fig", "apple", "banana"
```

## Collections

| Collection        | Ordering                 | Duplicates       | Docs                                                                            |
| ----------------- | ------------------------ | ---------------- | ------------------------------------------------------------------------------- |
| `SkipList`        | Insertion order          | Yes              | [docs.rs](https://docs.rs/skiplist/latest/skiplist/struct.SkipList.html)        |
| `OrderedSkipList` | Sorted by comparator     | Yes              | [docs.rs](https://docs.rs/skiplist/latest/skiplist/struct.OrderedSkipList.html) |
| `SkipSet`         | Sorted by comparator     | No               | [docs.rs](https://docs.rs/skiplist/latest/skiplist/struct.SkipSet.html)         |
| `SkipMap`         | Sorted by key comparator | No (unique keys) | [docs.rs](https://docs.rs/skiplist/latest/skiplist/struct.SkipMap.html)         |

**`SkipList`** stores elements in insertion order. It does not require elements to implement `Ord`. Use it when you need a positional sequence with cheap $O(\log n)$ insertion and removal anywhere in the list, not just at the ends.

**`OrderedSkipList`** always keeps elements in sorted order and tolerates duplicates. Unlike `BTreeSet`, inserting the same value twice places two adjacent entries in the list.

**`SkipSet`** wraps `OrderedSkipList` and enforces uniqueness: inserting a value that already exists is a no-op. It mirrors the `BTreeSet` API.

**`SkipMap`** stores unique key-value pairs sorted by key. Inserting a duplicate key replaces the existing value, identical to `BTreeMap` semantics.

Full API documentation is on [docs.rs](https://docs.rs/skiplist/).

## License

MIT
