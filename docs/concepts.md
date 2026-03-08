# Skip List Concepts

This page explains the ideas behind skip lists, when to reach for them, how the four collections in this crate relate to each other, and why the comparator design looks the way it does.

## How a Skip List Works

A skip list layers multiple linked lists on top of each other. The bottom layer
(level 0) contains every element. Each higher layer holds a randomly chosen
subset of the elements below it; each node is promoted independently with a
fixed probability (typically 0.5). A search starts at the top layer and drops
down whenever the next node at that layer would overshoot the target, arriving
at the answer in `$O(\log n)$` expected steps.

```text
Level 3:  head ----------> [20] ---------------------------------> tail
Level 2:  head --> [10] -> [20] ---------> [50] -----------------> tail
Level 1:  head --> [10] -> [20] -> [30] -> [50] -> [70] ---------> tail
Level 0:  head --> [10] -> [20] -> [30] -> [40] -> [50] -> [70] -> tail
```

To look up 40: start at level 3 and skip past 20, then drop to level 2 (50 is too far), then level 1 (50 is still too far), then step forward at level 0 to find 40.  Only four comparisons rather than the six that a plain linked list would need to traverse.

The probabilistic nature of the structure means that, with high probability, no single chain of nodes is excessively long.  Unlike a balanced tree, no rebalancing is needed after insertion or removal; the skip links are wired at node-creation time and left alone thereafter.

---

## When to Use a Skip List

Skip lists are a good fit when you need:

-   **Sorted sequences with duplicates** ([`OrderedSkipList`]): unlike `BTreeSet`, an ordered skip list happily holds multiple copies of the same value.
-   **Rank-based access** ([`SkipList`]): `$O(\log n)$` random access by position, like `Vec` but with `$O(\log n)$` insertion and removal anywhere.
-   **A sorted map or set with a custom ordering** ([`SkipMap`], [`SkipSet`]): any comparison function can be plugged in via the [`Comparator`] trait without requiring [`Ord`] on the element type.

If all you need is a set or map with unique keys and the default `Ord` ordering, the standard library's `BTreeSet` / `BTreeMap` will often be faster in practice due to better cache locality.  Skip lists shine when duplicate-tolerant ordering, custom comparators, or rank queries matter.

## The Four Collections

| Collection          | Ordering                 | Duplicates       | Use when…                                                                       |
| ------------------- | ------------------------ | ---------------- | ------------------------------------------------------------------------------- |
| [`SkipList`]        | Insertion order          | Yes              | You need a positional sequence with `$O(\log n)$` insert/remove/access anywhere |
| [`OrderedSkipList`] | Sorted by comparator     | Yes              | You need a sorted bag (multiple equal values, always in order)                  |
| [`SkipSet`]         | Sorted by comparator     | No               | You need a sorted set (each value at most once)                                 |
| [`SkipMap`]         | Sorted by key comparator | No (unique keys) | You need a sorted key-value map                                                 |

**[`SkipList`]** stores elements in insertion order, like a `Vec` but without the shifting cost of mid-list inserts and removes.  It does not require elements to be `Ord`; elements are accessed by numeric index.

**[`OrderedSkipList`]** always keeps elements in sorted order.  Unlike `BTreeSet` it tolerates duplicates; inserting the same value twice places two adjacent entries in the list.  Direct mutation of stored values is not exposed because it could break the sorted invariant.

**[`SkipSet`]** wraps [`OrderedSkipList`] and enforces uniqueness: inserting a value that already exists is a no-op.  It mirrors the `BTreeSet` API and adds set-algebra operations (union, intersection, difference, symmetric difference).

**[`SkipMap`]** stores unique key-value pairs sorted by key.  Inserting a duplicate key replaces the existing value and returns the old one, identical to `BTreeMap` semantics.  Values can be mutated in place because changing a value does not affect key order.

## Comparator Design

The ordered collections are parameterised over a `C: Comparator<T>` type parameter rather than requiring `T: Ord` on the struct definition.  This mirrors the approach taken by `BTreeMap` (which requires `K: Ord` only on methods, not on the struct), but goes one step further by making the ordering fully runtime-pluggable.

### Why not `T: Ord` on the struct?

Bounding the struct (`struct OrderedSkipList<T: Ord>`) would mean:

1.  Types that do not implement `Ord` (e.g. `f64`) can never be stored.
2.  You cannot provide a custom order for a type that _does_ implement `Ord` without wrapping it in a newtype.

The `Comparator<T>` approach avoids both problems.

### The built-in comparators

-   **[`OrdComparator`]**: delegates to `T: Ord`.  This is the default; zero overhead.
-   **[`FnComparator`]**: wraps any `fn(&T, &T) -> Ordering` (or compatible closure).  Use this whenever you need a custom order without writing a new type.
-   **[`PartialOrdComparator`]**: delegates to `T: PartialOrd` and panics if a comparison returns `None`.  Available only with the `partial-ord` feature.  See [`crate::docs::partial_ord`] for guidance on when this is appropriate.

### No `*By` variants

Because `FnComparator` can wrap any closure, separate `insert_by`, `remove_by`, and `contains_by` methods are not needed.  Choose the comparator once at construction time and use the regular API throughout.

## Capacity and Levels

Each node in a skip list carries a tower of forward pointers.  The height of that tower is chosen randomly by the level generator at insertion time.  With the default `Geometric` generator (promotion probability `p = 0.5`) and `N` levels, the expected number of nodes reaching the top level is approximately `$n / 2^{N-1}$`.

This means `N = 16` is optimal for collections of up to roughly `$2^{16} = 65{,}536$` elements: at that size, the top level spans the entire list in `$O(1)$` steps, giving the best average search path length.  For larger collections, increase the const generic `N`:

```rust
use skiplist::SkipList;

// Good for up to ~4 billion elements.
let mut large: SkipList<u32, 32> = SkipList::new();
```

When you know the expected size in advance, use `SkipList::with_capacity` to pre-configure the level generator so that skip links span the right number of nodes from the start:

```rust
use skiplist::SkipList;

let mut list = SkipList::<i32>::with_capacity(10_000);
```

If `capacity` implies more levels than `N` allows, the level count is clamped to `N`; the list remains correct but the skip links will be denser at the top than ideal.

The formula used is `levels = 1 + ceil(log2(capacity))`, derived from the condition that the expected number of nodes at the topmost level equals ~1 when the list holds `capacity` elements.

[`SkipList`]: crate::SkipList
[`OrderedSkipList`]: crate::OrderedSkipList
[`SkipSet`]: crate::SkipSet
[`SkipMap`]: crate::SkipMap
[`Comparator`]: crate::Comparator
[`OrdComparator`]: crate::OrdComparator
[`FnComparator`]: crate::FnComparator
[`PartialOrdComparator`]: crate::PartialOrdComparator
