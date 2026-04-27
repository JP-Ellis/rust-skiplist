# Gap Cursors

> **Unstable API.** The cursor API requires the `cursor` feature flag and may change in a future minor release without prior notice.
>
> ```toml
> [dependencies]
> skiplist = { version = "...", features = ["cursor"] }
> ```

This page explains the gap cursor model, how `lower_bound` and `upper_bound` position a cursor, and how to use cursors for common tasks.  For the complete method list, see the individual reference pages: [`ordered_skip_list::cursor`], [`skip_set::cursor`], and [`skip_map::cursor`].

## The gap cursor model

A cursor points at a **gap between two adjacent elements**, not at an element
itself.  Think of the gaps as the spaces in between:

```text
     ┌──── gap 0 (leftmost, before 10)
     │    ┌── gap 1 (between 10 and 20)
     │    │    ┌── gap 2 (between 20 and 30)
     │    │    │    ┌── gap 3 (rightmost, after 30)
     ↓    ↓    ↓    ↓
 head [10] [20] [30] tail
```

At any gap, the cursor can see two neighbours:

| Method      | Returns                                                    |
| ----------- | ---------------------------------------------------------- |
| `peek_prev` | the element on the **left** (`None` at the leftmost gap)   |
| `peek_next` | the element on the **right** (`None` at the rightmost gap) |

Moving the cursor:

| Method | Effect                                       |
| ------ | -------------------------------------------- |
| `next` | step right; returns the element just crossed |
| `prev` | step left; returns the element just crossed  |

### Why gap-based rather than element-based?

An element-based cursor (like a slice iterator) points _at_ a node.  Inserting before or after that node is natural, but the semantics become ambiguous when the pointed-to element is removed: does the cursor move forward or backward?

A gap-based cursor sidesteps the problem entirely.  The cursor's position is defined by what is on either side of it.  Inserting next to the cursor does not change where the cursor sits: `insert_after` keeps the new element on the right and the cursor unchanged; `insert_before` inserts on the left and advances the cursor so the new element becomes the new left neighbour.  Removing the right neighbour also leaves the cursor at the same gap; removing the left neighbour retreats it.

This is the same design as the `BTreeMap` cursors introduced in the Rust standard library (see [RFC 2570](https://github.com/rust-lang/rfcs/blob/master/text/2570-linked-list-cursors.md) and [BTree cursors tracking issue](https://github.com/rust-lang/rust/issues/107540)).

## Positioning a cursor: `lower_bound` vs `upper_bound`

Both factory methods accept a `Bound<&Q>` and return a cursor at a gap chosen relative to some query value `q`.

| Method        | Bound variant  | Gap placed…                         |
| ------------- | -------------- | ----------------------------------- |
| `lower_bound` | `Unbounded`    | before the first element (leftmost) |
| `lower_bound` | `Included(&q)` | before the first element `>= q`     |
| `lower_bound` | `Excluded(&q)` | before the first element `> q`      |
| `upper_bound` | `Unbounded`    | after the last element (rightmost)  |
| `upper_bound` | `Included(&q)` | after the last element `<= q`       |
| `upper_bound` | `Excluded(&q)` | after the last element `< q`        |

`lower_bound` lands at the **left edge** of the matching region; `upper_bound` lands at the **right edge**.  For a set containing `[1, 2, 2, 3]`:

```text
lower_bound(Included(&2))   →  gap between 1 and the first 2
upper_bound(Included(&2))   →  gap between the last 2 and 3
lower_bound(Excluded(&2))   →  gap between the last 2 and 3
upper_bound(Excluded(&2))   →  gap between 1 and the first 2
```

Notice that `lower_bound(Included(&q))` and `upper_bound(Excluded(&q))` are equivalent, as are `lower_bound(Excluded(&q))` and `upper_bound(Included(&q))`.

## How-to guides

### Find the neighbours of a value

```rust
# #[cfg(feature = "cursor")] {
use skiplist::SkipSet;
use core::ops::Bound;

let set: SkipSet<i32> = [1, 3, 5, 7].into_iter().collect();

// What is immediately before and after 4 in the set?
let cur = set.lower_bound(Bound::Included(&4));
assert_eq!(cur.peek_prev(), Some(&3));  // largest element < 4
assert_eq!(cur.peek_next(), Some(&5));  // smallest element >= 4
# }
```

### Collect elements in a half-open range

Navigate from `lower_bound` until `peek_next` exits the range:

```rust
# #[cfg(feature = "cursor")] {
use skiplist::SkipSet;
use core::ops::Bound;

let set: SkipSet<i32> = (1..=10).collect();

// Collect elements in [3, 7).
let mut cur = set.lower_bound(Bound::Included(&3));
let mut result = Vec::new();
while let Some(&v) = cur.peek_next() {
    if v >= 7 { break; }
    cur.next();
    result.push(v);
}
assert_eq!(result, [3, 4, 5, 6]);
# }
```

### Insert a value at a known gap

`lower_bound_mut` / `upper_bound_mut` return a [`CursorMut`] that supports insertion.  Use `insert_after` to place the new element immediately to the right of the cursor:

```rust
# #[cfg(feature = "cursor")] {
use skiplist::ordered_skip_list::OrderedSkipList;
use core::ops::Bound;

let mut list = OrderedSkipList::<i32>::new();
for v in [1, 3, 5] { list.insert(v); }

// Insert 2 without re-searching from the root.
{
    let mut cur = list.lower_bound_mut(Bound::Included(&2));
    // cur is at the gap between 1 and 3.
    cur.insert_after(2).expect("2 is in order");
}

let vals: Vec<_> = list.iter().copied().collect();
assert_eq!(vals, [1, 2, 3, 5]);
# }
```

### Batch-insert a pre-sorted sequence

Because each insertion starts from the cursor's current position rather than the root, inserting a pre-sorted slice into a fixed gap costs `$O(k \log n)$` rather than `$O(k \log(n + k))$`:

```rust
# #[cfg(feature = "cursor")] {
use skiplist::ordered_skip_list::OrderedSkipList;
use core::ops::Bound;

let mut list: OrderedSkipList<i32> = [1, 10].into_iter().collect();
let to_insert = [2, 3, 4, 5]; // must be sorted

let mut cur = list.lower_bound_mut(Bound::Included(&2));
for v in to_insert {
    cur.insert_after(v).expect("values are in order");
    // Advance past the element we just inserted so the next
    // insert_after is positioned correctly.
    cur.next();
}

let vals: Vec<_> = list.iter().copied().collect();
assert_eq!(vals, [1, 2, 3, 4, 5, 10]);
# }
```

### Drain a range with a mutable cursor

`remove_next` removes the right neighbour and leaves the cursor in place.  Repeat until `peek_next` is outside the range:

```rust
# #[cfg(feature = "cursor")] {
use skiplist::SkipSet;
use core::ops::Bound;

let mut set: SkipSet<i32> = (1..=10).collect();

// Remove all elements in [3, 7].
let mut cur = set.lower_bound_mut(Bound::Included(&3));
while cur.peek_next().map_or(false, |&v| v <= 7) {
    cur.remove_next();
}

let remaining: Vec<_> = set.iter().copied().collect();
assert_eq!(remaining, [1, 2, 8, 9, 10]);
# }
```

### Mutate map values while navigating

[`SkipMap`]'s `CursorMut::peek_next` returns `(&K, &mut V)`, allowing the value to be updated without re-searching:

```rust
# #[cfg(feature = "cursor")] {
use skiplist::SkipMap;
use core::ops::Bound;

let mut map: SkipMap<i32, i32> = [(1, 10), (2, 20), (3, 30)].into_iter().collect();

// Double every value in the range [1, 2].
let mut cur = map.lower_bound_mut(Bound::Included(&1));
while let Some((k, v)) = cur.peek_next() {
    if *k > 2 { break; }
    *v *= 2;
    cur.next();
}

assert_eq!(map.get(&1), Some(&20));
assert_eq!(map.get(&2), Some(&40));
assert_eq!(map.get(&3), Some(&30)); // unchanged
# }
```

## Stability note

The cursor API is modelled on the `BTreeMap` and `LinkedList` cursor design from [RFC 2570](https://github.com/rust-lang/rfcs/pull/2570) and the [BTreeMap cursor tracking issue](https://github.com/rust-lang/rust/issues/107540), which has not yet been stabilised in the Rust standard library as of this writing.  Because the design space is still open, this implementation is likewise marked **unstable**: the API may change in a future minor release.  Breaking changes will be noted in the changelog.

[`ordered_skip_list::cursor`]: crate::ordered_skip_list::cursor
[`skip_set::cursor`]: crate::skip_set::cursor
[`skip_map::cursor`]: crate::skip_map::cursor
[`CursorMut`]: crate::ordered_skip_list::cursor::CursorMut
[`SkipMap`]: crate::SkipMap
