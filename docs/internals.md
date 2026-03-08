# Skip List Internals

This page is aimed at contributors and advanced users who need to understand how nodes are allocated, linked, and accessed safely.  User-facing concepts are in [`crate::docs::concepts`] instead.

## Node Structure

All four public collections share a common node type: `Node<V, N>`.  Each `Node` holds:

-   An optional value of type `V` (absent only on the sentinel head node).
-   A `prev` raw pointer to the preceding node (absent on the head).
-   A `next` raw pointer that _owns_ the following node (absent on the tail).
-   An array of up to `N` skip-link entries (`Link<V, N>`), each recording the target node pointer and the number of level-0 hops the link spans.

As a concrete example:

```text
[4] head
[3] head --------------------------> 6
[2] head ---------------------> 5 -> 6
[1] head ------> 2 -----------> 5 -> 6 ------> 8
[*] head -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
```

Traversal at level 0 follows plain `prev` / `next` pointers.  To move from node 1 to node 10 quickly, the search walks: `head → 6 → 8 → 9 → 10`.

## Ownership Model

Nodes live on the heap.  The ownership chain is:

1.  Each collection struct owns the **head** node (a sentinel with no value).
2.  Each node **owns** the immediately following node through its `next` pointer.

This means dropping the collection automatically drops the head, which drops the first data node, which drops the second, and so on; no manual cleanup loop required.

Skip links do **not** own their targets.  They are non-owning raw pointers used only for fast traversal.  The `Link` type stores a `NonNull` pointer plus a distance; it never participates in the drop chain.

## Pointer Invalidation Rules

Pointer safety is the most critical concern when working below the public API.  There are two ways a pointer can be silently invalidated:

1.  **Dropping a node without unwiring its links.**  If a node is freed while other nodes still hold skip links pointing to it, those links become dangling.  All removal operations must unwire every skip link that spans the removed node before the node is freed.

2.  **Moving a node.**  Nodes must stay at the address they were allocated at for as long as any pointer points to them.  This is why `Node` is always heap-allocated via `Box::new` and never returned by value from traversal helpers.  A function must not return a `Node` by value unless that node is fully detached from the list.

## Linking and Levels

In addition to `prev` / `next`, each node has an array of skip links.  Key observations:

-   A node's _level_ (or _height_) is the number of skip links it has, excluding `prev` / `next`.  In the example above, node 1 has height 0, node 2 has height 1, and node 5 has height 2.
-   The **head** node always has the maximum number of levels (`N`).  Unused levels point to nothing.
-   If a node is linked at level `k`, it is reachable from all levels `0..=k`.

When inserting a node, the level generator draws a random height for the new node.  The visitor records the closest preceding node at each level (_precursor_) during the downward search, and the insertion routine wires the new node into every level up to its height.

## Pointer Accessor Reference

`Node` exposes three accessor families for `next` and `prev`:

| Method                            | Returns                 | When to use                                                                                                                                                            |
| --------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `next()` / `prev()`               | `Option<NonNull<Self>>` | **Default choice.** Any time the pointer is stored across loop iterations, passed to another function, or may later be written through.                                |
| `next_as_ref()` / `prev_as_ref()` | `Option<&Self>`         | Inline, read-only access only.  Safe when the result is consumed in the same expression and never converted back to a raw pointer, e.g. `node.next_as_ref()?.value()`. |
| `next_as_mut()` / `prev_as_mut()` | `Option<&mut Self>`     | Short-lived exclusive mutation; see individual method safety docs.                                                                                                     |

### The Frozen anti-pattern

**Never** write `node.next_as_ref().map(NonNull::from)` (or the `prev` equivalent).  Calling `next_as_ref()` creates a _shared_ reborrow of the neighbouring node; Tree Borrows records this with a **Frozen** provenance tag.  Converting the resulting `&Node` back to `NonNull` via `NonNull::from` carries that Frozen tag into the raw pointer.  Any subsequent write through that pointer, or through any child pointer derived from it, is undefined behaviour under Tree Borrows.

If you need a raw pointer to the next or previous node, use `next()` / `prev()` directly: they return the stored `NonNull` without creating any reborrow.

---

## Why `NonNull` Instead of `Box`

The head node is stored as `NonNull<Node<T, N>>` rather than `Box<Node<T, N>>`.  The reason is Tree Borrows provenance.

`Box<T>` receives special **Unique** retagging when accessed: each access creates a new Reserved child tag.  Any write through a sibling tag, such as the raw pointers held by other nodes' skip links, disables the Box's tag.  Under Miri's Tree Borrows model, this is reported as undefined behaviour even though the code is logically correct.

Storing the head as `NonNull` with a single provenance tag (obtained from `Box::into_raw(Box::new(...))`) means all accesses share the same root provenance.  Both shared reads (`head_ref`) and exclusive writes (`head_mut`, `insert_after`, etc.) are valid under Tree Borrows.

The invariant is maintained manually: `head` is exclusively owned by the collection struct and freed in `Drop` via `Box::from_raw`.

This same rationale applies to the `Node::insert_after` helper: the `Box::into_raw` call in the allocator preserves root provenance for the newly created node, which is then wired into the list's existing provenance tree.
