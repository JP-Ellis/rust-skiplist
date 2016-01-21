=============
Rust-Skiplist
=============

:status: published
:title: Rust-Skiplist
:slug: rust-skiplist
:date: 2015-12-12
:sort: 5
:template: page_index
:share: True
:github: https://github.com/JP-Ellis/rust-skiplist/
:release: True
:summary: A skiplist is a way of storing elements in such a way that elements
          can be efficiently accessed, inserted and removed, all in
          \\(O(\\log(n))\\) on average. Rust-SkipList is an implementation of
          skiplists inside `Rust <https://www.rust-lang.org/>`_.

A skiplist is a way of storing elements in such a way that elements can be
efficiently accessed, inserted and removed, all in \\(O(\\log(n))\\) on average.
Rust-SkipList is an implementation of skiplists inside `Rust
<https://www.rust-lang.org/>`_.

Conceptually, a skiplist resembles something like:

.. code-block:: text

   <head> ----------> [2] --------------------------------------------------> [9] ---------->
   <head> ----------> [2] ------------------------------------[7] ----------> [9] ---------->
   <head> ----------> [2] ----------> [4] ------------------> [7] ----------> [9] --> [10] ->
   <head> --> [1] --> [2] --> [3] --> [4] --> [5] --> [6] --> [7] --> [8] --> [9] --> [10] ->

where we each node ``[x]`` has references to nodes further down the list,
allowing the algorithm to effectively skip ahead.

When accessing elements, the algorithm begins with the head node on the first
(top-most) level and moves as far right without exceeding the element desired.
The algorithm then moves down a level and repeats the process until it finds the
desired node.

The bottom-most level is simply a linked list (i.e. each link always goes to the
next node in the list), and each level up, the links become progressively longer
skipping more and more elements.  The height of each node is determined
randomly; typically with a geometric distribution with \\(p = 0.5\\).  This way
the number of nodes of height \\(n\\) is approximately twice the number of nodes
of height \\(n+1\\) and provides a good trade-off between memory usage and
speed.

In order to use this crate, just add the following to your project's
``Cargo.toml``:

.. code-block:: toml

   [dependencies]

   skiplist = "*"

and the documentation can be accessed `here
<https://jp-ellis.github.io/rust-skiplist/skiplist/>`_.
