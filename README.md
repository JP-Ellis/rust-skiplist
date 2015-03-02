Rust Skiplist
=============

A [skiplist](http://en.wikipedia.org/wiki/Skip_list) provides a way of storing
data in a list in such as way that they are always sorted.  This implementation
is done in [Rust](http://www.rust-lang.org/).  In general, an operation
(insertion, removal, access) on an element that is (or will be) in the `i`th
position will be executed in `O(log(i))`.
