-   We use `cargo nextest run` to run tests.
-   When referencing modules within the same crate, prefer `crate::module::function` over `super::module::function`.
-   Use `cargo +nightly fmt` to format your code.
-   We avoid introducing panics where possible. If you must introduce a panic, prefer `expect` over `unwrap`.
