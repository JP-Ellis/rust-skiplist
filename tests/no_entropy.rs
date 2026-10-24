//! Exercises the collections through the constructors that need no entropy
//! source.
//!
//! The test file must build with `--no-default-features`, where
//! `Geometric::new`, `Default` and every collection `new` are compiled out.
//! Generators therefore come from `Geometric::new_with_seed` or
//! `SkipList::with_capacity_and_seed`.

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use skiplist::{Geometric, OrderedSkipList, SkipList, SkipMap, SkipSet};

    /// Returns a seeded 16-level generator with `q = 0.5`.
    fn seeded(seed: u64) -> Result<Geometric> {
        Ok(Geometric::new_with_seed(16, 0.5, seed)?)
    }

    #[test]
    fn skip_list_with_level_generator() -> Result<()> {
        let mut list: SkipList<i32> = SkipList::with_level_generator(seeded(1)?);
        list.extend([10, 20, 30]);
        list.push_front(0);
        assert_eq!(list.len(), 4);
        assert_eq!(list.get(0), Some(&0));
        assert_eq!(list.remove(2), 20);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [0, 10, 30]);
        Ok(())
    }

    #[test]
    fn skip_list_with_capacity_and_seed() {
        let mut list = SkipList::<i32>::with_capacity_and_seed(1_000, 42);
        list.extend(0..1_000);
        assert_eq!(list.len(), 1_000);
        assert_eq!(list.get(500), Some(&500));
    }

    #[test]
    fn skip_list_append_same_capacity() {
        let mut a = SkipList::<i32>::with_capacity_and_seed(100, 1);
        let mut b = SkipList::<i32>::with_capacity_and_seed(100, 2);
        a.extend(0..50);
        b.extend(50..100);
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(
            a.iter().copied().collect::<Vec<_>>(),
            (0..100).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ordered_skip_list_with_level_generator() -> Result<()> {
        let mut sorted: OrderedSkipList<i32> = OrderedSkipList::with_level_generator(seeded(7)?);
        sorted.extend([30, 10, 20, 10]);
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted.get_by_index(0), Some(&10));
        assert_eq!(sorted.iter().copied().collect::<Vec<_>>(), [10, 10, 20, 30]);
        Ok(())
    }

    #[test]
    fn skip_set_with_level_generator() -> Result<()> {
        let mut set: SkipSet<i32> = SkipSet::with_level_generator(seeded(3)?);
        assert!(set.insert(2));
        assert!(set.insert(1));
        assert!(!set.insert(2));
        assert!(set.contains(&1));
        assert_eq!(set.first(), Some(&1));
        assert_eq!(set.len(), 2);
        Ok(())
    }

    #[test]
    fn skip_map_with_level_generator() -> Result<()> {
        let mut map: SkipMap<&str, i32> = SkipMap::with_level_generator(seeded(5)?);
        assert_eq!(map.insert("b", 2), None);
        assert_eq!(map.insert("a", 1), None);
        assert_eq!(map.insert("b", 3), Some(2));
        assert_eq!(map.get("b"), Some(&3));
        assert_eq!(map.get_by_index(0), Some((&"a", &1)));
        Ok(())
    }
}
