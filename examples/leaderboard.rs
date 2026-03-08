//! Realistic use case: Live sports leaderboard.
//!
//! This example uses `OrderedSkipList` to maintain a live sorted league
//! standings table.  The collection's ordering is entirely decoupled from
//! `Ord` on the element type: a `FnComparator` closure computes a composite
//! ranking score (points → goal difference → goals for) and the list keeps
//! teams in that order at all times.
//!
//! Demonstrates:
//!   - Inserting structs with a custom comparator.
//!   - Rank-based access (top-N query).
//!   - Mid-season update: remove a team, update its stats, reinsert.
//!   - Final sorted iteration to print the standings table.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example leaderboard
//! ```

#![expect(
    clippy::print_stdout,
    clippy::arithmetic_side_effects,
    reason = "This example is for demonstration, not a test."
)]

use pretty_assertions::assert_eq;
use skiplist::{FnComparator, OrderedSkipList, level_generator::geometric::Geometric};

/// Type alias for a round's results: a team name and a slice of match results.
type RoundResults<'a> = (&'a str, &'a [(&'a str, u32, u32)]);

/// Statistics for a single team in the league.
#[derive(Debug, Clone, PartialEq)]
struct Team {
    /// Team name (unique identifier).
    name: &'static str,
    /// Matches played.
    played: u32,
    /// Number of matches won.
    wins: u32,
    /// Number of matches drawn.
    draws: u32,
    /// Number of matches lost.
    losses: u32,
    /// Goals scored by the team.
    goals_for: u32,
    /// Goals conceded by the team.
    goals_against: u32,
}

/// Type alias for our standings table: an ordered skip list of teams, sorted by
/// the `standings_order` comparator defined below.
type Standings =
    OrderedSkipList<Team, 16, FnComparator<fn(&Team, &Team) -> std::cmp::Ordering>, Geometric>;

impl Team {
    /// Create a new team with zero stats.
    fn new(name: &'static str) -> Self {
        Self {
            name,
            played: 0,
            wins: 0,
            draws: 0,
            losses: 0,
            goals_for: 0,
            goals_against: 0,
        }
    }

    /// Record the result of a match.
    fn record_result(&mut self, gf: u32, ga: u32) {
        self.played += 1;
        self.goals_for += gf;
        self.goals_against += ga;
        match gf.cmp(&ga) {
            std::cmp::Ordering::Greater => self.wins += 1,
            std::cmp::Ordering::Equal => self.draws += 1,
            std::cmp::Ordering::Less => self.losses += 1,
        }
    }

    /// Points earned: 3 per win, 1 per draw.
    fn points(&self) -> u32 {
        self.wins * 3 + self.draws
    }

    /// Goal difference (positive is better).
    fn goal_difference(&self) -> i32 {
        self.goals_for.cast_signed() - self.goals_against.cast_signed()
    }
}

/// Comparator: rank teams by points (desc), then goal difference (desc), then
/// goals scored (desc).  Ties broken by name for determinism.
fn standings_order(a: &Team, b: &Team) -> std::cmp::Ordering {
    b.points()
        .cmp(&a.points())
        .then(b.goal_difference().cmp(&a.goal_difference()))
        .then(b.goals_for.cmp(&a.goals_for))
        .then(a.name.cmp(b.name))
}

fn main() {
    // --- Season start: all teams at zero ---
    let team_names = ["Ajax", "Benfica", "Celtic", "Dortmund", "Everton"];

    let mut standings: Standings = OrderedSkipList::with_comparator(FnComparator(standings_order));

    for name in team_names {
        standings.insert(Team::new(name));
    }

    // --- Round 1 results ---
    //
    // When we update a team's stats we must remove it from the list, mutate
    // it, and reinsert it.  This preserves the sorted invariant; in-place
    // mutation is not permitted on ordered collections because it would
    // silently break the ordering.

    let round1: &[RoundResults] = &[
        ("Ajax", &[("Benfica", 2, 0)]),
        ("Celtic", &[("Dortmund", 1, 3)]),
        ("Everton", &[]),
    ];
    apply_results(&mut standings, round1);

    println!("Standings after Round 1:");
    print_standings(&standings);

    // --- Round 2 results ---
    let round2: &[RoundResults] = &[
        ("Ajax", &[("Celtic", 1, 1)]),
        ("Benfica", &[("Dortmund", 0, 2)]),
        ("Everton", &[]),
    ];
    apply_results(&mut standings, round2);

    println!("\nStandings after Round 2:");
    print_standings(&standings);

    // --- Top-3 query ---
    //
    // `get_by_index` is O(log n) and directly accesses the element at rank k.
    println!("\nTop 3 teams:");
    for i in 0..standings.len().min(3) {
        if let Some(team) = standings.get_by_index(i) {
            println!("  {}. {} ({} pts)", i + 1, team.name, team.points());
        }
    }

    // Ajax leads after a win and a draw (4 pts).
    assert_eq!(standings.get_by_index(0).map(|t| t.name), Some("Ajax"));

    println!("\nDone!");
}

/// Apply a batch of match results to the standings.
#[expect(
    clippy::expect_used,
    reason = "We panic on invalid input for simplicity; \
        a real application would handle errors gracefully."
)]
fn apply_results(standings: &mut Standings, results: &[RoundResults]) {
    for (team_name, matches) in results {
        if matches.is_empty() {
            continue;
        }

        // Find and remove the team by scanning (O(n), acceptable for small
        // leagues; a real application would use an auxiliary index).
        let pos = standings
            .iter()
            .position(|t| t.name == *team_name)
            .expect("team must exist");
        let mut team = standings.remove(pos);

        for &(_, gf, ga) in *matches {
            team.record_result(gf, ga);
        }

        standings.insert(team);
    }
}

/// Print the full standings table.
fn print_standings(standings: &Standings) {
    println!(
        "{:<12} {:>2} {:>2} {:>2} {:>2} {:>4} {:>5}",
        "Team", "P", "W", "D", "L", "GD", "Pts"
    );
    println!("{}", "-".repeat(40));
    for team in standings {
        println!(
            "{:<12} {:>2} {:>2} {:>2} {:>2} {:>+4} {:>5}",
            team.name,
            team.played,
            team.wins,
            team.draws,
            team.losses,
            team.goal_difference(),
            team.points(),
        );
    }
}
