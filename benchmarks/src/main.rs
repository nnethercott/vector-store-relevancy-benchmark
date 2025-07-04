use std::collections::HashMap;
use std::fmt::Write as _;

use benchmarks::scenarios::ScenarioSearch;
use benchmarks::{hannoy_bench, scenarios, MatLEView, RNG_SEED};
use byte_unit::Byte;
use clap::Parser;
use enum_iterator::Sequence;
use hannoy::distances::{BinaryQuantizedCosine, Cosine, Euclidean};
use itertools::{iproduct, Itertools};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::SeedableRng;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use slice_group_by::GroupBy;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn parse_number_with_underscores(s: &str) -> Result<usize, std::num::ParseIntError> {
    s.replace('_', "").parse()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(long, value_enum)]
    datasets: Vec<scenarios::Dataset>,

    #[arg(long, value_enum)]
    distances: Vec<scenarios::ScenarioDistance>,

    /// The list of recall to be tested.
    #[arg(long, default_value_t = String::from("1,10,20,50,100,500"))]
    recall_tested: String,

    /// Number of vectors to evaluate from the datasets.
    #[arg(long, default_value_t = 10_000, value_parser = parse_number_with_underscores)]
    count: usize,

    /// hnsw build param
    #[arg(long, default_value_t = 400)]
    ef_construction: usize,

    /// hnsw search param
    #[arg(long, default_value_t = 10)]
    ef_search: usize,

    /// When set to true, will print all the steps it goes through.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let Args { datasets, count, distances, recall_tested, verbose, ef_construction, ef_search } =
        Args::parse();

    if verbose {
        // Initialize tracing with the specified level
        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let filter = format!("hannoy=debug,benchmarks=debug");
            EnvFilter::new(filter)
        });

        FmtSubscriber::builder()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .init();
    }

    let datasets = set_or_all::<_, MatLEView<f32>>(datasets);
    let distances = set_or_all::<_, scenarios::ScenarioDistance>(distances);
    let recall_tested: Vec<usize> = recall_tested
        .split(',')
        .enumerate()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(i, n)| {
            n.trim()
                .parse()
                .unwrap_or_else(|_| panic!("Could not parse recall value `{n}` at index `{i}`."))
        })
        .collect();

    let scenaris: Vec<_> = iproduct!(datasets, distances)
        .map(|(dataset, distance)| (dataset, distance))
        .sorted()
        .collect();

    let mut previous_dataset = None;
    for grp in scenaris
        .linear_group_by(|(da, dia), (db, dib)| da == db && dia == dib)
    {
        let (dataset, distance) = &grp[0];

        if previous_dataset != Some(dataset.name()) {
            previous_dataset = Some(dataset.name());
            dataset.header();
            if dataset.len() != count {
                let c = count.min(dataset.len());
                println!(
                    "\x1b[1m{c}\x1b[0m vectors are used for this measure",
                );
            }
        }

        let points: Vec<_> =
            dataset.iter().take(count).enumerate().map(|(i, v)| (i as u32, v)).collect();

        let mut recall_tested_s = String::new();
        recall_tested
            .iter()
            .for_each(|recall| write!(&mut recall_tested_s, "{recall:4}, ").unwrap());
        let recall_tested_s = recall_tested_s.trim_end_matches(", ");
        println!("Recall tested is:   [{recall_tested_s}]");

        let max = recall_tested.iter().max().copied().unwrap_or_default();
        // If we have no recall we can skip entirely the generation of the queries
        let queries = if max == 0 {
            Vec::new()
        } else {
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            (0..100)
                .map(|_| points.choose(&mut rng).unwrap())
                .map(|(id, target)| {
                    let mut points = points.clone();
                    points.par_sort_unstable_by_key(|(_, v)| match distance {
                        scenarios::ScenarioDistance::Cosine => {
                            OrderedFloat(benchmarks::distance::<Cosine>(target, v))
                        }
                        scenarios::ScenarioDistance::BqCosine => {
                            OrderedFloat(benchmarks::distance::<BinaryQuantizedCosine>(target, v))
                        }
                        scenarios::ScenarioDistance::Euclidean => {
                            OrderedFloat(benchmarks::distance::<Euclidean>(target, v))
                        }
                    });

                    let answer = points
                        .iter()
                        .map(|(id, _)| *id)
                        .take(max)
                        .collect::<Vec<_>>();

                    (id, target, answer)
                })
                .collect()
        };
        println!("Starting indexing process");

        // macro simplifying benchmark execution depending on distance type
        macro_rules! run {
            ($D: ty) => {
                hannoy_bench::prepare_and_run::<$D, _>(
                    &points,
                    ef_construction,
                    verbose,
                    |time_to_index, env, database| {
                        hannoy_bench::run_scenarios(
                            env,
                            time_to_index,
                            distance,
                            &queries,
                            &recall_tested,
                            ef_search,
                            database,
                        );
                    },
                )
            };
        }

        match distance {
            scenarios::ScenarioDistance::Cosine => run!(Cosine),
            scenarios::ScenarioDistance::BqCosine => run!(BinaryQuantizedCosine),
            scenarios::ScenarioDistance::Euclidean => run!(Euclidean),
        };

        println!();
    }
}

fn set_or_all<S, T>(datasets: Vec<S>) -> Vec<T>
where
    S: Sequence,
    S: Into<T>,
{
    if datasets.is_empty() {
        enum_iterator::all::<S>().map(Into::into).collect()
    } else {
        datasets.into_iter().map(Into::into).collect()
    }
}
