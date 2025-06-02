use std::collections::HashMap;
use std::fmt::Write as _;

use arroy::distances::{BinaryQuantizedCosine, Cosine, Hamming};
use benchmarks::scenarios::ScenarioSearch;
use benchmarks::{arroy_bench, scenarios, MatLEView, RNG_SEED};
use byte_unit::Byte;
use clap::Parser;
use enum_iterator::Sequence;
use itertools::{iproduct, Itertools};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::SeedableRng;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use slice_group_by::GroupBy;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(long, value_enum)]
    datasets: Vec<scenarios::Dataset>,

    #[arg(long, value_enum)]
    contenders: Vec<scenarios::ScenarioContender>,

    #[arg(long, value_enum)]
    distances: Vec<scenarios::ScenarioDistance>,

    #[arg(long, value_enum)]
    over_samplings: Vec<scenarios::ScenarioOversampling>,

    #[arg(long, value_enum)]
    filterings: Vec<scenarios::ScenarioFiltering>,

    /// The list of recall to be tested.
    #[arg(long, default_value_t = String::from("1,10,20,50,100,500"))]
    recall_tested: String,

    /// Number of vectors to evaluate from the datasets.
    #[arg(long, default_value_t = 10_000)]
    count: usize,

    /// These numbers correspond to the numbers of chunks that the dataset will be split into for indexing.
    ///
    /// Each number corresponds to a new indexation in x chunks. Use a comma to separate multiple features.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    number_of_chunks: Vec<usize>,

    /// Memory available for indexing.
    #[arg(long, default_value_t = Byte::MAX)]
    memory: Byte,

    /// When set to true, will print all the steps it goes through.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let Args {
        datasets,
        count,
        number_of_chunks,
        contenders,
        distances,
        over_samplings,
        filterings,
        memory,
        recall_tested,
        verbose,
    } = Args::parse();

    let datasets = set_or_all::<_, MatLEView<f32>>(datasets);
    let contenders = set_or_all::<_, scenarios::ScenarioContender>(contenders);
    let distances = set_or_all::<_, scenarios::ScenarioDistance>(distances);
    let over_samplings = set_or_all::<_, scenarios::ScenarioOversampling>(over_samplings);
    let filterings = set_or_all::<_, scenarios::ScenarioFiltering>(filterings);
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

    let scenaris: Vec<_> = iproduct!(datasets, distances, contenders, over_samplings, filterings)
        .map(|(dataset, distance, contender, oversampling, filtering)| {
            (dataset, distance, contender, ScenarioSearch { oversampling, filtering })
        })
        .sorted()
        .collect();

    let mut previous_dataset = None;
    for grp in scenaris
        .linear_group_by(|(da, dia, ca, _), (db, dib, cb, _)| da == db && dia == dib && ca == cb)
    {
        let (dataset, distance, contender, _) = &grp[0];
        let search: Vec<&ScenarioSearch> = grp.iter().map(|(_, _, _, s)| s).collect();

        if previous_dataset != Some(dataset.name()) {
            previous_dataset = Some(dataset.name());
            dataset.header();
            if dataset.len() != count {
                let c = count.min(dataset.len());
                println!(
                    "\x1b[1m{c}\x1b[0m vectors are used for this measure and {memory}B of memory",
                );
            }
        }

        let points: Vec<_> =
            dataset.iter().take(count).enumerate().map(|(i, v)| (i as u32, v)).collect();
        let memory = memory.as_u64() as usize;

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
                    points.par_sort_unstable_by_key(|(_, v)| {
                        OrderedFloat(benchmarks::distance::<Hamming>(target, v))
                    });

                    // We collect the different filtered versions here.
                    let filtered: HashMap<_, _> = search
                        .iter()
                        .map(|ScenarioSearch { filtering, .. }| {
                            let candidates = match filtering {
                                scenarios::ScenarioFiltering::NoFilter => None,
                                filtering => {
                                    let total = points.len() as f32;
                                    let filtering = filtering.to_ratio_f32();
                                    Some(
                                        points
                                            .iter()
                                            .map(|(id, _)| id)
                                            .take((total * filtering) as usize)
                                            .collect::<RoaringBitmap>(),
                                    )
                                }
                            };

                            // This is the real expected answer without the filtered out candidates.
                            let answer = points
                                .iter()
                                .map(|(id, _)| *id)
                                .filter(|&id| candidates.as_ref().map_or(true, |c| c.contains(id)))
                                .take(max)
                                .collect::<Vec<_>>();

                            (*filtering, (candidates, answer))
                        })
                        .collect();

                    (id, target, filtered)
                })
                .collect()
        };
        println!("Starting indexing process");

        for number_of_chunks in &number_of_chunks {
            match contender {
                scenarios::ScenarioContender::Qdrant => match distance {
                    scenarios::ScenarioDistance::Cosine => {
                        arroy_bench::prepare_and_run::<Cosine, _>(
                            &points,
                            *number_of_chunks,
                            memory,
                            verbose,
                            |time_to_index, env, database| {
                                arroy_bench::run_scenarios(
                                    env,
                                    time_to_index,
                                    distance,
                                    *number_of_chunks,
                                    &search,
                                    &queries,
                                    &recall_tested,
                                    database,
                                );
                            },
                        )
                    }
                },
                scenarios::ScenarioContender::Arroy => match distance {
                    scenarios::ScenarioDistance::Cosine => {
                        arroy_bench::prepare_and_run::<Hamming, _>(
                            &points,
                            *number_of_chunks,
                            memory,
                            verbose,
                            |time_to_index, env, database| {
                                arroy_bench::run_scenarios(
                                    env,
                                    time_to_index,
                                    distance,
                                    *number_of_chunks,
                                    &search,
                                    &queries,
                                    &recall_tested,
                                    database,
                                );
                            },
                        )
                    }
                },
            }
        }

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
