use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use arroy::internals::{self, NodeCodec};
use arroy::{Database, Distance, ItemId, Writer};
use byte_unit::{Byte, UnitType};
use heed::{EnvOpenOptions, RwTxn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use roaring::RoaringBitmap;

use crate::scenarios::*;
use crate::{partial_sort_by, Recall, RNG_SEED};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn prepare_and_run<D, F>(
    points: &[(u32, &[f32])],
    number_of_chunks: usize,
    memory: usize,
    verbose: bool,
    execute: F,
) where
    D: Distance,
    F: FnOnce(Duration, &heed::Env, Database<D>),
{
    let before_build = Instant::now();
    let dimensions = points[0].1.len();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut arroy_seed = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();

    let database =
        env.create_database::<internals::KeyCodec, NodeCodec<D>>(&mut wtxn, None).unwrap();
    let _inserted = load_into_arroy(
        &mut arroy_seed,
        &mut wtxn,
        database,
        dimensions,
        memory,
        points,
        number_of_chunks,
        verbose,
    );
    wtxn.commit().unwrap();

    (execute)(before_build.elapsed(), &env, database);
}

#[allow(clippy::too_many_arguments)]
pub fn run_scenarios<D: Distance>(
    env: &heed::Env,
    time_to_index: Duration,
    distance: &ScenarioDistance,
    number_of_chunks: usize,
    search: &[&ScenarioSearch],
    queries: &[(&u32, &&[f32], HashMap<ScenarioFiltering, (Option<RoaringBitmap>, Vec<u32>)>)],
    recall_tested: &[usize],
    database: arroy::Database<D>,
) {
    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    println!("indexing: {time_to_index:02.2?}, size: {database_size:#.2}, indexed in {number_of_chunks}x");

    for ScenarioSearch { oversampling, filtering } in search {
        let mut time_to_search = Duration::default();
        let mut recalls = Vec::new();
        for &number_fetched in recall_tested {
            let (correctly_retrieved, duration) = queries
                .par_iter()
                .map(|(&id, _target, relevants)| {
                    let rtxn = env.read_txn().unwrap();
                    let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

                    let (candidates, relevants) = &relevants[filtering];
                    // Only keep the top number fetched documents.
                    let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

                    let now = std::time::Instant::now();
                    let mut nns = reader.nns(number_fetched);
                    if let Some(oversampling) = oversampling.to_non_zero_usize() {
                        nns.oversampling(oversampling);
                    }
                    if let Some(candidates) = candidates.as_ref() {
                        nns.candidates(candidates);
                    }
                    let arroy_answer = nns.by_item(&rtxn, id).unwrap().unwrap();
                    let elapsed = now.elapsed();

                    let mut correctly_retrieved = Some(0);
                    for (id, _dist) in arroy_answer {
                        if relevants.contains(&id) {
                            if let Some(cr) = &mut correctly_retrieved {
                                *cr += 1;
                            }
                        } else if let Some(cand) = candidates.as_ref() {
                            // We set the counter to -1 if we return a filtered out candidated
                            if !cand.contains(id) {
                                correctly_retrieved = None;
                            }
                        }
                    }

                    (correctly_retrieved, elapsed)
                })
                .reduce(
                    || (Some(0), Duration::default()),
                    |(aanswer, aduration), (banswer, bduration)| {
                        (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
                    },
                );

            time_to_search += duration;
            // If non-candidate documents are returned we show a recall of -1
            let recall =
                correctly_retrieved.map_or(-1.0, |cr| cr as f32 / (number_fetched as f32 * 100.0));
            recalls.push(Recall(recall));
        }

        let filtered_percentage = filtering.to_ratio_f32() * 100.0;
        println!(
            "[arroy]  {distance:16?} {oversampling}: {recalls:?}, \
                                    searched for: {time_to_search:02.2?}, \
                                    searched in {filtered_percentage:#.2}%"
        );
    }
}

pub fn measure_arroy_distance<
    D: crate::Distance,
    const OVERSAMPLING: usize,
    const FILTER_SUBSET_PERCENT: usize,
>(
    dimensions: usize,
    memory: usize,
    points: &[(u32, &[f32])],
    number_of_chunks: usize,
    recall_tested: &[usize],
    verbose: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let now = std::time::Instant::now();
    let mut arroy_seed = StdRng::seed_from_u64(13);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<D::ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    let inserted = load_into_arroy(
        &mut arroy_seed,
        &mut wtxn,
        database,
        dimensions,
        memory,
        points,
        number_of_chunks,
        verbose,
    );
    wtxn.commit().unwrap();

    let filtered_percentage = FILTER_SUBSET_PERCENT as f32;
    let candidates = if FILTER_SUBSET_PERCENT >= 100 {
        None
    } else {
        let count = (inserted.len() as f32 * (filtered_percentage / 100.0)) as usize;
        Some(inserted.iter().take(count).collect::<RoaringBitmap>())
    };

    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    let time_to_index = now.elapsed();
    let mut total_duration = Duration::default();

    let mut recalls = Vec::new();
    for &number_fetched in recall_tested {
        if number_fetched > points.len() {
            break;
        }
        let queries: Vec<_> = (0..100).map(|_| points.choose(&mut rng).unwrap()).collect();

        let (correctly_retrieved, duration) = queries
            .into_par_iter()
            .map(|querying| {
                let rtxn = env.read_txn().unwrap();
                let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

                let relevant = partial_sort_by::<D>(
                    points
                        .iter()
                        // Only evaluate the candidate points
                        .filter(|(id, _)| {
                            candidates.as_ref().map_or(true, |cand| cand.contains(*id))
                        })
                        .map(|(i, v)| (*i, *v)),
                    querying.1,
                    number_fetched,
                );

                let now = std::time::Instant::now();
                let mut nns = reader.nns(number_fetched);
                nns.oversampling(NonZeroUsize::new(OVERSAMPLING).unwrap());
                if let Some(ref candidates) = candidates {
                    nns.candidates(candidates);
                }
                let arroy = nns.by_item(&rtxn, querying.0).unwrap().unwrap();
                let elapsed = now.elapsed();

                let mut correctly_retrieved = Some(0);
                for ret in arroy {
                    if relevant.iter().any(|(id, _, _)| *id == ret.0) {
                        if let Some(correctly_retrieved) = &mut correctly_retrieved {
                            *correctly_retrieved += 1;
                        }
                    } else if let Some(cand) = candidates.as_ref() {
                        // We set the counter to -1 if we return a filtered out candidated
                        if !cand.contains(ret.0) {
                            correctly_retrieved = None;
                        }
                    }
                }

                (correctly_retrieved, elapsed)
            })
            .reduce(
                || (Some(0), Duration::default()),
                |(aanswer, aduration), (banswer, bduration)| {
                    (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
                },
            );

        total_duration += duration;
        let recall = correctly_retrieved.unwrap_or(-1) as f32 / (number_fetched as f32 * 100.0);
        recalls.push(Recall(recall));
    }
    let time_to_search = total_duration;

    // make the distance name smaller
    let distance_name = D::name();
    println!(
        "[arroy]  {distance_name:22} x{OVERSAMPLING}: {recalls:?}, \
        indexed for: {time_to_index:02.2?}, \
        searched for: {time_to_search:02.2?}, \
        size on disk: {database_size:#.2}, \
        searched in {filtered_percentage:#.2}%"
    );
}

#[allow(clippy::too_many_arguments)]
fn load_into_arroy<D: arroy::Distance>(
    rng: &mut StdRng,
    wtxn: &mut RwTxn,
    database: Database<D>,
    dimensions: usize,
    memory: usize,
    points: &[(ItemId, &[f32])],
    number_of_chunks: usize,
    verbose: bool,
) -> RoaringBitmap {
    let mut candidates = RoaringBitmap::new();
    let avg_chunk_size = points.len() / number_of_chunks;
    for points in points.chunks(avg_chunk_size) {
        let writer = Writer::<D>::new(database, 0, dimensions);
        for (i, vector) in points.iter() {
            assert_eq!(vector.len(), dimensions);
            writer.add_item(wtxn, *i, vector).unwrap();
            assert!(candidates.push(*i));
        }
        let mut builder = writer.builder(rng);
        if verbose {
            builder.progress(|progress| println!("    {progress:?}"));
        }
        // builder.n_trees(1);
        builder.available_memory(memory).build(wtxn).unwrap();
    }
    candidates
}
