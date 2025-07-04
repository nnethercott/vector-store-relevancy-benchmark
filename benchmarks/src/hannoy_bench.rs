use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use byte_unit::{Byte, UnitType};
use hannoy::internals::{self, NodeCodec};
use hannoy::{Database, Distance, ItemId, Writer};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use roaring::RoaringBitmap;

use crate::Recall;
use crate::{scenarios::*, IndexingMetrics};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn prepare_and_run<D, F>(
    points: &[(u32, &[f32])],
    ef_construction: usize,
    verbose: bool,
    execute: F,
) where
    D: Distance,
    F: FnOnce(&IndexingMetrics, &heed::Env, Database<D>),
{
    let dimensions = points[0].1.len();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut hannoy_seed = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();
    let database =
        env.create_database::<internals::KeyCodec, NodeCodec<D>>(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let duration =
        load_into_hannoy(&mut hannoy_seed, &env, database, dimensions, points, ef_construction);

    (execute)(&duration, &env, database);
}

#[allow(clippy::too_many_arguments)]
pub fn run_scenarios<D: Distance>(
    env: &heed::Env,
    time_to_index: &IndexingMetrics,
    distance: &ScenarioDistance,
    // FIXME: won't work
    queries: &[(&u32, &&[f32], Vec<u32>)],
    recall_tested: &[usize],
    ef_search: usize,
    database: hannoy::Database<D>,
) {
    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    println!("Database size: {database_size:#.2}");
    println!("{time_to_index}");

    let mut time_to_search = Duration::default();
    let mut recalls = Vec::new();
    for &number_fetched in recall_tested {
        let (correctly_retrieved, duration) = queries
            .par_iter()
            .map(|(&id, target, relevants)| {
                let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

                let rtxn = env.read_txn().unwrap();
                let reader = hannoy::Reader::open(&rtxn, 0, database).unwrap();

                // time the search
                let now = std::time::Instant::now();
                let mut nns = reader.nns(number_fetched, ef_search);
                let hannoy_answer = nns.by_vector(&rtxn, target).unwrap();
                let elapsed = now.elapsed();

                let mut correctly_retrieved = Some(0);
                for (id, _dist) in hannoy_answer {
                    if relevants.contains(&id) {
                        if let Some(cr) = &mut correctly_retrieved {
                            *cr += 1;
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

    println!(
        "[arroy]  {distance:16?}: {recalls:?}, \
                    searched for: {time_to_search:02.2?}"
    );
}

#[allow(clippy::too_many_arguments)]
fn load_into_hannoy<D: hannoy::Distance>(
    rng: &mut StdRng,
    env: &heed::Env,
    database: Database<D>,
    dimensions: usize,
    points: &[(ItemId, &[f32])],
    ef_construction: usize,
) -> IndexingMetrics {
    let mut metrics = IndexingMetrics::new();
    let avg_chunk_size = points.len() / 1;
    let mut nb_vectors = 0;

    for points in points.chunks(avg_chunk_size) {
        tracing::info!("Inserting chunk of size {} in arroy", points.len());
        let mut wtxn = env.write_txn().unwrap();
        metrics.start_insertion();
        let writer = Writer::<D>::new(database, 0, dimensions);
        for (i, vector) in points.iter() {
            assert_eq!(vector.len(), dimensions);
            writer.add_item(&mut wtxn, *i, vector).unwrap();
        }
        metrics.end_insertion();

        tracing::info!("Starts building the trees");

        let mut builder = writer.builder(rng);

        metrics.start_building();
        builder.ef_construction(ef_construction).build(&mut wtxn).unwrap();
        metrics.end_building();
        wtxn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let reader = hannoy::Reader::open(&rtxn, 0, database).unwrap();
        drop(rtxn);

        nb_vectors += points.len();
        metrics.new_nb_vectors(nb_vectors);
        metrics.new_database_size(env.non_free_pages_size().unwrap() as usize);
    }

    metrics.end();
    metrics
}

