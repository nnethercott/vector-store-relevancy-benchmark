use core::fmt;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;
use hannoy::internals::{self, NodeCodec};
use hannoy::{Database, Distance, ItemId, Writer};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use roaring::RoaringBitmap;
use std::time::Instant;
use arroy::distances::*;
use byte_unit::rust_decimal::Decimal;
use byte_unit::{Byte, Unit, UnitType};

use crate::Recall;
use crate::scenarios::*;
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

    let mut hannoy_seed = StdRng::seed_from_u64(42);
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
    search: &[&ScenarioSearch],
    queries: &[(&u32, &&[f32], HashMap<ScenarioFiltering, (Option<RoaringBitmap>, Vec<u32>)>)],
    recall_tested: &[usize],
    database: hannoy::Database<D>,
) {
    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    println!("Database size: {database_size:#.2}");
    println!("{time_to_index}");

    for ScenarioSearch { oversampling: _, filtering } in search {
        let mut time_to_search = Duration::default();
        let mut recalls = Vec::new();
        for &number_fetched in recall_tested {
            let (correctly_retrieved, duration) = queries
                .par_iter()
                .map(|(&id, target, relevants)| {
                    let rtxn = env.read_txn().unwrap();
                    let reader = hannoy::Reader::open(&rtxn, 0, database).unwrap();

                    let (candidates, relevants) = &relevants[filtering];
                    // Only keep the top number fetched documents.
                    let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

                    let now = std::time::Instant::now();
                    // let mut nns = reader.nns(number_fetched, 1*number_fetched.min(1000));
                    let mut nns = reader.nns(number_fetched, 200);
                    let arroy_answer = nns.by_vector(&rtxn, target).unwrap();
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

            time_to_search += duration/(queries.len() as u32);
            // If non-candidate documents are returned we show a recall of -1
            let recall =
                correctly_retrieved.map_or(-1.0, |cr| cr as f32 / (number_fetched as f32 * (queries.len() as f32)));
            recalls.push(Recall(recall));
        }

        let filtered_percentage = filtering.to_ratio_f32() * 100.0;
        println!(
            "[hannoy]  {distance:16?} {recalls:?}, \
                                    searched for: {time_to_search:02.2?}, \
                                    searched in {filtered_percentage:#.2}%"
        );
    }
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

        nb_vectors += points.len();
        metrics.new_nb_vectors(nb_vectors);
        metrics.new_database_size(env.non_free_pages_size().unwrap() as usize);
    }

    metrics.end();
    metrics
}

#[derive(Debug)]
pub struct IndexingMetrics {
    start: Instant,
    end: Instant,
    insert_durations: Vec<(Instant, Instant)>,
    build_durations: Vec<(Instant, Instant)>,
    nb_vectors: Vec<usize>,
    database_size: Vec<usize>,
}

impl IndexingMetrics {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            end: Instant::now(),
            insert_durations: Vec::new(),
            build_durations: Vec::new(),
            nb_vectors: Vec::new(),
            database_size: Vec::new(),
        }
    }

    pub fn start_insertion(&mut self) {
        self.insert_durations.push((Instant::now(), Instant::now()));
    }

    pub fn end_insertion(&mut self) {
        self.insert_durations.last_mut().unwrap().1 = Instant::now();
    }

    pub fn start_building(&mut self) {
        self.build_durations.push((Instant::now(), Instant::now()));
    }

    pub fn end_building(&mut self) {
        self.build_durations.last_mut().unwrap().1 = Instant::now();
    }

    pub fn new_nb_vectors(&mut self, nb_vectors: usize) {
        self.nb_vectors.push(nb_vectors);
    }

    pub fn new_database_size(&mut self, size: usize) {
        self.database_size.push(size);
    }

    pub fn end(&mut self) {
        self.end = Instant::now();
    }
}

impl fmt::Display for IndexingMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Total time to index: {:.2?} (", self.end.duration_since(self.start))?;

        for (idx, ((insert_start, insert_end), (build_start, build_end))) in
            self.insert_durations.iter().zip(self.build_durations.iter()).enumerate()
        {
            if idx != 0 {
                write!(f, " + ")?;
            }
            write!(
                f,
                "{:.2?}",
                insert_end.duration_since(*insert_start) + build_end.duration_since(*build_start)
            )?;
        }
        writeln!(f, ")")?;

        // First step is to format all the lists in a vector of strings

        let vectors = self.nb_vectors.iter().map(|v| format!("{}", v)).collect::<Vec<_>>();
        let insertions = self
            .insert_durations
            .iter()
            .map(|(insert_start, insert_end)| {
                format!("{:.2?}", insert_end.duration_since(*insert_start))
            })
            .collect::<Vec<_>>();
        let builds = self
            .build_durations
            .iter()
            .map(|(build_start, build_end)| {
                format!("{:.2?}", build_end.duration_since(*build_start))
            })
            .collect::<Vec<_>>();
        let db_size = self
            .database_size
            .iter()
            .map(|v| {
                format!(
                    "{:.2}",
                    Byte::from_decimal_with_unit(Decimal::from(*v), Unit::B)
                        .unwrap()
                        .get_appropriate_unit(UnitType::Binary)
                )
            })
            .collect::<Vec<_>>();

        // Then we can retrieve the max length of each column in the list to pretty print the table later
        let max_lengths = vectors
            .iter()
            .zip(insertions.iter())
            .zip(builds.iter())
            .zip(db_size.iter())
            .map(|(((v, i), b), d)| {
                [v.len(), i.len(), b.len(), d.len()].into_iter().max().unwrap()
            })
            .collect::<Vec<_>>();

        write!(f, "  => Vectors:    ")?;
        for (idx, (nb_vectors, max_length)) in vectors.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{nb_vectors:>max_length$}")?;
        }
        writeln!(f, "")?;

        write!(f, "  => Insertions: ")?;
        for (idx, (insert, max_length)) in insertions.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{insert:>max_length$}")?;
        }
        writeln!(f, "")?;

        write!(f, "  => Builds:     ")?;
        for (idx, (build, max_length)) in builds.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{build:>max_length$}")?;
        }
        writeln!(f, "")?;

        write!(f, "  => Db size:    ")?;
        for (idx, (database_size, max_length)) in db_size.iter().zip(max_lengths.iter()).enumerate()
        {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{database_size:>max_length$}")?;
        }

        Ok(())
    }
}
