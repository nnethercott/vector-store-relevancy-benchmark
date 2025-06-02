#![allow(clippy::type_complexity)]

pub mod arroy_bench;
mod dataset;
mod qdrant_bench;
pub mod scenarios;

use core::fmt;

use arroy::distances::*;
use arroy::internals::UnalignedVector;
use arroy::ItemId;
pub use dataset::*;
use fast_distances::hamming;
use qdrant_client::qdrant::quantization_config;

pub const RNG_SEED: u64 = 38;

/// A generalist distance trait that contains the informations required to configure every engine
pub trait Distance {
    const BINARY_QUANTIZED: bool;
    const QDRANT_DISTANCE: qdrant_client::qdrant::Distance;
    type ArroyDistance: arroy::Distance;

    fn name() -> &'static str;
    fn qdrant_quantization_config() -> quantization_config::Quantization;
    fn real_distance(a: &[f32], b: &[f32]) -> f32;
}

macro_rules! arroy_distance {
    ($distance:ty => real: $real:ident, qdrant: $qdrant:ident, bq: $bq:expr) => {
        impl Distance for $distance {
            const BINARY_QUANTIZED: bool = $bq;
            const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
                qdrant_client::qdrant::Distance::$qdrant;
            type ArroyDistance = $distance;

            fn name() -> &'static str {
                stringify!($distance)
            }
            fn qdrant_quantization_config() -> quantization_config::Quantization {
                qdrant_client::qdrant::BinaryQuantization::default().into()
            }
            fn real_distance(a: &[f32], b: &[f32]) -> f32 {
                let a = ndarray::aview1(a);
                let b = ndarray::aview1(b);
                fast_distances::$real(&a, &b)
            }
        }
    };
}

arroy_distance!(BinaryQuantizedCosine => real: cosine, qdrant: Cosine, bq: true);
arroy_distance!(Cosine =>  real: cosine, qdrant: Cosine, bq: false);
arroy_distance!(BinaryQuantizedEuclidean => real: euclidean, qdrant: Euclid, bq: true);
arroy_distance!(Euclidean => real: euclidean, qdrant: Euclid, bq: false);
arroy_distance!(BinaryQuantizedManhattan => real: manhattan, qdrant: Manhattan, bq: true);
arroy_distance!(Manhattan => real: manhattan, qdrant: Manhattan, bq: false);
// arroy_distance!(DotProduct => real: dot, qdrant: Dot);
impl Distance for Hamming {
    const BINARY_QUANTIZED: bool = false;
    const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
        qdrant_client::qdrant::Distance::Manhattan;
    type ArroyDistance = Hamming;

    fn name() -> &'static str {
        stringify!($distance)
    }
    fn qdrant_quantization_config() -> quantization_config::Quantization {
        qdrant_client::qdrant::BinaryQuantization::default().into()
    }
    fn real_distance(a: &[f32], b: &[f32]) -> f32 {
        // manually quantize
        let a: Vec<f32> = a.iter().map(|&val| if val>0.0 {1.0} else {0.0}).collect();
        let b: Vec<f32> = b.iter().map(|&val| if val>0.0 {1.0} else {0.0}).collect();

        let a = ndarray::aview1(&a);
        let b = ndarray::aview1(&b);
        fast_distances::hamming(&a, &b) as f32
    }
}

fn partial_sort_by<'a, D: crate::Distance>(
    mut vectors: impl Iterator<Item = (ItemId, &'a [f32])>,
    sort_by: &[f32],
    elements: usize,
) -> Vec<(ItemId, &'a [f32], f32)> {
    let mut ret = Vec::with_capacity(elements);
    ret.extend(vectors.by_ref().take(elements).map(|(i, v)| (i, v, distance::<D>(sort_by, v))));
    ret.sort_by(|(_, _, left), (_, _, right)| left.total_cmp(right));

    if ret.is_empty() {
        return ret;
    }

    for (item_id, vector) in vectors {
        let distance = distance::<D>(sort_by, vector);
        if distance < ret.last().unwrap().2 {
            match ret.binary_search_by(|(_, _, d)| d.total_cmp(&distance)) {
                Ok(i) | Err(i) => {
                    ret.pop();
                    ret.insert(i, (item_id, vector, distance))
                }
            }
        }
    }

    ret
}

pub fn distance<D: crate::Distance>(left: &[f32], right: &[f32]) -> f32 {
    D::real_distance(left, right)
}

pub struct Recall(pub f32);

impl fmt::Debug for Recall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            // red
            f32::NEG_INFINITY..=0.25 => write!(f, "\x1b[1;31m")?,
            // yellow
            0.25..=0.5 => write!(f, "\x1b[1;33m")?,
            // green
            0.5..=0.75 => write!(f, "\x1b[1;32m")?,
            // blue
            0.75..=0.90 => write!(f, "\x1b[1;34m")?,
            // cyan
            0.90..=0.999 => write!(f, "\x1b[1;36m")?,
            // underlined cyan
            0.999..=f32::INFINITY => write!(f, "\x1b[1;4;36m")?,
            _ => (),
        }
        write!(f, "{:.2}\x1b[0m", self.0)
    }
}
