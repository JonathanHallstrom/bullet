use std::sync::atomic::{AtomicU64, Ordering};

use rand::{Rng, rng};
use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, WDL},
};

fn piece_count_acceptance(board: &Board) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    static PIECE_COUNT_STATS: [AtomicU64; 33] = [const { AtomicU64::new(0) }; 33];
    static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;

    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

pub fn should_keep(entry: &Board, m: Move, eval: i16, wdl: f32) -> bool {
    const DEFAULT_VIRI_FILTER: Filter = Filter {
        min_ply: 16,
        min_pieces: 0,
        filter_tactical: true,
        filter_check: true,
        filter_castling: true,
        max_eval: 16000,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.90,

        wdl_filtered: false,

        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.0,
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    !DEFAULT_VIRI_FILTER.should_filter(m, eval.into(), entry, wdl, &mut rng)
        && rng.random_bool(piece_count_acceptance(entry))
}
