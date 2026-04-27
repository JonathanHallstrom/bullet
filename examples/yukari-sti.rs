use std::sync::atomic::{AtomicUsize, Ordering};

use bullet_lib::{
    LocalSettings, TrainingSchedule, TrainingSteps,
    game::{
        inputs::{self, SparseInputType},
        outputs::MaterialCount,
    },
    lr,
    nn::optimiser::{self, AdamW},
    trainer::save::SavedFormat,
    value::{ValueTrainerBuilder, loader},
    wdl,
};
use bullet_trainer::run::logger;
use bulletformat::ChessBoard;
use montyformat::chess::{Attacks, Piece, Side};
use viriformat::dataformat::Filter;

fn main() {
    logger::set_cbcs(true);

    const L1_SIZE: usize = 128;
    const SCALE: f32 = 400.0;
    const QA: i16 = 255;
    const QB: i16 = 64;
    const OUTPUT_BUCKETS: usize = 8;
    const SUPERBATCHES: usize = 200;

    let inputs = ThreatInputs;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("ftw").round().quantise::<i16>(QA),
            SavedFormat::id("ftb").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // weights
            let ft = builder.new_affine("ft", inputs.num_inputs(), L1_SIZE);
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, OUTPUT_BUCKETS);

            // inference
            let stm_subnet = ft.forward(stm_inputs).screlu();
            let ntm_subnet = ft.forward(ntm_inputs).screlu();
            let ft_out = stm_subnet.concat(ntm_subnet);
            l1.forward(ft_out).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: std::env::args().nth(1).unwrap().to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::LinearDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * 0.3f32.powi(5),
            final_superbatch: SUPERBATCHES,
        },
        save_rate: 200,
    };
    let optimiser_params =
        optimiser::AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };

    trainer.optimiser.set_params(optimiser_params);

    let settings = LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let data_loader =
        loader::ViriBinpackLoader::new("/k4/vine_data/vine_37/mixed_data_chonked.vf", 8192, 8, Filter::default());

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        //println!("{:?}", trainer.optimiser().graph.get_input("stm").get_sparse_vals());
        //println!("{:?}", trainer.optimiser().graph.get_input("nstm").get_sparse_vals());
        println!("FEN:  {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

macro_rules! init_add_assign {
    (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size + 1];
        let mut val = $init;
        while $sq < $size {
            res[$sq] = val;
            val += {$($rest)+};
            $sq += 1;
        }

        res[$size] = val;

        res
    }};
}

pub mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 2 * indices::PAWN;
    pub const BISHOP: usize = KNIGHT + 2 * indices::KNIGHT[64];
    pub const ROOK: usize = BISHOP + 2 * indices::BISHOP[64];
    pub const QUEEN: usize = ROOK + 2 * indices::ROOK[64];
    pub const KING: usize = QUEEN + 2 * indices::QUEEN[64];
    pub const END: usize = KING + 2 * indices::KING[64];
}

pub mod indices {
    use super::attacks;

    pub const PAWN: usize = 84;
    pub const KNIGHT: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize);
    pub const BISHOP: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
    pub const KING: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KING[sq].count_ones() as usize);
}

pub mod attacks {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);

    pub const KING: [u64; 64] = init!(|sq, 64| {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & !A) >> 1) | ((k & !H) << 1);
        k ^ (1 << sq)
    });
}

pub fn map_piece_threat(piece: usize, src: usize, dest: usize, enemy: bool) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest, enemy),
        Piece::KNIGHT => map_threat(src, dest, enemy, &indices::KNIGHT, &attacks::KNIGHT, offsets::KNIGHT),
        Piece::BISHOP => map_threat(src, dest, enemy, &indices::BISHOP, &attacks::BISHOP, offsets::BISHOP),
        Piece::ROOK => map_threat(src, dest, enemy, &indices::ROOK, &attacks::ROOK, offsets::ROOK),
        Piece::QUEEN => map_threat(src, dest, enemy, &indices::QUEEN, &attacks::QUEEN, offsets::QUEEN),
        Piece::KING => map_threat(src, dest, enemy, &indices::KING, &attacks::KING, offsets::KING),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_threat(
    src: usize,
    dest: usize,
    enemy: bool,
    indices: &[usize; 65],
    attacks: &[u64; 64],
    offset: usize,
) -> Option<usize> {
    let idx = indices[src] + below(src, dest, attacks);
    let threat = offset + usize::from(enemy) * indices[64] + idx;

    assert!(threat >= offset);
    assert!(threat < offsets::END);

    Some(threat)
}

fn map_pawn_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let up = usize::from(dest > src);
    let diff = dest.abs_diff(src);
    let id = if diff == [9, 7][up] { 0 } else { 1 };
    let attack = 2 * (src % 8) + id - 1;
    let threat = offsets::PAWN + usize::from(enemy) * indices::PAWN + (src / 8 - 1) * 14 + attack;

    assert!(threat < offsets::KNIGHT, "{threat}");

    Some(threat)
}

const BOARD_INPUTS: usize = 768;
const TOTAL_THREATS: usize = 2 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + BOARD_INPUTS;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static SQRED: AtomicUsize = AtomicUsize::new(0);
static EVALS: AtomicUsize = AtomicUsize::new(0);
static MAX: AtomicUsize = AtomicUsize::new(0);
const TRACK: bool = false;

pub fn print_feature_stats() {
    let count = COUNT.load(Ordering::Relaxed);
    let sqred = SQRED.load(Ordering::Relaxed);
    let evals = EVALS.load(Ordering::Relaxed);
    let max = MAX.load(Ordering::Relaxed);

    let mean = count as f64 / evals as f64;
    let var = sqred as f64 / evals as f64 - mean.powi(2);
    let pct = 1.96 * var.sqrt();

    println!("Total Evals: {evals}");
    println!("Maximum Active Features: {max}");
    println!("Active Features: {mean:.3} +- {pct:.3} (95%)");
}

fn map_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                f([0, 384][side] + 64 * (piece - 2) + sq);
                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(piece, sq, dest, enemy) {
                        f(BOARD_INPUTS + side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        "768hm+TI".to_string()
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}
