use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bullet_lib::{
    game::{
        formats::{
            bulletformat::ChessBoard,
            montyformat::chess::{Attacks, Piece, Side},
        },
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
    },
    nn::InitSettings,
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::{ViriBinpackLoader, viribinpack::ViriFilter},
        save::save_to_checkpoint,
    },
    wdl::WdlScheduler,
};
use bullet_trainer::{
    model::{
        DenseInput, ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat,
        SparseInput,
    },
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, HostPool, TrainingSchedule, TrainingSteps, train},
};
use rand::{Rng, rng, seq::SliceRandom};
use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, WDL},
};

const L1_SIZE: usize = 4096;
const L2_SIZE: usize = 128;
const L3_SIZE: usize = 256;
const S0_SUPERBATCHES: usize = 100;
const S1_SUPERBATCHES: usize = 1200;
const S2_SUPERBATCHES: usize = 600;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [usize; 32] = [
     0,  0,  1,  1,
     2,  2,  2,  2,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
];
const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

pub type InputTy = ((((SparseInput, SparseInput), SparseInput), SparseInput), DenseInput<f32>);

pub fn make_inputs_mapper(
    inputs: &ModelInputs<InputTy>,
    pp: PawnPawnInputs,
    psqt: ChessBucketsMirrored,
    wdl: impl WdlScheduler + 'static,
) -> ModelInputsMapper<ChessBoard> {
    ModelInputsMapper::build(inputs, move |pos, step, ((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), target)| {
        let mut cnt = 0;
        psqt.map_features(pos, |stm, ntm| {
            stm_psqt[cnt] = stm.try_into().unwrap();
            ntm_psqt[cnt] = ntm.try_into().unwrap();
            cnt += 1;
        });

        if cnt < psqt.max_active() {
            stm_psqt[cnt] = -1;
            ntm_psqt[cnt] = -1;
        }

        let mut stm_cnt = 0;
        let mut ntm_cnt = 0;

        pp.map_features(
            pos,
            |stm| {
                stm_pp[stm_cnt] = stm.try_into().unwrap();
                stm_cnt += 1;
            },
            |ntm| {
                ntm_pp[ntm_cnt] = ntm.try_into().unwrap();
                ntm_cnt += 1;
            },
        );

        assert_eq!(stm_cnt, ntm_cnt);

        if stm_cnt < pp.max_active() {
            stm_pp[stm_cnt] = -1;
            ntm_pp[stm_cnt] = -1;
        }

        let result = f32::from(pos.result) / 2.0;
        let score = 1.0 / (1.0 + (f32::from(-pos.score) / 400.0).exp());
        let lambda = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
        target[0] = lambda * result + (1. - lambda) * score;
    })
}

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

fn main() {
    let pp = PawnPawnInputs::new(three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);

    let inputs = ModelInputs::default()
        .add_sparse("stm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(&inputs, |builder, ((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), target)| {
        let l0_pp = builder.new_affine("l0/pp/", pp.num_inputs(), L1_SIZE);

        let l0f = builder.new_weights("l0/fac", (L1_SIZE, 768), InitSettings::Zeroed);

        let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
        let mut l0_psqt = builder.new_weights("l0/psqt", (L1_SIZE, psqt.num_inputs()), psqt_init);
        l0_psqt = l0_psqt + l0f.repeat(psqt.num_inputs() / 768);

        let l1 = builder.new_affine("l1/", L1_SIZE, L2_SIZE);
        let l2 = builder.new_affine("l2/", L2_SIZE * 2, L3_SIZE);
        let l3 = builder.new_affine("l3/", L3_SIZE, 1);

        let ft = |pp_in, psqt_in, start, end| {
            (l0_pp.slice(start, end).forward(pp_in) + l0_psqt.slice_rows(start, end).matmul(psqt_in)).crelu()
        };
        let stm_hidden = ft(stm_pp, stm_psqt, 0, L1_SIZE / 2) * ft(stm_pp, stm_psqt, L1_SIZE / 2, L1_SIZE);
        let ntm_hidden = ft(ntm_pp, ntm_psqt, 0, L1_SIZE / 2) * ft(ntm_pp, ntm_psqt, L1_SIZE / 2, L1_SIZE);
        let hl1 = stm_hidden.concat(ntm_hidden);

        let l0_out_norm = hl1.reduce_sum_rows() / (L1_SIZE as f32);

        let l1_out = l1.forward(hl1);
        let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

        let hl3 = l2.forward(hl2).screlu();
        let out = l3.forward(hl3);

        let loss = out.sigmoid().squared_error(target);
        let loss = loss + 0.005 * l0_out_norm;

        (Some(loss.reduce_sum_batch()), vec![("output".to_string(), out)])
    });

    let weights = ModelWeights::new(&defn, 12412421);
    let device = DefaultDevice::new(0).unwrap();

    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let default_params = AdamWParams { ..AdamWParams::default() };
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), default_params).unwrap();

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..default_params };
    optimiser.set_params_for_weight("l0/fac", stricter_clipping);
    optimiser.set_params_for_weight("l0/psqt", stricter_clipping);
    optimiser.set_params_for_weight("l1/w", stricter_clipping);
    let ti_clipping = AdamWParams { max_weight: 0.49, min_weight: -0.49, ..default_params };
    optimiser.set_params_for_weight("l0/pp/w", ti_clipping);

    let saved_format = vec![
        SavedFormat::id("l0/psqt").transform(|weights, values| {
            let fac = weights.get("l0/fac").values.f32().repeat(NUM_INPUT_BUCKETS);
            assert_eq!(values.len(), fac.len());
            values.iter().zip(fac).map(|(&a, b)| a + b).collect()
        }),
        SavedFormat::id("l0/pp/w"),
        SavedFormat::id("l0/pp/b"),
        SavedFormat::id("l1/w").transpose(),
        SavedFormat::id("l1/b"),
        SavedFormat::id("l2/w").transpose(),
        SavedFormat::id("l2/b"),
        SavedFormat::id("l3/w").transpose(),
        SavedFormat::id("l3/b"),
    ];

    let reader = {
        let buffer_size_mb = 16384;
        let threads = 16;
        fn filter(entry: &viriformat::chess::board::Board, m: Move, eval: i16, wdl: f32) -> bool {
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

        let dataset_paths = [
            "/k4/quant_data/pgns/vfdata/new_data/shuf/mix_*.vf",
            "/k4/quant_data/pgns/vfdata/new_data/7322_relabeled/*.vf",
        ]
        .iter()
        .flat_map(|pattern| glob::glob(pattern).expect("successfully found dataset"))
        .map(|f| f.unwrap())
        .collect::<Vec<_>>();
        let dataset_filenames = dataset_paths.iter().map(|f| f.to_str().unwrap()).collect::<Vec<_>>();
        ViriBinpackLoader::new_interleave_multiple(
            &dataset_filenames,
            buffer_size_mb,
            threads,
            ViriFilter::Custom(filter),
        )
    };

    let run = |stage, start_superbatch, end_superbatch, lr_schedule, mapper, optimiser: &mut Optimiser<_, AdamW<_>>| {
        train(
            optimiser,
            TrainingSchedule {
                steps: TrainingSteps {
                    batch_size: 16_384,
                    batches_per_superbatch: 6104,
                    start_superbatch,
                    end_superbatch,
                },
                lr_schedule,
                log_rate: 128,
            },
            ReadMapLoader::new(reader.clone(), mapper, 8),
            |_, _, _| {},
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(100) || superbatch == step.final_superbatch() {
                    let name = format!("quant_tipp_large_stage{stage}-{superbatch}");
                    save_to_checkpoint(optimiser, &saved_format, &format!("checkpoints/{name}"));
                    println!("Saved [{name}]");
                }
            },
        )
        .unwrap();
    };

    let initial_lr = 1e-3;
    let final_lr = 1e-3 * 0.3f32.powi(4);
    let cosine_lr = |sbs, initial, final_lr| lr::CosineDecayLR { initial_lr: initial, final_lr, final_superbatch: sbs };
    run(
        0,
        1,
        S0_SUPERBATCHES,
        cosine_lr(S0_SUPERBATCHES, 1e-5, 1e-4).boxed(),
        make_inputs_mapper(&inputs, pp.clone(), psqt, wdl::LinearWDL { start: 0.0, end: 0.0 }),
        &mut optimiser,
    );

    run(
        1,
        1,
        S1_SUPERBATCHES,
        cosine_lr(S1_SUPERBATCHES, initial_lr, final_lr).boxed(),
        make_inputs_mapper(&inputs, pp.clone(), psqt, wdl::LinearWDL { start: 0.25, end: 0.75 }),
        &mut optimiser,
    );

    run(
        2,
        1,
        S2_SUPERBATCHES,
        cosine_lr(S2_SUPERBATCHES, final_lr, final_lr / 20.0).boxed(),
        make_inputs_mapper(&inputs, pp.clone(), psqt, wdl::LinearWDL { start: 1.0, end: 1.0 }),
        &mut optimiser,
    );

    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let pool = HostPool::new(device.clone());
    let evaluator_mapper = make_inputs_mapper(&inputs, pp.clone(), psqt, wdl::ConstantWDL { value: 0.0 });

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
        "3N4/b2R2p1/3q3r/6P1/4k1nQ/7B/8/K7 w - - 0 1",
        "k2B1Q1q/8/b7/4p3/3Pr3/1N5R/2n5/1K6 w - - 0 1",
        "1B3q2/8/r5n1/8/Rp1N1PQ1/8/4bk2/2K5 w - - 0 1",
        "8/5NR1/5q1b/8/7p/3P2B1/6Q1/1k1K1n1r w - - 0 1",
        "8/8/6r1/4B3/3Q3p/N1nq4/5RP1/b3K2k b - - 0 1",
        "3qn2Q/1R6/8/1N3b1p/4B3/1kP5/r7/5K2 b - - 0 1",
        "3rBR2/2qQ1p2/N7/2P2b2/6n1/k7/8/6K1 b - - 0 1",
        "k7/8/p1rB1q2/7Q/4R3/2N2n2/7P/6bK b - - 0 1",
        "2n2Rr1/Bk5p/N7/2Q3q1/b7/8/KP6/8 w - - 0 1",
        "8/Q6r/3qR1P1/b4p2/k7/3B4/1KN2n2/8 b - - 0 1",
        "2nR4/1qB5/2p5/7r/4bQ2/1P1N4/2K1k3/8 w - - 0 1",
        "8/2Q1B3/n3qR1r/bk1p4/1P6/8/3K4/7N w - - 0 1",
        "7r/4b3/4k1N1/2q4n/1Q2B3/R5p1/1P2K3/8 b - - 0 1",
        "2r1n1k1/NbR5/6B1/2p1P3/8/8/5K2/q6Q b - - 0 1",
        "2Q2R2/P1pn4/q1N5/1b5k/1r6/B7/6K1/8 b - - 0 1",
        "1Nr2b2/R1p5/5q2/7B/2P5/3nk3/7K/1Q6 w - - 0 1",
        "4Q3/6P1/1k3p2/4N3/2r5/K6b/1n1B2Rq/8 b - - 0 1",
        "1B5Q/1n6/2p1rN2/3R4/3P4/1K3k2/3b4/6q1 w - - 0 1",
        "3n4/3q4/5Q2/4rP2/1N2p3/2K2B2/5k2/2b4R b - - 0 1",
        "6B1/2k5/2n1R3/1q2p3/2P4Q/3K4/r5b1/3N4 w - - 0 1",
        "8/8/b6N/R3pr1n/Q7/1Pk1K3/4B3/5q2 b - - 0 1",
        "3Q2r1/4P2R/1b6/8/8/1B3K2/4p2q/1k1n1N2 w - - 0 1",
        "bR5q/2r3B1/2Q1P3/8/2n5/1N1p2K1/k7/8 w - - 0 1",
        "1q1b2r1/8/8/2p5/4N3/3k1P1K/2nB1Q2/4R3 w - - 0 1",
        "5rRq/8/1Qn5/8/K7/P1B4b/1p2N3/7k w - - 0 1",
        "1n6/8/B3q3/5R2/1KPb2N1/7Q/r4p2/2k5 w - - 0 1",
        "q3N1R1/8/1B5n/2p5/2K2P2/7r/1b1k4/7Q w - - 0 1",
        "1B6/N6q/2b5/7R/P2K4/1Q1pr3/6n1/2k5 b - - 0 1",
        "1R3q2/p3Q1n1/4N3/6r1/4K1B1/2P5/7b/4k3 w - - 0 1",
        "1k6/2RQP3/1p6/b7/1B3K2/r1n5/3Nq3/8 b - - 0 1",
        "b7/k7/5P2/n2N4/5pK1/2q5/2B2R2/r4Q2 b - - 0 1",
        "1B6/P4q2/5r2/8/1k2n2K/5b2/1NR1p3/6Q1 w - - 0 1",
        "8/3Pk3/B2r4/K5N1/b7/3n1p1Q/2R5/5q2 w - - 0 1",
        "q2Q1R2/2p4N/1b1P4/1K6/1B3r2/8/8/n2k4 w - - 0 1",
        "n1k5/5pq1/R4b2/2K5/3N4/7P/4BrQ1/8 b - - 0 1",
        "8/4Q3/B7/3KN1P1/3b4/nk3p2/8/R4r1q w - - 0 1",
        "b6n/B1k5/8/4KN1r/1Q6/7R/6Pp/5q2 b - - 0 1",
        "6k1/7r/8/bB3K1N/1R1q4/4Q3/2nP1p2/8 w - - 0 1",
        "Q6R/8/2B1q3/3N1nK1/2kb4/P7/r6p/8 w - - 0 1",
        "8/p5r1/k7/6PK/3b4/2B5/n4qQ1/3N2R1 w - - 0 1",
        "4kb2/6r1/K7/p7/6n1/2N5/2BP1qR1/7Q w - - 0 1",
        "6q1/1BN5/1K3P2/3br1np/3R4/Q7/8/5k2 w - - 0 1",
        "5n2/5q2/1NK5/k1P3r1/3p4/7Q/B6b/1R6 w - - 0 1",
        "B3r3/3p4/N2K2k1/1Q6/2R5/1bP5/1q5n/8 w - - 0 1",
        "BR2Q3/4N3/1n2K3/k7/1p1b1q2/8/5P2/7r b - - 0 1",
        "1k6/7R/5K1N/1pQ5/1n6/P4b2/1r6/6qB b - - 0 1",
        "8/3k4/3NnPK1/3QR3/3r2pB/8/4b3/q7 w - - 0 1",
        "1Q6/4q3/NB5K/1R1r4/3P4/bp1k4/6n1/8 w - - 0 1",
        "3Br3/K7/2q1N3/7n/8/4PbRQ/1p1k4/8 w - - 0 1",
        "R2r4/pK1b4/1n4NB/7P/8/3Q4/6k1/4q3 b - - 0 1",
        "3N2r1/2KP4/8/1B1p4/2b5/3RQq2/2k5/7n w - - 0 1",
        "5q2/1N1KB3/5b2/p4R2/4k3/P7/Q7/4n1r1 b - - 0 1",
        "NR6/4K3/1q3r2/3Q3P/3n2k1/8/7p/B5b1 b - - 0 1",
        "q7/1N1B1K2/1Q6/5b2/5pP1/6r1/n6k/R7 w - - 0 1",
        "2R5/2n1k1K1/5r2/3P4/2Q4p/2q5/6NB/7b w - - 0 1",
        "3n1Qr1/3p3K/8/3B4/R5b1/4P3/1qN4k/8 w - - 0 1",
        "K7/3k4/3n2b1/1P2r3/8/p2Bq3/3R4/3QN3 b - - 0 1",
        "1K6/8/3rRN2/1BP3b1/3p4/8/k2n2q1/5Q2 w - - 0 1",
        "2K5/6Bn/p4r2/2P1Q3/1qb5/8/2R5/3kN3 w - - 0 1",
        "3K4/8/2bP4/1qN5/2n3B1/3R4/4Qrp1/6k1 b - - 0 1",
        "1B2K1k1/P3b3/5q2/3R4/1pQ2r1n/8/8/6N1 b - - 0 1",
        "5K2/p4P1b/5QB1/4q3/6k1/8/4r3/R1n1N3 b - - 0 1",
        "6K1/8/b6R/N2p2P1/8/q1Q5/6r1/2Bk3n b - - 0 1",
        "7K/r2R3b/1Q6/8/2q5/1nPB2k1/N3p3/8 w - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let ev_inputs = evaluator_mapper.map(&pool, &[pos], Default::default(), 1).unwrap().to_device(&device).unwrap();
        let output = evaluator.evaluate(&ev_inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * value);
    }
}

pub fn three_file_band_mask() -> [u64; 64] {
    const A: u64 = 0x0101_0101_0101_0101;
    let mut masks = [0; 64];
    for (sq, mask) in masks.iter_mut().enumerate().take(56).skip(8) {
        let f = sq & 7;
        let mut m: u64 = A << f;
        if f > 0 {
            m |= A << (f - 1);
        }
        if f < 7 {
            m |= A << (f + 1);
        }
        *mask = m;
    }
    masks
}

#[derive(Clone)]
pub struct PawnPawnInputs {
    threats: Arc<Threats>,
    masks: [u64; 64],
}

impl PawnPawnInputs {
    pub const TOTAL_PAIRS: usize = 96 * 95 / 2;
    const MAX_PAIRS: usize = 16 * 15 / 2;

    pub fn new(masks: [u64; 64]) -> Self {
        Self { threats: Arc::new(Threats::new()), masks }
    }

    pub fn _total_threats(&self) -> usize {
        self.threats.num_inputs()
    }

    pub fn num_inputs(&self) -> usize {
        Self::TOTAL_PAIRS + self.threats.num_inputs()
    }

    pub fn max_active(&self) -> usize {
        Self::MAX_PAIRS + self.threats.max_active()
    }

    fn pawn_id(colour: usize, sq: usize) -> usize {
        colour * 48 + sq - 8
    }

    fn pair_index(id_a: usize, id_b: usize) -> usize {
        let lo = id_a.min(id_b);
        let hi = id_a.max(id_b);
        hi * (hi - 1) / 2 + lo
    }

    fn emit_same_colour(&self, bb: u64, colour: usize, f: &mut impl FnMut(usize)) {
        let mut outer = bb;
        while outer != 0 {
            let sq_a = outer.trailing_zeros() as usize;
            outer &= outer - 1;
            let id_a = Self::pawn_id(colour, sq_a);
            map_bb(outer & self.masks[sq_a], |sq_b| f(Self::pair_index(id_a, Self::pawn_id(colour, sq_b))));
        }
    }

    fn collect_pairs(&self, bbs: [u64; 8], f: &mut impl FnMut(usize)) {
        let friendly = bbs[Side::WHITE] & bbs[Piece::PAWN];
        let enemy = bbs[Side::BLACK] & bbs[Piece::PAWN];

        self.emit_same_colour(friendly, 0, f);

        map_bb(friendly, |sq_a| {
            let id_a = Self::pawn_id(0, sq_a);
            map_bb(enemy & self.masks[sq_a], |sq_b| f(Self::pair_index(id_a, Self::pawn_id(1, sq_b))));
        });

        self.emit_same_colour(enemy, 1, f);
    }

    pub fn map_features(&self, pos: &ChessBoard, mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
        let bbs = build_bbs(pos);
        self.threats.map(bbs, |stm| on_stm(Self::TOTAL_PAIRS + stm), |ntm| on_ntm(Self::TOTAL_PAIRS + ntm));
        self.collect_pairs(normalize_hm(bbs), &mut on_stm);
        self.collect_pairs(normalize_hm(flip_view(bbs)), &mut on_ntm);
    }
}

pub struct Threats {
    pawn_map: [usize; 12],
    non_pk_data: [PieceThreatsData; 4],
    offsets: [usize; 5],
}

impl Threats {
    pub fn new() -> Self {
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

        let knight = PieceThreatsData::new(
            Piece::KNIGHT,
            [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN],
            |sq| {
                let n = 1 << sq;
                let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
                let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
                (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
            },
        );

        let bishop =
            PieceThreatsData::new(Piece::BISHOP, [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK], |sq| {
                let rank = sq / 8;
                let file = sq % 8;
                DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
            });

        let rook = PieceThreatsData::new(Piece::ROOK, [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK], |sq| {
            let rank = sq / 8;
            let file = sq % 8;
            (0xFF << (rank * 8)) ^ (0x0101_0101_0101_0101 << file)
        });

        let queen = PieceThreatsData::new(
            Piece::QUEEN,
            [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN],
            |sq| bishop.attacks[sq] | rook.attacks[sq],
        );

        let mut offsets = [4 * 84; 5];
        for (i, &cnt) in [10 * knight.count, 8 * bishop.count, 8 * rook.count, 10 * queen.count].iter().enumerate() {
            offsets[i + 1] = offsets[i] + cnt;
        }

        Self {
            pawn_map: make_targets([Piece::KNIGHT, Piece::ROOK]),
            non_pk_data: [knight, bishop, rook, queen],
            offsets,
        }
    }

    pub fn max_active(&self) -> usize {
        128
    }

    pub fn num_inputs(&self) -> usize {
        2 * self.offsets[4]
    }

    pub fn map(&self, bbs: [u64; 8], mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
        let stm_king = (bbs[Side::WHITE] & bbs[Piece::KING]).trailing_zeros() as usize;
        let ntm_king = (bbs[Side::BLACK] & bbs[Piece::KING]).trailing_zeros() as usize;
        let stm_mask = if stm_king % 8 > 3 { 7 } else { 0 };
        let ntm_mask = 56 ^ if ntm_king % 8 > 3 { 7 } else { 0 };

        let mut pieces = [13; 64];
        for side in [Side::WHITE, Side::BLACK] {
            for piece in Piece::PAWN..=Piece::KING {
                let pc = 6 * side + piece - 2;
                map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
            }
        }

        let occ = bbs[0] | bbs[1];

        for side in [Side::WHITE, Side::BLACK] {
            let stm_offset = self.offsets[4] * side;
            let ntm_offset = self.offsets[4] * (side ^ 1);

            for piece in Piece::PAWN..Piece::KING {
                map_bb(bbs[side] & bbs[piece], |sq| {
                    let threats = match piece {
                        Piece::PAWN => Attacks::pawn(sq, side),
                        Piece::KNIGHT => Attacks::knight(sq),
                        Piece::BISHOP => Attacks::bishop(sq, occ),
                        Piece::ROOK => Attacks::rook(sq, occ),
                        Piece::QUEEN => Attacks::queen(sq, occ),
                        _ => unreachable!(),
                    } & occ;

                    map_bb(threats, |dest| {
                        let target = pieces[dest];
                        if let Some(idx) = self.map_single(piece, sq ^ stm_mask, dest ^ stm_mask, target) {
                            on_stm(stm_offset + idx);
                        }

                        let ntm_target = (target + 6) % 12;
                        if let Some(idx) = self.map_single(piece, sq ^ ntm_mask, dest ^ ntm_mask, ntm_target) {
                            on_ntm(ntm_offset + idx);
                        }
                    });
                });
            }
        }
    }

    fn map_single(&self, piece: usize, src: usize, dest: usize, target: usize) -> Option<usize> {
        if piece == Piece::PAWN {
            if self.pawn_map[target] == usize::MAX {
                return None;
            }
            let id = if dest.abs_diff(src) == [9, 7][(dest > src) as usize] { 0 } else { 1 };
            let attack = 2 * (src % 8) + id - 1;
            Some(self.pawn_map[target] * 84 + (src / 8 - 1) * 14 + attack)
        } else {
            self.non_pk_data[piece - 3].map(src, dest, target, self.offsets[piece - 3])
        }
    }
}

pub struct PieceThreatsData {
    piece: usize,
    attacks: [u64; 64],
    indices: [usize; 64],
    targets: [usize; 12],
    count: usize,
}

impl PieceThreatsData {
    pub fn new<const N: usize>(piece: usize, valid: [usize; N], f: impl Fn(usize) -> u64) -> Self {
        let mut attacks = [0; 64];
        let mut indices = [0; 64];
        let mut count = 0;

        for sq in 0..64 {
            attacks[sq] = f(sq);
            indices[sq] = count;
            count += attacks[sq].count_ones() as usize;
        }

        Self { piece, attacks, indices, targets: make_targets(valid), count }
    }

    pub fn map(&self, src: usize, dest: usize, target: usize, offset: usize) -> Option<usize> {
        if self.targets[target] == usize::MAX || (dest > src && target % 6 == self.piece - 2) {
            return None;
        }
        let idx = self.indices[src] + (self.attacks[src] & ((1 << dest) - 1)).count_ones() as usize;
        Some(offset + self.targets[target] * self.count + idx)
    }
}

fn map_bb(mut bb: u64, mut f: impl FnMut(usize)) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn build_bbs(pos: &ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let bit = 1 << sq;
        bbs[usize::from(pc & 8 > 0)] |= bit;
        bbs[2 + usize::from(pc & 7)] |= bit;
    }
    bbs
}

fn flip_view(mut bbs: [u64; 8]) -> [u64; 8] {
    bbs.swap(Side::WHITE, Side::BLACK);
    for bb in bbs.iter_mut() {
        *bb = bb.swap_bytes();
    }
    bbs
}

fn normalize_hm(mut bbs: [u64; 8]) -> [u64; 8] {
    let ksq = (bbs[Side::WHITE] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = bb.swap_bytes().reverse_bits();
        }
    }
    bbs
}

fn make_targets<const N: usize>(valid: [usize; N]) -> [usize; 12] {
    let mut targets = [usize::MAX; 12];
    for i in 0..N {
        targets[valid[i] - 2] = i;
        targets[valid[i] + 4] = i + N;
    }
    targets
}
