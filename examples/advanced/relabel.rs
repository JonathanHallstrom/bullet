mod inputs;
mod net;

use std::{
    env,
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

use bullet_lib::{game::formats::bulletformat::ChessBoard, trainer::schedule::wdl};
use bullet_trainer::{
    model::{ModelEvaluator, ModelWeights},
    run::{DefaultDevice, Step},
};
use viriformat::{chess::piece::Colour, dataformat::Game};

const BATCH_SIZE: usize = 16384;
const GAMES_PER_ITER: usize = 16;
const SCALE: f32 = 400.0;
const EVAL_CLAMP: i32 = 32767;

fn gpu_error(err: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{err:?}"))
}

fn stored_eval(output: f32, black_to_move: bool) -> i16 {
    let score = (SCALE * output).round() as i32;
    let score = if black_to_move { -score } else { score };
    score.clamp(-EVAL_CLAMP, EVAL_CLAMP) as i16
}

fn relabel(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    mut evaluate: impl FnMut(&[ChessBoard], &mut Vec<f32>) -> io::Result<()>,
) -> io::Result<(usize, usize)> {
    let mut games = Vec::new();
    let mut boards = Vec::new();
    let mut scores = Vec::new();
    let mut move_buffers = Vec::new();
    let mut output_moves = Vec::new();
    let mut game_count = 0;
    let mut position_count = 0;

    loop {
        let mut got_any_games = false;
        for _ in 0..GAMES_PER_ITER {
            // break if we hit the end of the file
            if reader.fill_buf()?.is_empty() {
                break;
            }
            let game = Game::deserialise_from(reader, move_buffers.pop().unwrap_or_default())?;
            game.splat_to_bulletformat_with_filter_callback(
                |board| {
                    boards.push(board);
                    Ok(())
                },
                |_, _, _, _, _| false,
            )
            .map_err(io::Error::other)?;
            games.push(game);
            got_any_games = true;
        }
        let padding;
        if got_any_games {
            padding = 0;
        } else {
            let positions = boards.len();
            padding = positions.next_multiple_of(BATCH_SIZE) - positions;
            boards.resize(positions + padding, Default::default());
        }

        let evaluated = boards.len() / BATCH_SIZE * BATCH_SIZE;
        for batch in boards[..evaluated].chunks(BATCH_SIZE) {
            let expected = scores.len() + BATCH_SIZE;
            evaluate(batch, &mut scores)?;
            assert_eq!(scores.len(), expected);
        }
        scores.truncate(scores.len() - padding);
        boards.drain(..evaluated);

        let mut ready_games = 0;
        let mut ready_scores = 0;
        for game in &games {
            if ready_scores + game.len() > scores.len() {
                break;
            }
            ready_games += 1;
            ready_scores += game.len();
        }
        {
            let mut ready = scores.drain(..ready_scores);
            for game in games.drain(..ready_games) {
                let mut black_to_move = game.initial_position().turn() == Colour::Black;
                output_moves.clear();
                let mut relabelled = Game { initial_position: game.initial_position, moves: output_moves };
                for mv in game.moves() {
                    relabelled.add_move(mv, stored_eval(ready.next().unwrap(), black_to_move));
                    black_to_move = !black_to_move;
                }
                relabelled.serialise_into(writer)?;
                output_moves = relabelled.into_move_buffer();
                move_buffers.push(game.into_move_buffer());
            }
        }
        game_count += ready_games;
        position_count += ready_scores;

        if !got_any_games {
            assert!(games.is_empty() && scores.is_empty());
            return Ok((game_count, position_count));
        }
    }
}

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = args.next().ok_or_else(|| io::Error::other("missing checkpoint directory"))?;
    let input = args.next().ok_or_else(|| io::Error::other("missing input vf file"))?;
    let output = args.next().ok_or_else(|| io::Error::other("missing output vf file"))?;

    let (model_inputs, pp, psqt, output_buckets, defn) = net::build();
    let mut weights = ModelWeights::zeroed(&defn);
    let weights_path = Path::new(&checkpoint).join("optimiser_state/weights.bin");
    weights.load_from(BufReader::new(File::open(weights_path)?))?;

    let device = DefaultDevice::new(0).map_err(gpu_error)?;
    let mut evaluator = ModelEvaluator::with_batch_size(&defn, device.clone(), BATCH_SIZE).map_err(gpu_error)?;
    evaluator.load_weights(&weights).map_err(gpu_error)?;
    let mapper =
        inputs::make_inputs_mapper((&model_inputs, &pp, psqt, output_buckets), wdl::ConstantWDL { value: 0.0 });

    let mut reader = BufReader::new(File::open(input)?);
    let output = OpenOptions::new().write(true).create_new(true).open(output)?;
    let mut writer = BufWriter::new(output);
    let start = Instant::now();
    let (game_count, position_count) = relabel(&mut reader, &mut writer, |batch, scores| {
        let inputs = mapper.map(batch, Step::default(), 1).to_device(&device).map_err(gpu_error)?;
        let outputs = evaluator.evaluate(&inputs).map_err(gpu_error)?;
        let output = outputs["output"].to_host().map_err(gpu_error)?;
        scores.extend_from_slice(output.f32());
        Ok(())
    })?;

    writer.flush()?;
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "relabelled {game_count} games, {position_count} positions in {elapsed:.2}s ({:.0} pos/s)",
        position_count as f64 / elapsed,
    );
    Ok(())
}
