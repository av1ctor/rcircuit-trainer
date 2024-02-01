mod replay_buffer;
mod dqn;
mod agent;
mod base_agent;
mod dqnagent;
mod ddqn_agent;
mod gl_renderer;

use std::collections::HashSet;
use std::f32::consts::PI;
use agent::Agent;
use base_agent::BaseAgentOptions;
use ddqn_agent::DdqnAgent;
use brains_engine::{
    asset::GameAssetType, 
    env::{Asset, AssetData, AssetValue, Env}, 
    game::{Game, GameOptions, LevelOptions, ObjectFOV}, 
    map::Map, 
    renderer::Renderer
};
use dqn::DqnOptions;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use sdl2::keyboard::Scancode;
use tch::{Tensor, Device};
use gl_renderer::GlRenderer;
use three_rs::{core::Transformable, renderer::Event};

const LEARNING: bool = false;
const NUM_PLAYERS: usize = if LEARNING {1} else {4};
const SCREEN_W: u32 = 1000;
const SCREEN_H: u32 = 600;
const FPS: f32 = 60.0; // frames per sec
const DELTA_TIME: f32 = if LEARNING {1.0} else {60.0 / FPS};
const RAYS: usize = 15;
const FOV: f32 = 180.0 * (PI / 180.0);
const PLAYER_RADIUS: f32 = 0.5;

const REPLAY_BUFFER_CAPACITY: usize = 15_000;
const TRAINING_BATCH_SIZE: usize = 32;
const MAX_EPISODES: usize = 40;
const LEARNING_RATE: f64 = 0.0005;
const REPLACE_TARGET_CNT: usize = 128;
const GAMMA: f64 = 0.99;
const EPSILON: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 1e-5;
const NUM_HIDDEN: usize = 32;
const MAX_STEPS: usize = 10000;

struct Player {
    last_thought: u32,
    steps_per_thought: u32,
    obs: Tensor,
}

fn main(
) -> Result<(), String> {
    let tensors_path = format!("./weights/{}rays-{}hidden-{}epsilon.safetensors", 
        RAYS, NUM_HIDDEN, EPSILON_MIN);
    let mut rng = StdRng::seed_from_u64(1234);
    let mut _pov = 0;

    let map_suffix = if LEARNING {"-training"} else {""};
    let map_path = format!("./maps/level-0/map{}.glb", map_suffix);

    let (level_bytes, navmesh_bytes) = if let Ok(level_bytes) = 
        std::fs::read(&map_path) {
        if let Ok(navmesh_bytes) = 
            std::fs::read(format!("./maps/level-0/navmesh{}.bin", map_suffix)) {
            (level_bytes, navmesh_bytes)
        }
        else {
            panic!("Error loading level-0's navmesh");
        }
    }
    else {
        panic!("Error loading level-0's model");
    };

    let mut renderer = GlRenderer::new(SCREEN_W, SCREEN_H)
        .expect("Renderer initialization failed");

    let mut game = Game::new(
        Map::new(
            &level_bytes.to_vec(), 
            &navmesh_bytes.to_vec()
        ),
        GameOptions {
            level: LevelOptions {
                num: 0
            },
            fov: ObjectFOV { 
                rays: RAYS, 
                angle: FOV, 
                max_dist: 2.5,
            },
            num_players: NUM_PLAYERS,
            min_keys: 4,
            player_radius: PLAYER_RADIUS,
            debug: LEARNING,
            training: LEARNING,
            max_steps: MAX_STEPS,
        }
    );

    let options = BaseAgentOptions { 
        dqn: DqnOptions {
            learning_rate: LEARNING_RATE,
            n_actions: game.get_num_actions(),
            n_obs: game.get_num_obs(),
            n_hidden: NUM_HIDDEN
        },
        gamma: GAMMA,
        epsilon: if LEARNING {EPSILON} else {EPSILON_MIN},
        epsilon_dec: if LEARNING {EPSILON_DECAY} else {0.0},
        epsilon_min: if LEARNING {EPSILON_MIN} else {EPSILON_MIN},
        batch_size: TRAINING_BATCH_SIZE,
        replay_buffer_capacity: REPLAY_BUFFER_CAPACITY,
        replace_target_cnt: REPLACE_TARGET_CNT,
    };
    
    for asset in game.get_assets() {
        let asset = if asset.typ == GameAssetType::Level as usize {
            Asset::new(
                asset.typ, 
                AssetData::new(&[
                    (&"path", &AssetValue::Text(map_path.clone()))
                ])
            )
        }
        else {
            asset
        };

        renderer.load_asset(&asset)
            .expect(&format!("Asset not found: {}", asset.typ));
    }

    let mut agent = DdqnAgent::new(&options);

    if !LEARNING {
        agent.load(tensors_path.to_string())?;
    }

    let mut players = vec![];
    for _ in 0..NUM_PLAYERS {
        players.push(Player {
            last_thought: 0,
            steps_per_thought: if LEARNING { 
                1000 
            } 
            else {
                rng.gen_range(1000..10000)
            },
            obs: Tensor::default(),
        });
    }

    let mut best_win_rate = 0.0;
    let mut sample = 10;
    let mut results: Vec<usize> = vec![];

    let mut eps = 0;
    'main: while eps < MAX_EPISODES {
        let mut score = 0.0f32;

        let mut steps = 0;
        let mut running = true;

        let obs = &game.reset(&mut renderer, &mut rng);
        for i in 0..game.options.num_players {
            players[i].last_thought = 0;
            players[i].obs = Tensor::from_slice(&obs[i]).to(Device::Cuda(0));
        }

        game.render(&mut renderer, false);

        while running {
            let events = renderer.poll_events();
            for event in events {
                match event {
                    Event::Quit => {
                        break 'main;
                    },
                    Event::KeyDown(scancode) => {
                        if scancode == Scancode::Escape as usize {
                            break 'main;
                        }
                        else if scancode as i32 >= Scancode::Num1 as i32 && 
                            scancode as i32 <= Scancode::Num9 as i32 {
                            _pov = ((scancode as i32 - Scancode::Num1 as i32) as usize) % 
                                game.options.num_players;
                        }
                        else if scancode == Scancode::Left as usize {
                            renderer.camera.translate_x(-10.0);
                        }
                        else if scancode == Scancode::Right as usize {
                            renderer.camera.translate_x(10.0);
                        }
                        else if scancode == Scancode::Up as usize {
                            renderer.camera.translate_y(10.0);
                        }
                        else if scancode == Scancode::Down as usize {
                            renderer.camera.translate_y(-10.0);
                        }
                        else if scancode == Scancode::KpPlus as usize {
                            renderer.camera.translate_z(-10.0);
                        }
                        else if scancode == Scancode::KpMinus as usize {
                            renderer.camera.translate_z(10.0);
                        }
                        else if scancode == Scancode::Q as usize {
                            renderer.camera.rotate_y(-0.1);
                        }
                        else if scancode == Scancode::W as usize {
                            renderer.camera.rotate_y(0.1);
                        }
                    }
                }
            }

            let mut actions = vec![None; game.options.num_players];

            let players_idx = gen_players_idx(&mut game, &mut rng);
            
            for i in 0..game.options.num_players {
                let p = players_idx[i];
                let player = &mut players[p];
                
                player.last_thought += 1000;
                if player.last_thought >= player.steps_per_thought {
                    player.last_thought = 0;

                    let action_ts = agent.choose_action(&player.obs, &mut rng)
                        .unwrap_or(Tensor::from(game.random_action(&mut rng) as i64)
                            .to(Device::Cuda(0)));

                    let mut action = vec![0i64; 1];
                    action_ts.copy_data(&mut action, 1);
                    actions[p] = Some(action[0] as usize);
                }
            }

            let step = game.step(&actions, DELTA_TIME, &mut renderer);
            steps += 1;

            running = !step.terminated && !step.timedout;
            if !running {
                results.push(if step.winner.is_some() {1} else {0});
            }

            if LEARNING {
                score += step.players[0].reward;
            }

            for i in 0..step.players.len() {
                let obs = &step.players[i].obs;
                let next_obs_ts = Tensor::from_slice(obs).to(Device::Cuda(0));

                if LEARNING {
                    if players[0].last_thought == 0 {
                        let action_ts = Tensor::from_slice(&[actions[0].unwrap() as i64])
                            .to(Device::Cuda(0));

                        agent.store_transition(
                            &players[0].obs, 
                            &action_ts, 
                            &step.players[0].reward.into(), 
                            &next_obs_ts, 
                            &step.terminated.into(),
                            &step.timedout.into()
                        );

                        agent.learn(&mut rng);
                    }
                }

                players[i].obs = next_obs_ts;
            }

            game.render(&mut renderer, steps % if LEARNING {12} else {6} != 0);            
        }

        let win_rate = results.iter().rev().take(sample).sum::<usize>() as f32 / (sample as f32);

        if win_rate > best_win_rate {
            best_win_rate = win_rate;
            if LEARNING {
                if best_win_rate >= 0.9 {
                    if agent.get_epsilon() <= EPSILON_MIN {
                        agent.save(tensors_path.to_string())?;
                    }
                    if best_win_rate == 1.0 {
                        sample *= 2;
                        best_win_rate /= 2.0;
                    }
                }
            }
        }

        println!("ep: {}, steps: {}, score: {:.0}, epsilon: {:.2}, win rate: {:.1}%, best rate: {:.1}%", 
            eps, steps, score, agent.get_epsilon(), win_rate * 100.0, best_win_rate * 100.0);

        eps += 1;
    }

    Ok(())
}

fn gen_players_idx(
    game: &mut Game,
    rng: &mut StdRng
) -> Vec<usize> {
    let mut indexes = HashSet::new();

    while indexes.len() < game.options.num_players {
        indexes.insert(rng.gen_range(0..game.options.num_players));
    }

    indexes.into_iter().collect()
}