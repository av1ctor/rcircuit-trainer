mod replay_buffer;
mod dqn;
mod agent;
mod base_agent;
mod dqnagent;
mod ddqn_agent;
mod three_renderer;

use std::collections::HashSet;
use std::f32::consts::PI;
use agent::Agent;
use base_agent::BaseAgentOptions;
use ddqn_agent::DdqnAgent;
use rcircuit_engine::{
    consts::SEEK_TARGET_SKILL, 
    engine::{asset::GameAssetRef, Game, GameOptions, MapInfo}, 
    entities::PlayerFOV, 
    map::Map
};
use dqn::DqnOptions;
use neuralarena_cdk::models::{
    assets::{Asset, AssetData, AssetType, AssetValue}, 
    env::{Env, EnvPlayerOptions}, 
    renderer::Renderer
};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use sdl2::keyboard::Scancode;
use tch::{Tensor, Device};
use three_renderer::ThreeRenderer;
use three_rs::{core::Transformable, renderer::Event};

const LEARNING: bool = false;
const NUM_PLAYERS: usize = if LEARNING {1} else {4};
const MAP_NUM: usize = 2;
const SCREEN_W: u32 = 1000;
const SCREEN_H: u32 = 600;
const FPS: f32 = 60.0*3.0; // frames per sec
const TICKS_PER_RENDER: u32 = (1000.0 / FPS) as u32;
const DELTA_TIME: f32 = if LEARNING {1.0} else {600.0 / FPS};
const RAYS: usize = 15;
const FOV: f32 = 180.0 * (PI / 180.0);

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
    active: bool,
    last_thought: u32,
    steps_per_thought: u32,
    obs: Tensor,
}

fn main(
) -> Result<(), String> {
    let tensors_path = format!("./tensors/{}.safetensors", SEEK_TARGET_SKILL);
    let mut rng = StdRng::seed_from_u64(1234);
    let mut _pov = 0;

    let map_suffix = if LEARNING {"-training"} else {""};
    let map_path = format!("./maps/{}/map{}.glb", MAP_NUM, map_suffix);

    let (level_bytes, navmesh_bytes) = if let Ok(level_bytes) = 
        std::fs::read(&map_path) {
        if let Ok(navmesh_bytes) = 
            std::fs::read(format!("./maps/{}/navmesh{}.bin", MAP_NUM, map_suffix)) {
            (level_bytes, navmesh_bytes)
        }
        else {
            panic!("Error loading level-0's navmesh");
        }
    }
    else {
        panic!("Error loading level-0's model");
    };

    let mut renderer = ThreeRenderer::new(SCREEN_W, SCREEN_H)
        .expect("Renderer initialization failed");

    let mut game = Game::new(
        Map::new(
            &level_bytes.to_vec(), 
            &navmesh_bytes.to_vec()
        ),
        MapInfo {
            map_id: GameAssetRef::Map0 as usize,
            num_keys: 0,
        },
        GameOptions {
            fov: PlayerFOV { 
                rays: RAYS, 
                angle: FOV, 
                max_dist: 2.5,
            },
            num_players: NUM_PLAYERS,
            min_keys: 4,
            debug: true,
            training: LEARNING,
        }
    );

    let options = BaseAgentOptions { 
        dqn: DqnOptions {
            learning_rate: LEARNING_RATE,
            n_actions: game.get_num_actions(SEEK_TARGET_SKILL),
            n_obs: game.get_num_obs(SEEK_TARGET_SKILL),
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
    
    for (asset_ref, asset) in game.get_assets(
            None, 
            0, 
            &vec![None; NUM_PLAYERS]
        ) {
        let asset = if asset.typ == AssetType::Map {
            Asset::new(
                asset.typ, 
                AssetData::new(&[
                    (&"model_url", &AssetValue::Text(map_path.clone()))
                ])
            )
        }
        else {
            asset
        };

        renderer.load_asset(asset_ref, &asset)
            .expect(&format!("Asset not found: {}", asset_ref));
    }

    let mut agent = DdqnAgent::new(&options);

    if !LEARNING {
        agent.load(tensors_path.to_string())?;
    }

    let mut inst_players = vec![];
    for _ in 0..NUM_PLAYERS {
        inst_players.push(Player {
            active: true,
            last_thought: 0,
            steps_per_thought: if LEARNING { 
                1000 
            } 
            else {
                rng.gen_range(1000..3000)
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
        let mut finished = false;

        let obs = &game.reset(
            &vec![
                EnvPlayerOptions{ 
                    radius: 0.8, 
                    height: 1.8, 
                    xp: 0.0 
                }; 
                NUM_PLAYERS
            ],
            &mut renderer, 
            &mut rng
        );
        for i in 0..game.options.num_players {
            inst_players[i].last_thought = 0;
            inst_players[i].obs = Tensor::from_slice(
                &obs[i][SEEK_TARGET_SKILL]).to(Device::Cuda(0));
        }

        game.render(&mut renderer, 0.0);

        let mut players_idx = create_players_idx(&mut game, &mut rng);

        while !finished {
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

            let ticks_initial = renderer.ticks();

            let mut actions = vec![(0usize, None); game.options.num_players];

            circ_players_idx(&mut players_idx);
            
            for i in 0..game.options.num_players {
                let p = players_idx[i];
                let inst_player = &mut inst_players[p];

                actions[i].0 = p;
                
                if inst_player.active {
                    inst_player.last_thought += 1000;
                    if inst_player.last_thought >= inst_player.steps_per_thought {
                        inst_player.last_thought = 0;

                        let action_ts = agent.choose_action(&inst_player.obs, &mut rng)
                            .unwrap_or(Tensor::from(game.random_action(SEEK_TARGET_SKILL, &mut rng) as i64)
                                .to(Device::Cuda(0)));

                        let mut action = vec![0i64; 1];
                        action_ts.copy_data(&mut action, 1);
                        actions[i].1 = Some(action[0] as usize);
                    }
                }
            }

            let step = game.step(&actions, DELTA_TIME, &mut renderer, &mut rng);
            
            steps += 1;
            let timedout = steps >= MAX_STEPS;
            finished = timedout || step.terminated;

            if finished {
                results.push(if step.winner.is_some() {1} else {0});
            }

            if LEARNING && step.players.len() > 0 {
                score += step.players[0].reward;
            }

            let mut i = 0;
            for step_player in step.players {
                let inst_player = &mut inst_players[i];

                let next_obs_ts = Tensor::from_slice(
                    &step_player.obs[SEEK_TARGET_SKILL]).to(Device::Cuda(0));

                if LEARNING {
                    if let Some(action) = actions[0].1 {
                        let action_ts = Tensor::from_slice(&[action as i64])
                            .to(Device::Cuda(0));

                        agent.store_transition(
                            &inst_player.obs, 
                            &action_ts, 
                            &step_player.reward.into(), 
                            &next_obs_ts, 
                            &step.terminated.into(),
                            &timedout.into()
                        );

                        agent.learn(&mut rng);
                    }
                }

                inst_player.active = step_player.active;
                inst_player.obs = next_obs_ts;

                i += 1;
            }

            game.render(&mut renderer, if LEARNING {if steps % 12 != 0 {0.0} else {1.0}} else {1.0});

            if !LEARNING {
                let ticks_elapsed = renderer.ticks() - ticks_initial;
                if ticks_elapsed < TICKS_PER_RENDER {
                    renderer.delay(TICKS_PER_RENDER - ticks_elapsed);
                }
            }          
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

fn create_players_idx(
    game: &mut Game,
    rng: &mut StdRng
) -> Vec<usize> {
    let mut indexes = HashSet::new();

    while indexes.len() < game.options.num_players {
        indexes.insert(rng.gen_range(0..game.options.num_players));
    }

    indexes.into_iter().collect()
}

fn circ_players_idx(
    indexes: &mut Vec<usize>
) {
    let len = indexes.len();
    
    let first = indexes[0];
    for i in 0..len-1 {
        indexes[i] = indexes[i+1];
    }

    indexes[len-1] = first;
}