use rand::{rngs::StdRng, Rng};
use tch::Tensor;

use crate::{replay_buffer::ReplayBuffer, dqn::{DqnOptions, DeepQNetwork}};

#[derive(Clone)]
pub struct BaseAgentOptions {
    pub dqn: DqnOptions,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_dec: f64,
    pub epsilon_min: f64,
    pub batch_size: usize,
    pub replay_buffer_capacity: usize,
    pub replace_target_cnt: usize,
}

pub struct BaseAgent {
    pub options: BaseAgentOptions,
    pub memory: ReplayBuffer,
    pub learn_step_counter: usize,
    pub q_eval: DeepQNetwork,
    pub q_next: DeepQNetwork,
}

impl BaseAgent {
    pub fn new(
        options: &BaseAgentOptions
    ) -> Self {
        let memory =
            ReplayBuffer::new(
                options.replay_buffer_capacity, 
                options.dqn.n_obs, 
                options.dqn.n_actions
            );

        let q_eval = DeepQNetwork::new(&options.dqn);
        let q_next = DeepQNetwork::new(&options.dqn);
        
        Self {
            options: options.clone(),
            memory,
            learn_step_counter: 0,
            q_eval,
            q_next,
        }
    }

    pub fn choose_action(
        &mut self, 
        obs: &Tensor,
        rng: &mut StdRng
    ) -> Option<Tensor> {
        if rng.gen::<f64>() > self.options.epsilon {
            let actions = tch::no_grad(|| self.q_eval.forward(obs));
            Some(actions.argmax(0, false))
        }
        else {
            None
        }
    }
    
    pub fn store_transition(
        &mut self, 
        state: &Tensor, 
        action: &Tensor, 
        reward: &Tensor, 
        next_state: &Tensor,
        terminated: &Tensor,
        timedout: &Tensor
    ) {
        self.memory.store(state, action, reward, next_state, terminated, timedout);
    }

    pub fn sample_memory(
        &mut self,
        rng: &mut StdRng
    ) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.memory.sample(self.options.batch_size, rng)
    }

    pub fn decrement_epsilon(
        &mut self
    ) {
        if self.options.epsilon > self.options.epsilon_min {
            self.options.epsilon -= self.options.epsilon_dec;
        } 
        else {
            self.options.epsilon = self.options.epsilon_min;
        }
    }

    pub fn replace_target_network(
        &mut self
    ) {
        if self.learn_step_counter % self.options.replace_target_cnt == 0 {
            self.q_next.var_store.copy(&self.q_eval.var_store).unwrap();
        }
    }

    pub fn save(
        &self,
        path: String
    ) -> Result<(), String> {
        if let Err(err) = self.q_eval.save(path) {
            Err(err.to_string())
        }
        else {
            Ok(())
        }
    }

    pub fn load(
        &mut self,
        path: String
    ) -> Result<(), String> {
        if let Err(err) = self.q_eval.load(path) {
            Err(err.to_string())
        }
        else {
            self.options.epsilon = self.options.epsilon_min;
            Ok(())
        }
    }
}
