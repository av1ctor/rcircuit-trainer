use rand::rngs::StdRng;
use tch::{Tensor, Scalar};
use crate::{base_agent::{BaseAgent, BaseAgentOptions}, agent::Agent};

pub struct DqnAgent {
    pub base: BaseAgent,
}

impl Agent for DqnAgent {
    fn new(
        options: &BaseAgentOptions
    ) -> Self {
        Self {
            base: BaseAgent::new(options),
        }
    }

    fn choose_action(
        &mut self, 
        obs: &Tensor,
        rng: &mut StdRng
    ) -> Option<Tensor> {
        self.base.choose_action(obs, rng)
    }

    fn store_transition(
        &mut self, 
        state: &Tensor, 
        action: &Tensor, 
        reward: &Tensor, 
        next_state: &Tensor,
        terminated: &Tensor,
        timedout: &Tensor
    ) {
        self.base.store_transition(
            state, action, reward, next_state, terminated, timedout
        )
    }

    fn learn(
        &mut self,
        rng: &mut StdRng
    ) {
        let (states, actions, rewards, next_states, terminated, _) =
            match self.base.sample_memory(rng) {
                Some(v) => v,
                _ => return
            };

        let q_pred = self.base.q_eval.forward(&states).index_select(1, &actions);
        
        let loss = tch::no_grad(|| {
            let q_next = self.base.q_next.forward(&next_states);
            let q_next = q_next.masked_fill(&terminated.eq(1i64), Scalar::from(0.0f64));

            let max_actions = q_next.argmax(1, false);
            let q_target = rewards + self.base.options.gamma * q_next.index_select(1, &max_actions);

            q_target
        }).mse_loss(&q_pred, tch::Reduction::Mean);

        self.base.q_eval.opt.zero_grad();
        loss.backward();
        self.base.q_eval.opt.step();
        self.base.learn_step_counter += 1;

        self.base.decrement_epsilon();
    }

    fn save(
        &self,
        path: String
    ) -> Result<(), String> {
        self.base.save(path)
    }

    fn load(
        &mut self,
        path: String
    ) -> Result<(), String> {
        self.base.load(path)
    }

    fn get_epsilon(
        &self
    ) -> f64 {
        self.base.options.epsilon
    }
}
