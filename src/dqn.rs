use tch::{Tensor, Device, nn::{self, OptimizerConfig, Optimizer}};

#[derive(Clone)]
pub struct DqnOptions {
    pub learning_rate: f64,
    pub n_obs: usize,
    pub n_hidden: usize,
    pub n_actions: usize,
}

pub struct DeepQNetwork {
    pub var_store: nn::VarStore,
    pub options: DqnOptions,
    pub network: nn::Sequential,
    pub opt: Optimizer,
    pub device: Device,
}

impl Clone for DeepQNetwork {
    fn clone(
        &self
    ) -> Self {
        let mut new = Self::new(&self.options);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl DeepQNetwork {
    pub fn new(
        options: &DqnOptions
    ) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cuda(0));
        let opt = nn::AdamW::default().build(&var_store, options.learning_rate).unwrap();
        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(
                    p / "in-hid1", 
                    options.n_obs as _, 
                    options.n_hidden as _, 
                    Default::default()
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "hid1-out", 
                    options.n_hidden as _, 
                    options.n_actions as _, 
                    Default::default()
                )),
            device: p.device(),
            var_store,
            opt,
            options: options.clone(),
        }
    }

    pub fn forward(
        &self, 
        obs: &Tensor
    ) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }

    pub fn save(
        &self,
        path: String
    ) -> Result<(), tch::TchError> {
        self.var_store.save(path)
    }

    pub fn load(
        &mut self,
        path: String
    ) -> Result<(), tch::TchError> {
        self.var_store.load(path)
    }
}