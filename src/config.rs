use serde::Deserialize;
use std::error::Error;
use std::fs;

#[derive(Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub architecture: String,
    pub prefer_mmap: bool,
    pub context_token_length: usize,
    pub use_gpu: bool,
    pub gpu_layers: usize,
}

#[derive(Deserialize)]
pub struct Configuration {
    pub model: ModelConfig,
}

impl Configuration {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let config_str = fs::read_to_string("config.toml")?;
        let config: Configuration = toml::from_str(&config_str)?;
        Ok(config)
    }
}
