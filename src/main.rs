
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod hf_helper;

use std::sync::Arc;
use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{Device, DType, Tensor, Error};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::{apply_repeat_penalty};
// use candle_transformers::models::llama as model;
// use model::{Llama, LlamaConfig};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::collections::HashSet;

use tide::prelude::*;
use tide::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Input {
    data: String,
}

#[derive(Debug, Serialize)]
struct Output {
    result: String,
}



pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}


// Hardcoded values
const EOS_TOKEN: [&str; 7] = ["", "</s>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<eos_token>", "<|end_of_text|>"];
const DEFAULT_PROMPT: &str = "My favorite theorem is ";
const CPU: bool = false;
const TEMPERATURE: f64 = 0.0  ;
const TOP_P: Option<f64> = None;
const TOP_K: Option<usize> = None;
const SEED: u64 = 299792458;
const SAMPLE_LEN: usize = 10;
const DTYPE: Option<&str> = None;
const REPEAT_PENALTY: f32 = 1.5;
const REPEAT_LAST_N: usize = 256;

// fn main() -> Result<()> {
//     let output = execute_model("Say Hello");
//     Ok(())
// }
#[derive(Debug, Clone)] // Add Clone trait to FileOutputs
struct ModelData {
    tokenizer: Tokenizer,
    llama: ModelWeights,
    // Add more fields as needed
}


#[async_std::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_outputs = load_files().await?;
    let shared_state = Arc::new(file_outputs);

    let mut app = tide::with_state(shared_state);

    // let mut app = tide::new();
    app.at("/process").post(handle_post);
    app.listen("127.0.0.1:8080").await?;
    Ok(())

    
}

async fn handle_post(mut req: Request<Arc<ModelData>>) -> tide::Result {
    let input: Input = req.body_json().await?;
    let model_data = req.state().clone();
    let result = execute_model(&input.data, &model_data)?;
    let output = Output { result };
    
    let response = Response::builder(StatusCode::Ok)
        .body(json!(output))
        .content_type(tide::http::mime::JSON)
        .build();

    Ok(response)
}


async fn load_files() -> Result<ModelData, Box<dyn std::error::Error>> {
    let device = device(CPU)?;

    let tokenizer_filename = "/root/llm-models/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mut llama = {
        let model_path = "/root/llm-models/hub/models--QuantFactory--Meta-Llama-3-8B-GGUF/snapshots/1ca85c857dce892b673b988ad0aa83f2cb1bbd19/Meta-Llama-3-8B-Instruct.Q8_0.gguf";

        let mut file = std::fs::File::open(&model_path)?;

        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({})",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
        );
        ModelWeights::from_gguf(model, &mut file, &device)?
    };
    // Load more files as needed

    Ok(ModelData {
        tokenizer,
        llama,
        // Initialize more fields as needed
    })
}

fn execute_model(prompt: &str, model_data: &ModelData) -> Result<String> {
    
    let tokenizer = &model_data.tokenizer.clone(); 
    let llama = &mut model_data.llama.clone(); 

    let device = device(CPU)?;


    let eos_tokens: HashSet<&str> = EOS_TOKEN.iter().cloned().collect();

    // Convert EOS tokens to their respective IDs in the tokenizer
    let eos_token_ids: HashSet<u32> = eos_tokens.iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect();

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = TokenOutputStream::new(tokenizer.clone());

    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = {
        let temperature = TEMPERATURE;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (TOP_K, TOP_P) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(SEED, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    let mut decoded_tokens = Vec::new(); // Initialize a vector to store decoded tokens

    for index in 0..SAMPLE_LEN {
        let (context_size, context_index) = {(tokens.len(), 0)};
   
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        // let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = llama.forward(&input, context_index)?;
        let logits = logits.squeeze(0)?;
        let logits = if REPEAT_PENALTY == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(REPEAT_LAST_N);
            apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // if Some(next_token) == eos_token_id {
        //     break;
        // }

        if eos_token_ids.contains(&next_token) {
            println!("\nEncountered an EOS token, breaking out of the loop.");
            break;
        }

        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
            decoded_tokens.push(t.to_string()); // Collect the decoded token
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
        decoded_tokens.push(rest.to_string()); // Collect any remaining decoded tokens
    }

    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(decoded_tokens.join(""))
}