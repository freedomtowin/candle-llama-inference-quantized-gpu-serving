use candle_core::{Error};
use anyhow::{bail, Error as E, Result, anyhow};
use std::path::{Path, PathBuf};
use std::fs::File;
use serde_json::Value;


pub fn get(path: &Path) -> Option<PathBuf> {
    println!("Checking path: '{}'", path.display());
    if path.exists() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(json_file: &str, base_model_path: &Path) -> Result<Vec<PathBuf>> {
    let file = File::open(json_file)?;
    let json: Value = serde_json::from_reader(file)?;
    let weight_map = json.get("weight_map")
        .ok_or_else(|| anyhow!("no weight map in {}", json_file))?
        .as_object()
        .ok_or_else(|| anyhow!("weight map in {} is not a map", json_file))?;

    let safetensors_files: Vec<PathBuf> = weight_map.values()
        .filter_map(|v| v.as_str())
        .map(|file| {
            let mut path = base_model_path.to_path_buf();
            path.push(file);
            get(&path).ok_or_else(|| anyhow!("File not found: {}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(safetensors_files)
}