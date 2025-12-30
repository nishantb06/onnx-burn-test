# ONNX to Burn: all-MiniLM-L6-v2 Inference in Rust

This project demonstrates how to import the **all-MiniLM-L6-v2** sentence transformer model from ONNX format into Rust using the [Burn](https://burn.dev/) deep learning framework, and perform inference.

## Overview

- **Model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - A sentence transformer that maps sentences to 384-dimensional dense vector embeddings
- **Framework**: [Burn](https://github.com/tracel-ai/burn) - A deep learning framework written in Rust
- **Conversion**: Uses `burn-import` to convert ONNX model to native Rust code at build time

## Project Structure

```
onnx-burn-test/
├── build.rs                    # Generates Rust code from ONNX at build time
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Library exposing the generated model
│   ├── main.rs
│   ├── bin/
│   │   └── allMiniLML6.rs      # Inference script
│   └── model/
│       ├── mod.rs              # Module including generated code
│       └── allMiniLML6.onnx    # The ONNX model file
└── README.md
```

## How It Works

### 1. Build-time ONNX Conversion

The `build.rs` script uses `burn-import` to convert the ONNX model into native Rust code during compilation:

```rust
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/allMiniLML6.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

### 2. Model Interface

The generated model has the following interface:

```rust
pub fn forward(
    &self,
    input_ids: Tensor<B, 2, Int>,      // [batch_size, seq_len]
    attention_mask: Tensor<B, 2, Int>, // [batch_size, seq_len]
    token_type_ids: Tensor<B, 2, Int>, // [batch_size, seq_len]
) -> (Tensor<B, 3>, Tensor<B, 2>)      // (last_hidden_state, pooler_output)
```

- **Inputs**: Tokenized text (token IDs, attention mask, token type IDs)
- **Outputs**: 
  - `last_hidden_state`: Hidden states for all tokens `[batch_size, seq_len, 384]`
  - `pooler_output`: Sentence embedding `[batch_size, 384]`

### 3. Running Inference

The inference script performs a forward pass with sample tokenized input:

```rust
// Example: "Hello world" tokenized with BERT tokenizer
// [CLS] = 101, Hello = 7592, world = 2088, [SEP] = 102
let input_ids: Vec<i64> = vec![101, 7592, 2088, 102];

let (last_hidden_state, pooler_output) = model.forward(
    input_ids_tensor,
    attention_mask_tensor,
    token_type_ids_tensor,
);
```

## Usage

### Prerequisites

- Rust (1.70+)
- Cargo

### Build

```bash
cargo build
```

### Run Inference

```bash
cargo run --bin allMiniLML6
```

### Expected Output

```
Loading model...
Model loaded successfully!
Input shape: [1, 4]
Running inference...
Inference complete!

Last hidden state shape: [1, 4, 384]
Pooler output (sentence embedding) shape: [1, 384]

Sentence embedding (first 10 values):
  [  0]: -0.039659
  [  1]: 0.022167
  [  2]: -0.011310
  [  3]: 0.027763
  [  4]: -0.024910
  [  5]: -0.006622
  [  6]: 0.019669
  [  7]: 0.053398
  [  8]: -0.117513
  [  9]: 0.007952
  ... (384 total dimensions)

Embedding L2 norm: 1.183152

Success! The model is working correctly.
```

## Model Details

| Property | Value |
|----------|-------|
| Hidden Size | 384 |
| Vocabulary Size | 30,522 |
| Max Sequence Length | 512 |
| Attention Heads | 12 |
| Layers | 6 |

## Dependencies

```toml
[dependencies]
burn = { version = "0.19.1", features = ["ndarray"] }

[build-dependencies]
burn-import = "0.19.1"
```

## Next Steps

For production use, you would want to:

1. **Add a tokenizer** - Use the `tokenizers` crate to properly tokenize input text
2. **Batch processing** - Process multiple sentences at once
3. **GPU acceleration** - Use `burn`'s WGPU or CUDA backends instead of ndarray

## References

- [Burn Documentation](https://burn.dev/docs)
- [Burn ONNX Import Guide](https://burn.dev/book/import/onnx-model.html)
- [all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

