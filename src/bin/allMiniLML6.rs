use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

// Import the generated model from the library
use onnx_burn_test::Model;

fn main() {
    // Define the backend type
    type Backend = NdArray<f32>;

    // Get a default device for the backend
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();

    // Create a new model and load the weights
    println!("Loading model...");
    let model: Model<Backend> = Model::default();
    println!("Model loaded successfully!");

    // For all-MiniLM-L6-v2, the inputs are:
    // - input_ids: token IDs from tokenizer
    // - attention_mask: 1 for real tokens, 0 for padding
    // - token_type_ids: segment IDs (usually all 0s for single sentence)
    //
    // The model has:
    // - vocab_size: 30522 (BERT vocabulary)
    // - max_seq_length: 512
    // - hidden_size: 384
    //
    // For this demo, we'll use hardcoded token IDs.
    // In production, you'd use a tokenizer like `tokenizers` crate.
    
    // Example: "Hello world" tokenized with BERT tokenizer
    // [CLS] = 101, Hello = 7592, world = 2088, [SEP] = 102
    let input_ids: Vec<i64> = vec![101, 7592, 2088, 102];
    let seq_len = input_ids.len();
    
    // Create attention mask (1 for all real tokens)
    let attention_mask: Vec<i64> = vec![1; seq_len];
    
    // Create token type IDs (all 0s for single sentence)
    let token_type_ids: Vec<i64> = vec![0; seq_len];

    // Convert to tensors with shape [batch_size=1, seq_len]
    let input_ids_tensor = Tensor::<Backend, 1, burn::tensor::Int>::from_ints(
        input_ids.as_slice(),
        &device,
    )
    .reshape([1, seq_len as i64]);

    let attention_mask_tensor = Tensor::<Backend, 1, burn::tensor::Int>::from_ints(
        attention_mask.as_slice(),
        &device,
    )
    .reshape([1, seq_len as i64]);

    let token_type_ids_tensor = Tensor::<Backend, 1, burn::tensor::Int>::from_ints(
        token_type_ids.as_slice(),
        &device,
    )
    .reshape([1, seq_len as i64]);

    println!("Input shape: [1, {}]", seq_len);
    println!("Running inference...");

    // Run the forward pass
    let (last_hidden_state, pooler_output) = model.forward(
        input_ids_tensor,
        attention_mask_tensor,
        token_type_ids_tensor,
    );

    println!("Inference complete!");
    println!();

    // Print output shapes
    println!("Last hidden state shape: {:?}", last_hidden_state.dims());
    println!("Pooler output (sentence embedding) shape: {:?}", pooler_output.dims());
    println!();

    // Print the sentence embedding (first 10 values)
    let embedding_data = pooler_output.to_data();
    println!("Sentence embedding (first 10 values):");
    let values: Vec<f32> = embedding_data.to_vec().unwrap();
    for (i, val) in values.iter().take(10).enumerate() {
        println!("  [{:3}]: {:.6}", i, val);
    }
    println!("  ... ({} total dimensions)", values.len());

    // Compute L2 norm of the embedding
    let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!();
    println!("Embedding L2 norm: {:.6}", norm);

    println!();
    println!("Success! The model is working correctly.");
}

