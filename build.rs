use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("upgraded_model.onnx")
        .out_dir("model/")
        .run_from_script();
}