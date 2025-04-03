use std::time::SystemTime;

use mock_data::mock_imagenet_tensor;
use ort::{
    execution_providers::{
        CUDAExecutionProvider, /*CoreMLExecutionProvider,*/ DirectMLExecutionProvider,
        ExecutionProvider, TensorRTExecutionProvider,
    },
    session::Session,
};
//use tracing::Level;
//use tracing_subscriber::FmtSubscriber;
mod mock_data;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a subscriber with the trace level explicitly set
    /*let subscriber = FmtSubscriber::builder()
    // Set the maximum level to TRACE
    .with_max_level(Level::DEBUG)
    // You can use a more specific filter if needed
    // .with_env_filter(EnvFilter::new("your_crate=trace,other_crate=debug"))
    .finish();*/
    run_sess(CUDAExecutionProvider::default())?;
    run_sess(TensorRTExecutionProvider::default())?;
    run_sess(DirectMLExecutionProvider::default())?;
    //run_sess(CoreMLExecutionProvider::default())?;
    run_sess_cpu(2)?;
    run_sess_cpu(4)?;
    run_sess_cpu(6)?;
    run_sess_cpu(8)?;
    run_sess_cpu(10)?;
    run_sess_cpu(12)?;
    run_sess_cpu(16)?;

    println!("\nEND OF TEST! THANK YOU");
    // Set the global default subscriber
    //tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");

    Ok(())
}

pub static CNN: &[u8] = include_bytes!("efficientnet.onnx");
pub static TRANSFORMER: &[u8] = include_bytes!("vision_model_bnb4.onnx");

pub fn run_sess(ep: impl ExecutionProvider) -> Result<(), ort::Error> {
    let mut builder = Session::builder()?;

    let ep_name = ep.as_str();
    if !ep.is_available()? {
        println!("{ep_name} unavailable!");
        return Ok(());
    }
    println!("\nTesting {ep_name} -----------");
    ep.register(&mut builder)?;
    let session = builder.commit_from_memory(CNN)?;
    let input_name = "actual_input";
    let mock_data = mock_imagenet_tensor()?;
    //let output_name = "output";
    let cpu_ts = SystemTime::now();
    session.run(ort::inputs![input_name => mock_data]?)?;
    let benchmark_res = cpu_ts.elapsed().unwrap().as_millis();
    println!("\tCNN Inference: {benchmark_res}ms");

    let mut builder = Session::builder()?;
    ep.register(&mut builder)?;
    let input_name = "pixel_values";
    let mock_data = mock_imagenet_tensor()?;
    let session = builder.commit_from_memory(TRANSFORMER)?;
    //let output_name = "output";
    let cpu_ts = SystemTime::now();
    session.run(ort::inputs![input_name => mock_data]?)?;
    let benchmark_res = cpu_ts.elapsed().unwrap().as_millis();
    println!("\tTRANSFORMER Inference: {benchmark_res}ms");
    Ok(())
}

pub fn run_sess_cpu(cores: usize) -> Result<(), ort::Error> {
    let session = Session::builder()?
        .with_intra_threads(cores)?
        .commit_from_memory(CNN)?;

    println!("\nTesting CPU ({cores} threads)-----------");

    let input_name = "actual_input";
    let mock_data = mock_imagenet_tensor()?;
    //let output_name = "output";
    let cpu_ts = SystemTime::now();
    session.run(ort::inputs![input_name => mock_data]?)?;
    let benchmark_res = cpu_ts.elapsed().unwrap().as_millis();
    println!("\tCNN Inference: {benchmark_res}ms");

    let input_name = "pixel_values";
    let mock_data = mock_imagenet_tensor()?;
    let session = Session::builder()?
        .with_intra_threads(cores)?
        .commit_from_memory(TRANSFORMER)?;
    //let output_name = "output";
    let cpu_ts = SystemTime::now();
    session.run(ort::inputs![input_name => mock_data]?)?;
    let benchmark_res = cpu_ts.elapsed().unwrap().as_millis();
    println!("\tTRANSFORMER Inference: {benchmark_res}ms");
    Ok(())
}
