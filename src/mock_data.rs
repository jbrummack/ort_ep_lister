use ort::value::{TensorValueType, Value};
use rand::Rng;

fn create_random_data() -> Vec<f32> {
    // Same dimensions as the original function
    let width = 224;
    let height = 224;
    let channels = 3;

    // Create a vector with the right size
    let mut normalized_data = vec![0.0; width * height * channels];

    // Use a random number generator
    let mut rng = rand::rng();

    // Fill with random values in [0.0, 1.0] range
    for c in 0..channels {
        for i in 0..(width * height) as usize {
            // Generate random value between 0.0 and 1.0
            let random_value = rng.random_range(0.0..1.0);

            // Apply the same normalization as in the original function
            normalized_data[c * (width * height) as usize + i] = (random_value) / 1.0;
        }
    }

    normalized_data
}

pub fn mock_imagenet_tensor() -> Result<Value<TensorValueType<f32>>, ort::Error> {
    let normalized_data = create_random_data();
    let tensor_shape = vec![1, 3, 224, 224];
    let tensor_args = (tensor_shape, normalized_data);
    let input_array = ort::value::Tensor::from_array(tensor_args)?;
    Ok(input_array)
}
