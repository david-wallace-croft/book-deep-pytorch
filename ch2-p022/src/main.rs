#![expect(unused_imports)]
#![expect(unused_mut)]
#![expect(unused_variables)]

use ::std::error::Error;
use ::std::path::{Path, PathBuf};
use ::tch::nn::VarStore;
use ::tch::vision::{alexnet, imagenet, resnet};
use ::tch::{Device, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
  let device: Device = Device::cuda_if_available();

  // Create a path variable for the neural network weights
  let mut var_store: VarStore = VarStore::new(device);

  // Standard ImageNet classes for AlexNet
  let nclasses: i64 = 1_000;

  // Pass the root path from the VarStore and the number of classes
  // let _model = alexnet::alexnet(&var_store.root(), nclasses);

  let resnet = resnet::resnet101(&var_store.root(), nclasses);

  let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");

  let project_root_path = Path::new(cargo_manifest_dir);

  // 3. Load the pretrained weights file
  // (Note: You must download the PyTorch-compatible resnet101 weights file
  // e.g., 'resnet101-5d3b4d8f.pth' from the official PyTorch repo and save it locally)
  let weights_path: PathBuf = project_root_path.join("resnet101-63fe2227.pth");

  // var_store.load(weights_path)?;

  // Append a file or subpath to the root path
  let file_path: PathBuf = project_root_path.join("bobby.jpg");

  let image_tensor: tch::Tensor =
    imagenet::load_image_and_resize224(file_path)?;

  let batch_tensor: tch::Tensor = image_tensor.unsqueeze(0);

  let output = batch_tensor.apply_t(&resnet, false);

  println!("Output shape: {:?}", output.size());

  println!("{output:?}");

  // Get the top 3 values and indices along the last dimension (-1)
  let (top_values, top_indices) = output.topk(3, -1, true, true);

  // Print the results
  println!("Top 3 values: {:?}", top_values);
  println!("Top 3 class indices: {:?}", top_indices);

  // Assuming top_values is your tensor from topk()
  // Convert the tensor to a 1D Rust vector of f64 (or f32)
  // let values: Vec<f64> = top_values.iter::<f64>().unwrap().collect();

  // println!("{values:?}");

  // // Alternatively, you can access them by index directly:
  // for i in 0..3 {
  //   let val = top_values.double_value(&[
  //     0, i,
  //   ]); // Assuming shape is [1, 3]
  //   println!("Top value {}: {}", i + 1, val);
  // }

  for i in 0..3 {
    let val = top_indices.double_value(&[
      0, i,
    ]);
    println!("Element {}: {}", i, val);
  }

  Ok(())
}

#[cfg(test)]
mod test {
  // use super::*;

  #[test]
  fn test() {
    // TODO
  }
}
