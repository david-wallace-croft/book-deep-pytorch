use ::std::path::PathBuf;
use ::tch::nn::{ModuleT, Path, VarStore};
use ::tch::vision::{imagenet, resnet};
use ::tch::{Device, Kind, TchError, Tensor};

const TRAIN: bool = false;

fn main() -> Result<(), TchError> {
  let (model, mut var_store): (Box<dyn ModuleT>, VarStore) = load_model()?;

  load_weights(&mut var_store)?;

  let image: Tensor = load_image()?;

  let unsqueezed: Tensor = image.unsqueeze(0);

  let output: Tensor = model.forward_t(&unsqueezed, TRAIN);

  let probabilities: Tensor = output.softmax(-1, Kind::Float);

  for (probability, class) in imagenet::top(&probabilities, 5).iter() {
    let percentage = probability * 100.0;

    println!("{class:50} {percentage:5.2}%");
  }

  Ok(())
}

fn get_file_path(filename: &str) -> PathBuf {
  let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");

  let cargo_manifest_dir_path = ::std::path::Path::new(cargo_manifest_dir);

  cargo_manifest_dir_path.join(filename)
}

fn load_image() -> Result<Tensor, TchError> {
  let file_path: PathBuf = get_file_path("bobby.jpg");

  let image_tensor: Tensor = imagenet::load_image_and_resize224(file_path)?;

  Ok(image_tensor)
}

fn load_model() -> Result<(Box<dyn ModuleT>, VarStore), TchError> {
  // let device: Device = Device::cuda_if_available();
  let device: Device = Device::Cpu;

  let var_store: VarStore = VarStore::new(device);

  let path: Path = var_store.root();

  let resnet = Box::new(resnet::resnet101(&path, imagenet::CLASS_COUNT));

  Ok((resnet, var_store))
}

fn load_weights(var_store: &mut VarStore) -> Result<(), TchError> {
  let file_path: PathBuf = get_file_path("resnet18.ot");

  var_store.load(file_path)?;

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
