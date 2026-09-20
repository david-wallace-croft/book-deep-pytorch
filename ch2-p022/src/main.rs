use ::std::path::PathBuf;
use ::tch::nn::{ModuleT, Path, VarStore};
use ::tch::vision::{alexnet, imagenet, resnet};
use ::tch::{Device, Kind, TchError, Tensor};

// This code is based on a tch example:
//   https://github.com/LaurentMazare/tch-rs/blob/main/examples/
//     pretrained-models/main.rs
// Download bobby.jpg from
//   https://github.com/deep-learning-with-pytorch/dlwpt-code-2e/blob/main/data/
//     p1ch2/bobby.jpg
// Download alexnet18.ot from
//   https://github.com/LaurentMazare/tch-rs/releases/download/mw/alexnet.ot
// Download resnet18.ot from
//   https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot

const IMAGE_FILENAME: &str = "bobby.jpg";
const TOP_COUNT: i64 = 5;
const TRAIN: bool = false;
const WEIGHTS_ALEXNET: &str = "alexnet.ot";
const WEIGHTS_RESNET: &str = "resnet18.ot";

fn main() -> Result<(), TchError> {
  println!("=== AlexNet");

  let model: Box<dyn ModuleT> = load_alexnet()?;

  label_image(model)?;

  println!("=== ResNet");

  let model: Box<dyn ModuleT> = load_resnet()?;

  label_image(model)
}

fn get_file_path(filename: &str) -> PathBuf {
  let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");

  let cargo_manifest_dir_path = ::std::path::Path::new(cargo_manifest_dir);

  cargo_manifest_dir_path.join(filename)
}

fn infer(
  image: Tensor,
  model: Box<dyn ModuleT>,
) -> Tensor {
  // println!("image: {image:?}");

  // "Returns a new tensor with a dimension of size one inserted at the
  // specified position"
  // https://docs.pytorch.org/docs/main/generated/torch.unsqueeze.html
  let unsqueezed: Tensor = image.unsqueeze(0);

  // println!("unsqueezed: {unsqueezed:?}");

  let output: Tensor = model.forward_t(&unsqueezed, TRAIN);

  // https://docs.pytorch.org/docs/main/generated/
  //   torch.nn.functional.softmax.html
  output.softmax(-1, Kind::Float)
}

fn label_image(model: Box<dyn ModuleT>) -> Result<(), TchError> {
  let image: Tensor = load_image()?;

  let probabilities: Tensor = infer(image, model);

  print_top(&probabilities);

  Ok(())
}

fn load_image() -> Result<Tensor, TchError> {
  let file_path: PathBuf = get_file_path(IMAGE_FILENAME);

  let image_tensor: Tensor = imagenet::load_image_and_resize224(file_path)?;

  Ok(image_tensor)
}

fn load_alexnet() -> Result<Box<dyn ModuleT>, TchError> {
  let device: Device = Device::cuda_if_available();

  let mut var_store: VarStore = VarStore::new(device);

  let path: Path = var_store.root();

  let alexnet = Box::new(alexnet::alexnet(&path, imagenet::CLASS_COUNT));

  let file_path: PathBuf = get_file_path(WEIGHTS_ALEXNET);

  var_store.load(file_path)?;

  Ok(alexnet)
}

fn load_resnet() -> Result<Box<dyn ModuleT>, TchError> {
  let device: Device = Device::cuda_if_available();

  let mut var_store: VarStore = VarStore::new(device);

  let path: Path = var_store.root();

  let resnet = Box::new(resnet::resnet18(&path, imagenet::CLASS_COUNT));

  let file_path: PathBuf = get_file_path(WEIGHTS_RESNET);

  // Weights must be loaded after VarStore is associated with the model

  var_store.load(file_path)?;

  Ok(resnet)
}

fn print_top(probabilities: &Tensor) {
  let top: Vec<(f64, String)> = imagenet::top(probabilities, TOP_COUNT);

  for (probability, class) in top.iter() {
    let percentage = probability * 100.0;

    println!("{class:50} {percentage:5.2}%");
  }
}
