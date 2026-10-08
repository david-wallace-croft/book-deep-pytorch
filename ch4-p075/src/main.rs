use ::candle_core::{DType, Device, Error, Result, Tensor};
use ::image::buffer::Pixels;
use ::image::{DynamicImage, ImageBuffer, ImageError, ImageReader, Rgb};
use ::std::fs::File;
use ::std::io::BufReader;
use ::std::path::{Path, PathBuf};

const IMAGE_CHANNELS: usize = 3;
const IMAGE_FILENAME: &str = "candle-256x256.jpg";
const IMAGE_HEIGHT: usize = 256;
const IMAGE_WIDTH: usize = 256;
const SCALING_FACTOR: f32 = 255.;
const SHAPE: (usize, usize, usize) =
  (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let image_path_buf: PathBuf = get_file_path(IMAGE_FILENAME);

  let image_tensor: Tensor = load_image_to_tensor_0(&image_path_buf, &device)?;

  println!("Image tensor: {image_tensor}\n");

  let image_tensor: Tensor = load_image_to_tensor_1(&image_path_buf, &device)?;

  println!("Image tensor: {image_tensor}\n");

  Ok(())
}

fn get_file_path(filename: &str) -> PathBuf {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let cargo_manifest_dir_path: &Path = Path::new(cargo_manifest_dir);

  cargo_manifest_dir_path.join(filename)
}

fn load_image_to_tensor_0(
  path: &PathBuf,
  device: &Device,
) -> Result<Tensor> {
  let dynamic_image: DynamicImage =
    ::image::open(path).map_err(|e: ImageError| Error::Msg(e.to_string()))?;

  let rgb_image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = dynamic_image.to_rgb8();

  let pixels: Pixels<'_, Rgb<u8>> = rgb_image_buffer.pixels();

  let normalized_pixels_vec: Vec<f32> = pixels
    .flat_map(|rgb: &Rgb<u8>| {
      let [
        r,
        g,
        b,
      ] = rgb.0;

      [
        r as f32 / SCALING_FACTOR,
        g as f32 / SCALING_FACTOR,
        b as f32 / SCALING_FACTOR,
      ]
    })
    .collect();

  let hwc_tensor: Tensor =
    Tensor::from_vec(normalized_pixels_vec, SHAPE, device)?;

  let chw_tensor: Tensor = hwc_tensor.permute((2, 0, 1))?;

  Ok(chw_tensor)
}

// An alternative implementation based on
// https://github.com/huggingface/candle/blob/main/candle-examples/src/
//   imagenet.rs
fn load_image_to_tensor_1(
  path: &PathBuf,
  device: &Device,
) -> Result<Tensor> {
  let image_reader: ImageReader<BufReader<File>> = ImageReader::open(path)?;

  let dynamic_image: DynamicImage =
    image_reader.decode().map_err(Error::wrap)?;

  let rgb_image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = dynamic_image.to_rgb8();

  let u8_vec: Vec<u8> = rgb_image_buffer.into_raw();

  let u8_tensor: Tensor = Tensor::from_vec(u8_vec, SHAPE, device)?;

  let hwc_tensor: Tensor = u8_tensor.to_dtype(DType::F32)?;

  let chw_tensor: Tensor = hwc_tensor.permute((2, 0, 1))?;

  let normalized_tensor: Tensor = (chw_tensor / (SCALING_FACTOR as f64))?;

  Ok(normalized_tensor)
}
