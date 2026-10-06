use ::candle_core::{Device, Error, Result, Tensor};
use ::image::buffer::Pixels;
use ::image::{DynamicImage, ImageBuffer, ImageError, Rgb};
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

  let image_tensor: Tensor = load_image_to_tensor(&image_path_buf, &device)?;

  println!("Image tensor shape: {:?}", image_tensor.shape());

  Ok(())
}

fn get_file_path(filename: &str) -> PathBuf {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let cargo_manifest_dir_path: &Path = Path::new(cargo_manifest_dir);

  cargo_manifest_dir_path.join(filename)
}

fn load_image_to_tensor(
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
