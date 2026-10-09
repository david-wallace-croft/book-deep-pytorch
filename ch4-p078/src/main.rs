use ::candle_core::{Device, Error, Result, Tensor};
use ::dicom::object::{self, FileDicomObject, InMemDicomObject, ReadError};
use ::dicom::pixeldata::{self, DecodedPixelData, PixelDecoder};
use ::std::fs::{self, DirEntry};
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let scan_dir: PathBuf = Path::new(cargo_manifest_dir)
    .join("volumetric-dicom")
    .join("2-LUNG 3.0  B70f-04083");

  let dicom_files: Vec<PathBuf> = get_dicom_files(scan_dir)?;

  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let volume_tensor: Tensor = load_dicom_volume(&dicom_files, &device)?;

  println!("volume_tensor: {:?}", volume_tensor);

  Ok(())
}

fn get_dicom_files<P: AsRef<Path>>(dir_path: P) -> Result<Vec<PathBuf>> {
  let mut dcm_files: Vec<PathBuf> = Vec::new();

  for entry_result in fs::read_dir(dir_path)? {
    let entry: DirEntry = entry_result?;

    let path: PathBuf = entry.path();

    if path.is_file()
      && let Some(extension) = path.extension()
      && extension.to_string_lossy().eq_ignore_ascii_case("dcm")
    {
      dcm_files.push(path);
    }
  }

  dcm_files.sort();

  Ok(dcm_files)
}

fn load_dicom_volume<P: AsRef<Path>>(
  slice_paths: &[P],
  device: &Device,
) -> Result<Tensor> {
  let mut slices: Vec<Tensor> = Vec::new();

  if slice_paths.is_empty() {
    return Err(Error::Msg("No DICOM paths provided".to_string()));
  }

  for path in slice_paths {
    let obj: FileDicomObject<InMemDicomObject> = object::open_file(path)
      .map_err(|e: ReadError| {
        Error::Msg(format!("Failed to open DICOM: {e}"))
      })?;

    let decoded: DecodedPixelData<'_> =
      obj.decode_pixel_data().map_err(|e: pixeldata::Error| {
        Error::Msg(format!("Pixel decoding failed: {e}"))
      })?;

    let width: usize = decoded.columns() as usize;

    let height: usize = decoded.rows() as usize;

    // Channels is usually one for grayscale

    let channels: usize = decoded.samples_per_pixel() as usize;

    let shape: (usize, usize, usize) = (channels, height, width);

    let raw_pixels: Vec<f32> =
      decoded
        .to_vec::<f32>()
        .map_err(|e: dicom::pixeldata::Error| {
          Error::Msg(format!("Pixel conversion failed: {e}"))
        })?;

    let slice_tensor: Tensor = Tensor::from_vec(raw_pixels, shape, device)?;

    slices.push(slice_tensor);
  }

  // Channels, Depth, Channels, Height, Width

  let volume_tensor: Tensor = Tensor::stack(&slices, 1)?;

  Ok(volume_tensor)
}
