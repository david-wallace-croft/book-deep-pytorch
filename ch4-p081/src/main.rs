use ::anyhow::Result;
use ::candle_core::{Device, Tensor};
use ::csv::{Reader, ReaderBuilder, StringRecord};
use ::std::fs::File;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let csv_file: PathBuf = Path::new(cargo_manifest_dir)
    .join("tabular-wine")
    .join("winequality-white.csv");

  let mut file_reader: Reader<File> = ReaderBuilder::new()
    .has_headers(true)
    .delimiter(b';')
    .from_path(csv_file)?;

  let headers: &StringRecord = file_reader.headers()?;

  let columns: usize = headers.len();

  println!("\n{headers:?}");

  let mut data: Vec<f32> = Vec::new();

  let mut rows: usize = 0;

  for result in file_reader.records() {
    let record: StringRecord = result?;

    rows += 1;

    for field in record.iter() {
      let value: f32 = field.parse()?;

      data.push(value);
    }
  }

  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let tensor = Tensor::from_vec(data, (rows, columns), &device)?;

  println!("{tensor:?}");

  Ok(())
}
