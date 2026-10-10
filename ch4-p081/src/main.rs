use ::anyhow::Result;
use ::candle_core::{DType, Device, IndexOp, Tensor};
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

  let mut f32_vec: Vec<f32> = Vec::new();

  let mut rows: usize = 0;

  for result in file_reader.records() {
    let record: StringRecord = result?;

    rows += 1;

    for field in record.iter() {
      let value: f32 = field.parse()?;

      f32_vec.push(value);
    }
  }

  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let data_tensor: Tensor =
    Tensor::from_vec(f32_vec, (rows, columns), &device)?;

  println!("Data:\n{data_tensor:?}\n");

  let rank: usize = data_tensor.rank();

  let column_dimension_index: usize = rank - 1;

  let column_dimension_size = data_tensor.dim(column_dimension_index)?;

  let score_column_index = column_dimension_size - 1;

  let scores_f32: Tensor = data_tensor.i((.., score_column_index))?;

  let scores: Tensor = scores_f32.to_dtype(DType::U8)?;

  println!("Scores:\n{scores}\n");

  Ok(())
}
