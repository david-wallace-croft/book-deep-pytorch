use ::anyhow::Result;
use ::candle_core::{DType, Device, IndexOp, Tensor};
use ::csv::{Reader, ReaderBuilder, StringRecord};
use ::std::fs::File;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  // page 81

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

  // page 82

  let data_tensor: Tensor =
    Tensor::from_vec(f32_vec, (rows, columns), &device)?;

  // page 83

  println!("Data:\n{data_tensor:?}\n");

  let rank: usize = data_tensor.rank();

  let column_dimension_index: usize = rank - 1;

  let column_dimension_size = data_tensor.dim(column_dimension_index)?;

  let score_column_index = column_dimension_size - 1;

  let scores_f32: Tensor = data_tensor.i((.., score_column_index))?;

  let scores: Tensor = scores_f32.to_dtype(DType::U8)?;

  println!("Scores:\n{scores}\n");

  // page 84

  let target_onehot = Tensor::zeros((rows, 10), DType::F32, &device)?;

  println!("\ntarget_onehot:\n{target_onehot}\n");

  let indices = scores.unsqueeze(1)?;

  println!("indices:\n{indices}\n");

  let source = Tensor::ones((rows, 1), DType::F32, &device)?;

  // "Writes all values from the tensor src into self at the indices specified
  // in the index tensor. For each value in src, its output index is specified
  // by its index in src for dimension != dim and by the corresponding value in
  // index for dimension = dim."
  // https://docs.pytorch.org/docs/main/generated/
  //   torch.Tensor.scatter_.html#torch.Tensor.scatter_

  let target_onehot = target_onehot.scatter(&indices, &source, 1)?;

  println!("target_onehot:\n{target_onehot}\n");

  let random_indices = Tensor::rand(0f32, rows as f32, &[5], &device)?;

  let random_indices = random_indices.to_dtype(DType::U32)?;

  println!("random_indices:\n{random_indices}\n");

  let random_scores = scores.i(&random_indices)?;

  println!("random_scores:\n{random_scores}\n");

  let random_scores_onehot = target_onehot.i(&random_indices)?;

  println!("random_scores_onehot:\n{random_scores_onehot}\n");

  // page 85

  // "target_unsqueezed" in the book is "indices" here

  println!("indices:\n{indices}\n");

  Ok(())
}
