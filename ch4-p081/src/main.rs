use ::anyhow::Result;
use ::candle_core::{DType, Device, IndexOp, Tensor};
use ::csv::{Reader, ReaderBuilder, StringRecord};
use ::std::fs::File;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  // page 81

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let wine_path: PathBuf = Path::new(cargo_manifest_dir)
    .join("tabular-wine")
    .join("winequality-white.csv");

  let mut file_reader: Reader<File> = ReaderBuilder::new()
    .has_headers(true)
    .delimiter(b';')
    .from_path(wine_path)?;

  let col_list: &StringRecord = file_reader.headers()?;

  println!("\n{col_list:?}");

  // page 82

  let columns: usize = col_list.len();

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

  let device: Device = Device::Cpu;

  let wineq: Tensor = Tensor::from_vec(f32_vec, (rows, columns), &device)?;

  println!("wineq:\n{wineq:?}\n");

  // page 83

  let rank: usize = wineq.rank();

  let column_dimension_index: usize = rank - 1;

  let column_dimension_size: usize = wineq.dim(column_dimension_index)?;

  let score_column_index: usize = column_dimension_size - 1;

  let data: Tensor = wineq.i((.., ..score_column_index))?;

  println!("data:\n{data}\n");

  let target: Tensor = wineq.i((.., score_column_index))?;

  let target: Tensor = target.to_dtype(DType::U8)?;

  println!("target:\n{target}\n");

  // page 84

  let target_onehot = Tensor::zeros((rows, 10), DType::F32, &device)?;

  println!("\ntarget_onehot:\n{target_onehot}\n");

  let target_unsqueezed = target.unsqueeze(1)?;

  println!("target_unsqueezed:\n{target_unsqueezed}\n");

  let source = Tensor::ones((rows, 1), DType::F32, &device)?;

  // "Writes all values from the tensor src into self at the indices specified
  // in the index tensor. For each value in src, its output index is specified
  // by its index in src for dimension != dim and by the corresponding value in
  // index for dimension = dim."
  // https://docs.pytorch.org/docs/main/generated/
  //   torch.Tensor.scatter_.html#torch.Tensor.scatter_

  let target_onehot = target_onehot.scatter(&target_unsqueezed, &source, 1)?;

  println!("target_onehot:\n{target_onehot}\n");

  let random_indices = Tensor::rand(0f32, rows as f32, &[5], &device)?;

  let random_indices = random_indices.to_dtype(DType::U32)?;

  println!("random_indices:\n{random_indices}\n");

  let random_scores = target.i(&random_indices)?;

  println!("random_scores:\n{random_scores}\n");

  let random_scores_onehot = target_onehot.i(&random_indices)?;

  println!("random_scores_onehot:\n{random_scores_onehot}\n");

  // page 85

  println!("target_unsqueezed:\n{target_unsqueezed}\n");

  // page 85 section 4.3.5

  let data_mean = data.mean(0)?;

  println!("data_mean:\n{data_mean}\n");

  // page 86

  let data_var = data.var(0)?;

  println!("data_var:\n{data_var}\n");

  let data_deviation = data.broadcast_sub(&data_mean)?;

  println!("data_deviation:\n{data_deviation}\n");

  let data_standard_deviation = data_var.sqrt()?;

  println!("data_standard_deviation:\n{data_standard_deviation}\n");

  let data_normalized =
    data_deviation.broadcast_div(&data_standard_deviation)?;

  println!("data_normalized:\n{data_normalized}\n");

  // page 87

  let bad_indexes = target.le(3u8)?;

  println!("bad_indexes:\n{bad_indexes}\n");

  let bad_indexes_sum = bad_indexes.sum(0)?;

  println!("bad_indexes_sum:\n{bad_indexes_sum}\n");

  let scores: Vec<u8> = target.to_vec1()?;

  let bad_indices_vec: Vec<u32> = scores
    .iter()
    .enumerate()
    .filter_map(|(index, &score): (usize, &u8)| {
      if score <= 3 {
        Some(index as u32)
      } else {
        None
      }
    })
    .collect();

  let bad_indices: Tensor = Tensor::new(bad_indices_vec, &device)?;

  println!("bad_indices:\n{bad_indices}\n");

  let bad_data_with_score = wineq.index_select(&bad_indices, 0)?;

  println!("bad_data_with_score:\n{bad_data_with_score}\n");

  let data_contiguous = data.contiguous()?;

  let bad_data = data_contiguous.index_select(&bad_indices, 0)?;

  println!("bad_data:\n{bad_data}\n");

  Ok(())
}
