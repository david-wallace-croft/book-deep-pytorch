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

  let col_list_string_record: &StringRecord = file_reader.headers()?;

  let col_list: Vec<String> = col_list_string_record
    .iter()
    .map(|s: &str| s.to_string())
    .collect();

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

  let mid_indices_vec: Vec<u32> = scores
    .iter()
    .enumerate()
    .filter_map(|(index, &score): (usize, &u8)| {
      if score > 3 && score < 7 {
        Some(index as u32)
      } else {
        None
      }
    })
    .collect();

  let mid_indices = Tensor::new(mid_indices_vec, &device)?;

  let mid_data = data_contiguous.index_select(&mid_indices, 0)?;

  let good_indices_vec: Vec<u32> = scores
    .iter()
    .enumerate()
    .filter_map(|(index, &score): (usize, &u8)| {
      if score >= 7 {
        Some(index as u32)
      } else {
        None
      }
    })
    .collect();

  let good_indices = Tensor::new(good_indices_vec, &device)?;

  let good_data = data_contiguous.index_select(&good_indices, 0)?;

  let bad_mean = bad_data.mean(0)?;

  println!("bad_mean:\n{bad_mean}\n");

  let mid_mean = mid_data.mean(0)?;

  println!("mid_mean:\n{mid_mean}\n");

  let good_mean = good_data.mean(0)?;

  println!("good_mean:\n{good_mean}\n");

  let bad_mean_vec = bad_mean.to_vec1::<f32>()?;

  let mid_mean_vec = mid_mean.to_vec1::<f32>()?;

  let good_mean_vec = good_mean.to_vec1::<f32>()?;

  col_list
    .iter()
    .zip(bad_mean_vec.iter())
    .zip(mid_mean_vec.iter())
    .zip(good_mean_vec.iter())
    .enumerate()
    .for_each(|(index, (((feature, bad_mean), mid_mean), good_mean))| {
      println!(
        "{:2} {:20} {:6.2} {:6.2} {:6.2}",
        index, feature, bad_mean, mid_mean, good_mean
      )
    });

  // page 88

  let total_sulfur_threshold = mid_mean_vec[6];

  println!("\ntotal_sulfur_threshold: {total_sulfur_threshold:.2}\n");

  let total_sulfur_data = data_contiguous.i((.., 6))?;

  let predicted_indexes = total_sulfur_data.lt(total_sulfur_threshold)?;

  println!("predicted_indexes:\n{predicted_indexes}\n");

  let predicted_indexes_sum = predicted_indexes.to_dtype(DType::U32)?.sum(0)?;

  println!("predicted_indexes_sum:\n{predicted_indexes_sum}\n");

  let actual_indexes = target.gt(5)?;

  println!("actual_indexes:\n{actual_indexes}\n");

  let actual_indexes_sum = actual_indexes.to_dtype(DType::U32)?.sum(0)?;

  println!("actual_indexes_sum:\n{actual_indexes_sum}\n");

  let overlap: Tensor = (&actual_indexes * &predicted_indexes)?;

  println!("overlap:\n{overlap}\n");

  let n_matches: u32 = overlap.to_dtype(DType::U32)?.sum(0)?.to_scalar()?;

  println!("n_matches: {n_matches}\n");

  let n_predicted: u32 = predicted_indexes
    .to_dtype(DType::U32)?
    .sum(0)?
    .to_scalar()?;

  println!("n_predicted: {n_predicted}\n");

  println!(
    "n_matches / n_predicted: {}\n",
    n_matches as f32 / n_predicted as f32
  );

  let n_actual: u32 =
    actual_indexes.to_dtype(DType::U32)?.sum(0)?.to_scalar()?;

  println!("n_actual: {n_actual}\n");

  println!(
    "n_matches / n_actual: {}\n",
    n_matches as f32 / n_actual as f32
  );

  Ok(())
}
