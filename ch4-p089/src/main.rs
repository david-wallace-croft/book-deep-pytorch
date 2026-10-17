#![expect(unused)]
use ::anyhow::Result;
use ::candle_core::{DType, Device, IndexOp, Tensor};
use ::csv::{Reader, ReaderBuilder, StringRecord};
use ::std::fs::File;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  // page 89

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let bikes_path: PathBuf = Path::new(cargo_manifest_dir)
    .join("bike-sharing-dataset")
    .join("hour-fixed.csv");

  let mut file_reader: Reader<File> = ReaderBuilder::new()
    .has_headers(true)
    .delimiter(b',')
    .from_path(bikes_path)?;

  let col_list_string_record: &StringRecord = file_reader.headers()?;

  let col_list: Vec<String> = col_list_string_record
    .iter()
    .map(|s: &str| s.to_string())
    .collect();

  println!("\n{col_list:?}\n");

  let columns: usize = col_list.len();

  let mut f32_vec: Vec<f32> = Vec::new();

  let mut rows: usize = 0;

  for result in file_reader.records() {
    let record: StringRecord = result?;

    rows += 1;

    for (index, field) in record.iter().enumerate() {
      if index == 1 {
        let value: String = field.parse()?;

        let day_of_month_str = &value[8..10];

        let day_of_month_f32: f32 = day_of_month_str.parse()?;

        f32_vec.push(day_of_month_f32);
      } else {
        let value: f32 = field.parse()?;

        f32_vec.push(value);
      }
    }
  }

  let device: Device = Device::Cpu;

  let bikes: Tensor = Tensor::from_vec(f32_vec, (rows, columns), &device)?;

  println!("bikes:\n{bikes}\n");

  Ok(())
}
