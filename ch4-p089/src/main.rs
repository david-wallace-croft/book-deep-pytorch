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

  // page 92

  println!("bikes.shape:\n{:?}\n", bikes.shape());

  println!("bikes.stride:\n{:?}\n", bikes.stride());

  // page 93

  let daily_bikes = bikes.reshape((rows / 24, 24, columns))?;

  println!("daily_bikes.shape:\n{:?}\n", daily_bikes.shape());

  println!("daily_bikes.stride:\n{:?}\n", daily_bikes.stride());

  let daily_bikes = daily_bikes.transpose(1, 2)?;

  println!("daily_bikes.shape:\n{:?}\n", daily_bikes.shape());

  println!("daily_bikes.stride:\n{:?}\n", daily_bikes.stride());

  // page 94

  let first_day = bikes.i(..24)?.to_dtype(DType::I64)?;

  println!("first_day:\n{first_day}\n");

  let weather_onehot =
    Tensor::zeros((first_day.dim(0)?, 4), DType::F32, &device)?;

  let first_day_weathersit = first_day.i((.., 9))?;

  println!("first_day[:,9]:\n{first_day_weathersit}\n");

  let first_day_weathersit_unsqueezed = first_day_weathersit.unsqueeze(1)?;

  println!(
    "first_day_weathersit_unsqueezed:\n{:?}\n",
    first_day_weathersit_unsqueezed
  );

  let one = Tensor::new(1_i64, &device)?;

  let indices = first_day_weathersit_unsqueezed
    .to_dtype(DType::I64)?
    .broadcast_sub(&one)?;

  println!("indices:\n{indices}\n");

  let source = Tensor::ones((24, 1), DType::F32, &device)?;

  println!("source:\n{source}\n");

  let weather_onehot = weather_onehot.scatter(&indices, &source, 1)?;

  println!("weather_onehot:\n{weather_onehot}\n");

  let concatenated = Tensor::cat(
    &[
      &(bikes.i(..24)?),
      &weather_onehot,
    ],
    1,
  )?
  .i(..1)?;

  println!("concatenated:\n{concatenated}\n");

  // page 95

  let daily_weather_onehot = Tensor::zeros(
    (daily_bikes.dim(0)?, 4, daily_bikes.dim(2)?),
    DType::F32,
    &device,
  )?;

  println!(
    "daily_weather_onehot.shape:\n{:?}\n",
    daily_weather_onehot.shape()
  );

  let daily_bikes_weathersit =
    daily_bikes.i((.., 9, ..))?.to_dtype(DType::I64)?;

  println!("daily_bikes[:,9,:]:\n{daily_bikes_weathersit}\n");

  let daily_bikes_weathersit_unsqueezed =
    daily_bikes_weathersit.unsqueeze(1)?;

  println!(
    "daily_bikes_weathersit_unsqueezed:\n{:?}\n",
    daily_bikes_weathersit_unsqueezed
  );

  let one = Tensor::new(1_i64, &device)?;

  let indices = daily_bikes_weathersit_unsqueezed
    .to_dtype(DType::I64)?
    .broadcast_sub(&one)?;

  println!("indices:\n{indices}\n");

  let source = Tensor::ones((730, 1, 24), DType::F32, &device)?;

  println!("source:\n{source}\n");

  let daily_weather_onehot =
    daily_weather_onehot.scatter(&indices, &source, 1)?;

  println!(
    "daily_weather_onehot.shape:\n{:?}\n",
    daily_weather_onehot.shape()
  );

  let daily_bikes = Tensor::cat(
    &[
      &daily_bikes,
      &daily_weather_onehot,
    ],
    1,
  )?;

  println!("daily_bikes:\n{daily_bikes}\n");

  let normalized_weather =
    ((daily_bikes.i((.., 9, ..))? - 1.0)? / 3.0)?.unsqueeze(1)?;

  println!("normalized_weather:\n{normalized_weather}\n");

  let daily_bikes = daily_bikes.contiguous()?;

  daily_bikes.slice_set(&normalized_weather, 1, 9)?;

  println!("daily_bikes:\n{daily_bikes}\n");

  // page 96

  let temp = daily_bikes.i((.., 10, ..))?;

  println!("temp:\n{temp}\n");

  let temp_min = temp.min_all()?;

  println!("temp_min:\n{temp_min}\n");

  let temp_max = temp.max_all()?;

  println!("temp_max:\n{temp_max}\n");

  let normalized_temp = daily_bikes
    .i((.., 10, ..))?
    .broadcast_sub(&temp_min)?
    .broadcast_div(&(temp_max - temp_min)?)?
    .unsqueeze(1)?;

  println!("normalized_temp:\n{normalized_temp}\n");

  daily_bikes.slice_set(&normalized_temp, 1, 10)?;

  println!("daily_bikes:\n{daily_bikes}\n");

  let temp_mean = temp.mean_all()?;

  println!("temp_mean:\n{temp_mean}\n");

  let temp_std = temp.flatten_all()?.var(0)?.sqrt()?;

  println!("temp_std:\n{temp_std}\n");

  let normalized_temp = temp
    .broadcast_sub(&temp_mean)?
    .broadcast_div(&temp_std)?
    .unsqueeze(1)?;

  println!("normalized_temp:\n{normalized_temp}\n");

  daily_bikes.slice_set(&normalized_temp, 1, 10)?;

  println!("daily_bikes:\n{daily_bikes}\n");

  Ok(())
}
