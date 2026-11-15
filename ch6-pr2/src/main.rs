#![expect(unused)]

use ::anyhow::{Result, anyhow};
use ::candle_core::{DType, Device, IndexOp, Tensor, Var};
use ::candle_nn::loss;
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{
  AdamW, Module, Optimizer, ParamsAdamW, SGD, Sequential, VarBuilder, VarMap,
};
use ::csv::{Reader, ReaderBuilder, StringRecord};
use ::plotters::prelude::*;
use ::rand::prelude::*;
use ::rand::rng;
use ::std::collections::HashMap;
use ::std::fs::File;
use ::std::path::{Path, PathBuf};
use ::std::sync::MutexGuard;

fn main() -> ::anyhow::Result<()> {
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

  let mut f64_vec: Vec<f64> = Vec::new();

  let mut rows: usize = 0;

  for result in file_reader.records() {
    let record: StringRecord = result?;

    rows += 1;

    for field in &record {
      let value: f64 = field.parse()?;

      f64_vec.push(value);
    }
  }

  let device: Device = Device::Cpu;

  let wineq: Tensor = Tensor::from_vec(f64_vec, (rows, columns), &device)?;

  println!("wineq:\n{wineq:?}\n");

  // page 83

  let rank: usize = wineq.rank();

  let column_dimension_index: usize = rank - 1;

  let column_dimension_size: usize = wineq.dim(column_dimension_index)?;

  let score_column_index: usize = column_dimension_size - 1;

  let data: Tensor = wineq.i((.., ..score_column_index))?;

  println!("data:\n{data}\n");

  let target: Tensor = wineq.i((.., score_column_index))?;

  let target: Tensor = target.unsqueeze(1)?;

  println!("target:\n{target}\n");

  // page 168 problem 2

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  let seq_model: Sequential = ::candle_nn::seq()
    .add(::candle_nn::linear(
      11,
      22,
      var_builder_args.pp("hidden_linear"),
    )?)
    .add_fn(|xs: &Tensor| xs.tanh())
    .add(::candle_nn::linear(
      22,
      1,
      var_builder_args.pp("output_linear"),
    )?);

  let var_vec: Vec<Var> = var_map.all_vars();

  let params_adam_w: ParamsAdamW = ParamsAdamW {
    // beta1: 0.9,
    // beta2: 0.999,
    // eps: 1e-8,
    lr: 1e-3,
    weight_decay: 0.,
    ..Default::default()
  };

  dbg!(&params_adam_w);

  let mut optimizer: AdamW = AdamW::new(var_vec, params_adam_w)?;

  let n_epochs: usize = 1_000;

  let loss_fn = loss::mse;

  // let (t_c_train, t_c_val, t_un_train, t_un_val, t_u, t_c_array, t_u_array) =
  //   make_training_data(&device)?;

  training_loop(
    n_epochs,
    &mut optimizer,
    &seq_model,
    loss_fn,
    &data,
    &target,
    // &t_un_train,
    // &t_un_val,
    // &t_c_train,
    // &t_c_val,
  )?;

  // let output: Tensor = seq_model.forward(&t_un_val)?;

  // println!("\noutput:\n{output}\n");

  // println!("answer:\n{t_c_val}\n");

  // let hidden_linear_weight_option: Option<&Var> =
  //   data.get("hidden_linear.weight");

  // let Some(hidden_linear_weight) = hidden_linear_weight_option else {
  //   return Err(anyhow!("not found"));
  // };

  // println!("hidden_linear_weight:\n{hidden_linear_weight}\n");

  // println!("t_u:\n{t_u}\n");

  // let t_un: Tensor = (0.1 * t_u)?;

  // let t_p: Tensor = seq_model.forward(&t_un)?;

  // println!("t_p:\n{t_p}\n");

  // let t_p_squeeze: Tensor = t_p.squeeze(1)?;

  // println!("t_p_squeeze:\n{t_p_squeeze}\n");

  // let t_p_vec: Vec<f64> = t_p_squeeze.to_vec1()?;

  // println!("t_p_vec:\n{:?}\n", t_p_vec);

  // let t_range_vec: Vec<f64> = (20..=90).map(f64::from).collect();

  // let t_range: Tensor = Tensor::new(t_range_vec.as_slice(), &device)?;

  // let t_range_unsqueezed: Tensor = t_range.unsqueeze(1)?;

  // println!("t_range_unsqueezed:\n{t_range_unsqueezed}\n");

  // let t_range_normalized: Tensor = (0.1 * t_range_unsqueezed)?;

  // println!("t_range_normalized:\n{t_range_normalized}\n");

  // let t_model: Tensor = seq_model.forward(&t_range_normalized)?;

  // println!("t_model:\n{t_model}\n");

  // let t_model_vec: Vec<f64> = t_model.squeeze(1)?.to_vec1()?;

  // plot_data(&t_c_array, &t_p_vec, &t_u_array, &t_range_vec, &t_model_vec)?;

  Ok(())
}

fn training_loop<F, M: Module, O: Optimizer>(
  n_epochs: usize,
  optimizer: &mut O,
  model: &M,
  loss_fn: F,
  data: &Tensor,
  target: &Tensor,
) -> Result<()>
where
  F: Fn(&Tensor, &Tensor) -> ::candle_core::Result<Tensor>,
{
  for epoch in 1..=n_epochs {
    let predicted: Tensor = model.forward(data)?;

    let loss: Tensor = loss_fn(&predicted, target)?;

    optimizer.backward_step(&loss)?;

    let loss_scalar: f64 = loss.to_scalar()?;

    if !loss_scalar.is_normal() {
      break;
    }

    if epoch == 1 || epoch % 100 == 0 {
      println!("Epoch {epoch}, Training Loss {loss_scalar:.6}");
    }
  }

  Ok(())
}
