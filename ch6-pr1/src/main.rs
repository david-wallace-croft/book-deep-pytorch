use ::anyhow::anyhow;
use ::candle_core::{DType, Device, IndexOp, Result, Tensor, Var};
use ::candle_nn::loss;
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{Module, Optimizer, SGD, Sequential, VarBuilder, VarMap};
use ::plotters::prelude::*;
use ::rand::prelude::*;
use ::rand::rng;
use ::std::collections::HashMap;
use ::std::path::{Path, PathBuf};
use ::std::sync::MutexGuard;

fn main() -> ::anyhow::Result<()> {
  let device: Device = Device::Cpu;

  // page 167 problem 1

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  let seq_model: Sequential = ::candle_nn::seq()
    .add(::candle_nn::linear(
      1,
      32,
      var_builder_args.pp("hidden_linear"),
    )?)
    .add_fn(|xs: &Tensor| xs.tanh())
    .add(::candle_nn::linear(
      32,
      1,
      var_builder_args.pp("output_linear"),
    )?);

  let var_vec: Vec<Var> = var_map.all_vars();

  println!("\nseq_model length: {}\n", seq_model.len());

  let data: MutexGuard<'_, HashMap<String, Var>> =
    var_map.data().lock().unwrap();

  let output_linear_bias_option: Option<&Var> = data.get("output_linear.bias");

  let Some(output_linear_bias) = output_linear_bias_option else {
    return Err(anyhow!("not found"));
  };

  println!("output_linear_bias:\n{output_linear_bias}\n");

  let learning_rate: f64 = 1e-5;

  let mut optimizer: SGD = SGD::new(var_vec, learning_rate)?;

  let n_epochs: usize = 10_000_000;

  let loss_fn = loss::mse;

  let (t_c_train, t_c_val, t_un_train, t_un_val, t_u, t_c_array, t_u_array) =
    make_training_data(&device)?;

  training_loop(
    n_epochs,
    &mut optimizer,
    &seq_model,
    loss_fn,
    &t_un_train,
    &t_un_val,
    &t_c_train,
    &t_c_val,
  )?;

  let output: Tensor = seq_model.forward(&t_un_val)?;

  println!("\noutput:\n{output}\n");

  println!("answer:\n{t_c_val}\n");

  let hidden_linear_weight_option: Option<&Var> =
    data.get("hidden_linear.weight");

  let Some(hidden_linear_weight) = hidden_linear_weight_option else {
    return Err(anyhow!("not found"));
  };

  println!("hidden_linear_weight:\n{hidden_linear_weight}\n");

  println!("t_u:\n{t_u}\n");

  let t_un: Tensor = (0.1 * t_u)?;

  let t_p: Tensor = seq_model.forward(&t_un)?;

  println!("t_p:\n{t_p}\n");

  let t_p_squeeze: Tensor = t_p.squeeze(1)?;

  println!("t_p_squeeze:\n{t_p_squeeze}\n");

  let t_p_vec: Vec<f64> = t_p_squeeze.to_vec1()?;

  println!("t_p_vec:\n{:?}\n", t_p_vec);

  let t_range_vec: Vec<f64> = (20..=90).map(f64::from).collect();

  let t_range: Tensor = Tensor::new(t_range_vec.as_slice(), &device)?;

  let t_range_unsqueezed: Tensor = t_range.unsqueeze(1)?;

  println!("t_range_unsqueezed:\n{t_range_unsqueezed}\n");

  let t_range_normalized: Tensor = (0.1 * t_range_unsqueezed)?;

  println!("t_range_normalized:\n{t_range_normalized}\n");

  let t_model: Tensor = seq_model.forward(&t_range_normalized)?;

  println!("t_model:\n{t_model}\n");

  let t_model_vec: Vec<f64> = t_model.squeeze(1)?.to_vec1()?;

  plot_data(&t_c_array, &t_p_vec, &t_u_array, &t_range_vec, &t_model_vec)?;

  Ok(())
}

#[expect(clippy::type_complexity)]
fn make_training_data(
  device: &Device
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, [f64; 11], [f64; 11])> {
  // page 160

  let t_c_array: [f64; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let t_u_array: [f64; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let t_c: Tensor = Tensor::new(&t_c_array, device)?;

  dbg!(&t_c);

  let t_u: Tensor = Tensor::new(&t_u_array, device)?;

  dbg!(&t_u);

  let t_c: Tensor = t_c.unsqueeze(1)?;

  dbg!(&t_c);

  let t_u: Tensor = t_u.unsqueeze(1)?;

  dbg!(&t_u);

  let n_samples: usize = t_u.dim(0)?;

  dbg!(n_samples);

  let n_val: usize = n_samples * 2 / 10;

  dbg!(n_val);

  let shuffled_indices = randperm(n_samples, device)?;

  dbg!(&shuffled_indices);

  let train_indices: Tensor =
    shuffled_indices.narrow(0, 0, n_samples - n_val)?;

  dbg!(&train_indices);

  let val_indices: Tensor =
    shuffled_indices.narrow(0, n_samples - n_val, n_val)?;

  dbg!(&val_indices);

  let t_u_train: Tensor = t_u.i(&train_indices)?;

  dbg!(&t_u_train);

  let t_c_train: Tensor = t_c.i(&train_indices)?;

  dbg!(&t_c_train);

  let t_u_val: Tensor = t_u.i(&val_indices)?;

  dbg!(&t_u_val);

  let t_c_val: Tensor = t_c.i(&val_indices)?;

  dbg!(&t_c_val);

  let t_un_train: Tensor = (0.1 * t_u_train)?;

  dbg!(&t_un_train);

  let t_un_val: Tensor = (0.1 * t_u_val)?;

  dbg!(&t_un_val);

  Ok((
    t_c_train, t_c_val, t_un_train, t_un_val, t_u, t_c_array, t_u_array,
  ))
}

fn plot_data(
  t_c_slice: &[f64],
  t_p_slice: &[f64],
  t_u_slice: &[f64],
  t_range_slice: &[f64],
  t_model_slice: &[f64],
) -> ::anyhow::Result<()> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let scatter_plot_path: PathBuf =
    Path::new(cargo_manifest_dir).join("temperature_plot.png");

  let root =
    BitMapBackend::new(&scatter_plot_path, (640, 480)).into_drawing_area();

  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
    .caption("Celsius-Fahrenheit", ("sans-serif", 25).into_font())
    .margin(50)
    .x_label_area_size(50)
    .y_label_area_size(50)
    .build_cartesian_2d(20.0..90.0, -10.0..35.)?;

  chart
    .configure_mesh()
    .x_desc("Fahrenheit")
    .y_desc("Celsius")
    .axis_desc_style(("sans-serif", 30))
    .draw()?;

  let data: Vec<(f64, f64)> = t_range_slice
    .iter()
    .zip(t_model_slice.iter())
    .map(|(u, p)| (*u, *p))
    .collect();

  chart
    .draw_series(LineSeries::new(data.iter().copied(), CYAN.stroke_width(4)))?;

  let data: Vec<(f64, f64)> = t_u_slice
    .iter()
    .zip(t_c_slice.iter())
    .map(|(u, p)| (*u, *p))
    .collect();

  chart.draw_series(
    data
      .iter()
      .map(|&(x, y)| Circle::new((x, y), 5, RED.filled())),
  )?;

  let data: Vec<(f64, f64)> = t_u_slice
    .iter()
    .zip(t_p_slice.iter())
    .map(|(u, p)| (*u, *p))
    .collect();

  chart.draw_series(data.iter().map(|&(x, y)| Cross::new((x, y), 8, BLACK)))?;

  root.present()?;

  Ok(())
}

fn randperm(
  n: usize,
  device: &Device,
) -> Result<Tensor> {
  let n_u32: u32 = u32::try_from(n)?;

  let mut indices: Vec<u32> = (0..n_u32).collect();

  let mut rng = rng();

  indices.shuffle(&mut rng);

  Tensor::new(indices.as_slice(), device)
}

#[expect(clippy::too_many_arguments)]
fn training_loop<F, M: Module, O: Optimizer>(
  n_epochs: usize,
  optimizer: &mut O,
  model: &M,
  loss_fn: F,
  t_u_train: &Tensor,
  t_u_val: &Tensor,
  t_c_train: &Tensor,
  t_c_val: &Tensor,
) -> Result<()>
where
  F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
  // page 162

  for epoch in 1..=n_epochs {
    let t_p_train: Tensor = model.forward(t_u_train)?;

    let loss_train: Tensor = loss_fn(&t_p_train, t_c_train)?;

    let t_p_val: Tensor = model.forward(t_u_val)?;

    let loss_val: Tensor = loss_fn(&t_p_val, t_c_val)?;

    optimizer.backward_step(&loss_train)?;

    let loss_train_scalar: f64 = loss_train.to_scalar()?;

    let loss_val_scalar: f64 = loss_val.to_scalar()?;

    if !loss_train_scalar.is_normal() || !loss_train_scalar.is_normal() {
      break;
    }

    if epoch == 1 || epoch % 10_000 == 0 {
      println!(
        "Epoch {}, Training Loss {:.6}, Validation Loss {:.6}",
        epoch, loss_train_scalar, loss_val_scalar
      );
    }
  }

  Ok(())
}
