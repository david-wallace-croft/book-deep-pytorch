use ::candle_core::{DType, Device, IndexOp, Result, Tensor, Var};
use ::candle_nn::loss;
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{Linear, Module, Optimizer, SGD, VarBuilder, VarMap};
use ::rand::prelude::*;
use ::rand::rng;

fn main() -> Result<()> {
  let device: Device = Device::Cpu;

  let (t_c_train, t_c_val, t_un_train, t_un_val) = make_training_data(&device)?;

  // page 161

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    var_builder_args.pp("linear");

  let linear_model: Linear = ::candle_nn::linear(1, 1, var_builder_args)?;

  println!("\nlinear_model:\n{:?}\n", linear_model);

  let parameters: Vec<Var> = var_map.all_vars();

  dbg!(&parameters);

  // Using a learning rate of 1e-4 instead of 1e-2 because momentum is not used.
  // Quoting from SGD source code: "Contrary to the PyTorch implementation of
  // SGD, this version does not support momentum"
  let learning_rate: f64 = 1e-4;

  let mut optimizer: SGD = SGD::new(parameters, learning_rate)?;

  // page 162

  // Increased the number of epochs to compensate for not using momentum
  let n_epochs: usize = 300_000;

  let loss_fn = loss::mse;

  training_loop(
    n_epochs,
    &mut optimizer,
    &linear_model,
    loss_fn,
    &t_un_train,
    &t_un_val,
    &t_c_train,
    &t_c_val,
  )?;

  let weight = linear_model.weight();

  println!("\nweight:\n{weight}\n");

  let bias = linear_model.bias().unwrap();

  println!("\nbias:\n{bias}\n");

  Ok(())
}

fn make_training_data(
  device: &Device
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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

  Ok((t_c_train, t_c_val, t_un_train, t_un_val))
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
