use ::candle_core::backprop::GradStore;
use ::candle_core::{Device, IndexOp, Result, Tensor, Var};
use ::candle_nn::{Optimizer, SGD};
use ::rand::prelude::*;
use ::rand::rng;

fn main() -> ::anyhow::Result<()> {
  let t_c_array: [f64; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let t_u_array: [f64; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let device: Device = Device::Cpu;

  let t_c: Tensor = Tensor::new(&t_c_array, &device)?;

  dbg!(&t_c);

  let t_u: Tensor = Tensor::new(&t_u_array, &device)?;

  dbg!(&t_u);

  // page 139

  let n_samples: usize = t_u.dim(0)?;

  dbg!(n_samples);

  let n_val: usize = n_samples * 2 / 10;

  dbg!(n_val);

  let shuffled_indices = randperm(n_samples, &device)?;

  dbg!(&shuffled_indices);

  let train_indices: Tensor =
    shuffled_indices.narrow(0, 0, n_samples - n_val)?;

  dbg!(&train_indices);

  let val_indices: Tensor =
    shuffled_indices.narrow(0, n_samples - n_val, n_val)?;

  dbg!(&val_indices);

  let train_t_u: Tensor = t_u.i(&train_indices)?;

  dbg!(&train_t_u);

  let train_t_c: Tensor = t_c.i(&train_indices)?;

  dbg!(&train_t_c);

  let val_t_u: Tensor = t_u.i(&val_indices)?;

  dbg!(&val_t_u);

  let val_t_c: Tensor = t_c.i(&val_indices)?;

  dbg!(&val_t_c);

  // page 140

  let train_t_un: Tensor = (0.1 * train_t_u)?;

  dbg!(&train_t_un);

  let val_t_un: Tensor = (0.1 * val_t_u)?;

  dbg!(&val_t_un);

  let params_var: &Var = &Var::new(
    &[
      1., 0.,
    ],
    &device,
  )?;

  dbg!(params_var);

  // Using a learning rate of 1e-5 instead of 1e-2 because momentum is not used.
  // Quoting from SGD source code: "Contrary to the PyTorch implementation of
  // SGD, this version does not support momentum"
  let learning_rate: f64 = 1e-5;

  let optimizer: SGD = SGD::new(vec![params_var.clone()], learning_rate)?;

  // Increased the number of epochs to compensate for not using momentum
  let n_epochs: usize = 1_500_000;

  let params_trained: Var = training_loop(
    n_epochs,
    optimizer,
    params_var,
    &train_t_un,
    &val_t_un,
    &train_t_c,
    &val_t_c,
  )?;

  dbg!(params_trained);

  Ok(())
}

fn loss_fn(
  t_p: &Tensor,
  t_c: &Tensor,
) -> Result<Tensor> {
  let squared_diffs = (t_p - t_c)?.sqr()?;

  squared_diffs.mean_all()
}

fn model(
  t_u: &Tensor,
  w: &Tensor,
  b: &Tensor,
) -> Result<Tensor> {
  t_u.broadcast_mul(w)?.broadcast_add(b)
}

fn randperm(
  n: usize,
  device: &Device,
) -> Result<Tensor> {
  let mut indices: Vec<u32> = (0..n as u32).collect();

  let mut rng = rng();

  indices.shuffle(&mut rng);

  Tensor::new(indices.as_slice(), device)
}

fn training_loop(
  n_epochs: usize,
  mut optimizer: SGD,
  params: &Var,
  train_t_u: &Tensor,
  val_t_u: &Tensor,
  train_t_c: &Tensor,
  val_t_c: &Tensor,
) -> Result<Var> {
  // page 140

  for epoch in 1..=n_epochs {
    let w: Tensor = params.i(0)?;

    let b: Tensor = params.i(1)?;

    let train_t_p: Tensor = model(train_t_u, &w, &b)?;

    let train_loss: Tensor = loss_fn(&train_t_p, train_t_c)?;

    let val_t_p: Tensor = model(val_t_u, &w, &b)?;

    let val_loss: Tensor = loss_fn(&val_t_p, val_t_c)?;

    let grad_store: GradStore = train_loss.backward()?;

    optimizer.step(&grad_store)?;

    let train_loss_scalar: f64 = train_loss.to_scalar()?;

    let val_loss_scalar: f64 = val_loss.to_scalar()?;

    if epoch <= 3 || epoch % 500 == 0 {
      println!(
        "Epoch {}, Training Loss {:.6}, Validation Loss {:.6}",
        epoch, train_loss_scalar, val_loss_scalar
      );

      if !train_loss_scalar.is_normal() || !val_loss_scalar.is_normal() {
        break;
      }
    }
  }

  Ok(params.clone())
}
