#![expect(unused)]

use ::candle_core::{DType, Device, Result, Tensor};

const DELTA: f32 = 0.1;
const LEARNING_RATE: f32 = 1e-2;

fn main() -> Result<()> {
  // page 114

  let t_c: [f32; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let t_u: [f32; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let device: Device = Device::Cpu;

  let t_c = Tensor::new(&t_c, &device)?;

  println!("t_c:\n{t_c}\n");

  let t_u = Tensor::new(&t_u, &device)?;

  println!("t_u:\n{t_u}\n");

  let w = Tensor::ones(&[], DType::F32, &device)?;

  println!("w:\n{w}\n");

  let b = Tensor::zeros(&[], DType::F32, &device)?;

  println!("b:\n{b}\n");

  let t_p = model(&t_u, &w, &b)?;

  println!("t_p:\n{t_p}\n");

  let loss = loss_fn(&t_p, &t_c)?;

  println!("loss:\n{loss}\n");

  // page 117

  let loss_rate_of_change_w =
    central_difference_approximation(&b, &device, &t_c, &t_u, &w)?;

  println!("loss_rate_of_change_w:\n{loss_rate_of_change_w}\n");

  // page 118

  let learning_rate = Tensor::new(&[LEARNING_RATE], &device)?;

  println!("learning_rate:\n{learning_rate}\n");

  let w_delta = loss_rate_of_change_w.broadcast_mul(&learning_rate)?;

  println!("w_delta:\n{w_delta}\n");

  let w = w.broadcast_sub(&w_delta)?;

  println!("w:\n{w}\n");

  Ok(())
}

fn central_difference_approximation(
  b: &Tensor,
  device: &Device,
  t_c: &Tensor,
  t_u: &Tensor,
  w: &Tensor,
) -> Result<Tensor> {
  let delta = Tensor::new(&[DELTA], device)?;

  let w_pos = w.broadcast_add(&delta)?;

  let w_neg = w.broadcast_sub(&delta)?;

  let model_pos = model(t_u, &w_pos, b)?;

  let model_neg = model(t_u, &w_neg, b)?;

  let loss_pos = loss_fn(&model_pos, t_c)?;

  let loss_neg = loss_fn(&model_neg, t_c)?;

  let loss_difference = (loss_pos - loss_neg)?;

  let twice_delta = Tensor::new(&[2. * DELTA], device)?;

  loss_difference.broadcast_div(&twice_delta)
}

fn dloss_fn(
  t_p: &Tensor,
  t_c: &Tensor,
) -> Result<Tensor> {
  // page 120
  // Easier to express the equation when using f64
  // Assuming dim(0) is equivalent to PyTorch size(0)
  let dsq_diffs = (2. * (t_p - t_c)?)? / t_p.dim(0)? as f64;

  #[allow(clippy::let_and_return)]
  dsq_diffs
}

fn dmodel_db(
  _t_u: &Tensor,
  _w: &Tensor,
  _b: &Tensor,
) -> f64 {
  // page 120
  1f64
}

fn dmodel_dw<'a>(
  t_u: &'a Tensor,
  _w: &Tensor,
  _b: &Tensor,
) -> &'a Tensor {
  // page 120
  t_u
}

fn grad_fn(
  t_u: &Tensor,
  t_c: &Tensor,
  t_p: &Tensor,
  w: &Tensor,
  b: &Tensor,
) -> Result<Tensor> {
  // page 120
  let dloss_dtp: Tensor = dloss_fn(t_p, t_c)?;

  let dloss_dw: Tensor = (&dloss_dtp * dmodel_dw(t_u, w, b))?;

  let dloss_db: Tensor = (dloss_dtp * dmodel_db(t_u, w, b))?;

  let dloss_dw_sum: Tensor = dloss_dw.sum_all()?;

  let dloss_db_sum: Tensor = dloss_db.sum_all()?;

  Tensor::stack(
    &[
      dloss_dw_sum,
      dloss_db_sum,
    ],
    0,
  )
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
