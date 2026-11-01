use ::candle_core::backprop::GradStore;
use ::candle_core::{Device, IndexOp, Result, Tensor, Var};
use ::candle_nn::{Optimizer, SGD};

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

  // page 133

  let params_var: &Var = &Var::new(
    &[
      1., 0.,
    ],
    &device,
  )?;

  dbg!(params_var);

  let params_var_tensor: &Tensor = params_var.as_tensor();

  let learning_rate: f64 = 1e-5;

  let mut optimizer: SGD = SGD::new(vec![params_var.clone()], learning_rate)?;

  let w_var_tensor: Tensor = params_var_tensor.i(0)?;

  let b_var_tensor: Tensor = params_var_tensor.i(1)?;

  let t_p: Tensor = model(&t_u, &w_var_tensor, &b_var_tensor)?;

  let loss: Tensor = loss_fn(&t_p, &t_c)?;

  let grad_store: GradStore = loss.backward()?;

  // page 134

  optimizer.step(&grad_store)?;

  dbg!(params_var);

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
