use ::candle_core::backprop::GradStore;
use ::candle_core::{Device, IndexOp, Result, Tensor, Var};
use ::candle_nn::{AdamW, Optimizer, ParamsAdamW};

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

  // page 135

  let params_var: &Var = &Var::new(
    &[
      1., 0.,
    ],
    &device,
  )?;

  dbg!(params_var);

  let params_adam_w: ParamsAdamW = ParamsAdamW {
    lr: 1e-1,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,
  };

  let optimizer: AdamW = AdamW::new(vec![params_var.clone()], params_adam_w)?;

  let n_epochs: usize = 2_000;

  let params_trained: Var =
    training_loop(n_epochs, optimizer, params_var, &t_u, &t_c)?;

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

fn training_loop(
  n_epochs: usize,
  mut optimizer: AdamW,
  params: &Var,
  t_u: &Tensor,
  t_c: &Tensor,
) -> Result<Var> {
  // page 134

  let print_interval: usize = n_epochs.div_ceil(10);

  for epoch in 1..=n_epochs {
    let w: Tensor = params.i(0)?;

    let b: Tensor = params.i(1)?;

    let t_p: Tensor = model(t_u, &w, &b)?;

    let loss: Tensor = loss_fn(&t_p, t_c)?;

    let grad_store: GradStore = loss.backward()?;

    optimizer.step(&grad_store)?;

    let loss_scalar: f64 = loss.to_scalar()?;

    if epoch % print_interval == 0 {
      println!("Epoch {epoch}, Loss {loss_scalar:.6}");

      if !loss_scalar.is_normal() {
        break;
      }
    }
  }

  Ok(params.clone())
}
