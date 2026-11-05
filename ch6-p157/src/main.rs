use ::candle_core::{DType, Device, IndexOp, Result, Tensor};
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{Linear, Module, VarBuilder, VarMap};
use ::rand::prelude::*;
use ::rand::rng;

fn main() -> ::anyhow::Result<()> {
  let device: Device = Device::Cpu;

  let t_c_array: [f64; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let t_u_array: [f64; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let t_c: Tensor = Tensor::new(&t_c_array, &device)?;

  dbg!(&t_c);

  let t_u: Tensor = Tensor::new(&t_u_array, &device)?;

  dbg!(&t_u);

  let t_c: Tensor = t_c.unsqueeze(1)?;

  dbg!(&t_c);

  let t_u: Tensor = t_u.unsqueeze(1)?;

  dbg!(&t_u);

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

  // page 157

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    var_builder_args.pp("linear");

  let linear_model: Linear = ::candle_nn::linear(1, 1, var_builder_args)?;

  println!("\nlinear_model:\n{:?}\n", linear_model);

  let output: Tensor = linear_model.forward(&t_un_val)?;

  println!("output:\n{output}\n");

  // page 158

  let weight: &Tensor = linear_model.weight();

  println!("weight:\n{weight}\n");

  let bias: &Tensor = linear_model.bias().unwrap();

  println!("bias:\n{bias}\n");

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
