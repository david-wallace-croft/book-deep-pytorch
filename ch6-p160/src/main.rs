use ::candle_core::{DType, Device, Tensor, Var};
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{Linear, VarBuilder, VarMap};
use ::candle_nn::{Optimizer, SGD};

fn main() -> ::anyhow::Result<()> {
  let device: Device = Device::Cpu;

  // page 160

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

  // page 161

  dbg!(&t_u.shape());

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    var_builder_args.pp("linear");

  let linear_model: Linear = ::candle_nn::linear(1, 1, var_builder_args)?;

  println!("\nlinear_model:\n{:?}\n", linear_model);

  let parameters: Vec<Var> = var_map.all_vars();

  dbg!(&parameters);

  let learning_rate: f64 = 1e-5;

  let _optimizer: SGD = SGD::new(parameters, learning_rate)?;

  Ok(())
}
