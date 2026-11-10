#![expect(unused)]

use ::candle_core::{DType, Device, IndexOp, Result, Shape, Tensor, Var, bail};
use ::candle_nn::loss;
use ::candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use ::candle_nn::{
  Linear, Module, Optimizer, SGD, Sequential, VarBuilder, VarMap,
};
use ::rand::prelude::*;
use ::rand::rng;
use ::std::collections::HashMap;
use ::std::sync::MutexGuard;

fn main() -> Result<()> {
  let device: Device = Device::Cpu;

  // page 164

  let var_map: VarMap = VarMap::new();

  let var_builder_args: VarBuilderArgs<'_, Box<dyn SimpleBackend>> =
    VarBuilder::from_varmap(&var_map, DType::F64, &device);

  // Create the Sequential model directly
  let seq_model: Sequential = ::candle_nn::seq()
    .add(::candle_nn::linear(
      1,
      13,
      var_builder_args.pp("hidden_linear"),
    )?)
    .add_fn(|xs: &Tensor| xs.tanh())
    .add(::candle_nn::linear(
      13,
      1,
      var_builder_args.pp("output_linear"),
    )?);

  let var_vec: Vec<Var> = var_map.all_vars();

  let parameter_shapes: Vec<&Shape> =
    var_vec.iter().map(|var: &Var| var.shape()).collect();

  println!("\nParameter shapes:\n{:?}\n", parameter_shapes);

  // page 165

  for (name, var) in var_map.data().lock().unwrap().iter() {
    println!("{}: {:?}", name, var.shape());
  }

  // page 166

  let data: MutexGuard<'_, HashMap<String, Var>> =
    var_map.data().lock().unwrap();

  let output_bias_option: Option<&Var> = data.get("output_linear.bias");

  let Some(output_bias) = output_bias_option else {
    bail!("not found");
  };

  println!("\noutput_bias:\n{output_bias}\n");

  // The following is not in the book

  let input: Tensor = Tensor::randn(0., 1., (5, 1), &device)?;

  let output: Tensor = seq_model.forward(&input)?;

  println!("Output:\n{output}\n");

  Ok(())
}
