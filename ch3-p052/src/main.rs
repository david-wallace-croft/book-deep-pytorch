#![doc = include_str!("../../README.md")]

use ::candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 52 section 3.6.3

  let double_points = Tensor::ones((10, 2), DType::F64, &device)?;

  println!("double_points.dtype: {:?}: ", double_points.dtype());

  let short_points = Tensor::new(
    &[
      [
        1i16, 2,
      ],
      [
        3, 4,
      ],
    ],
    &device,
  )?;

  // page 53

  println!("short_points.dtype: {:?}", short_points.dtype());

  let double_points =
    Tensor::ones((10, 2), DType::F32, &device)?.to_dtype(DType::F64)?;

  println!("double_points.dtype: {:?}", double_points.dtype());

  let short_points =
    Tensor::ones((10, 2), DType::I32, &device)?.to_dtype(DType::I16)?;

  println!("short_points.dtype: {:?}", short_points.dtype());

  let points_64 = Tensor::rand(0., 1., 5, &device)?;

  println!("points_64: {points_64:?}");

  let points_32 = Tensor::rand(0f32, 1., 5, &device)?;

  println!("points_32: {points_32:?}");

  let points_short = points_64.to_dtype(DType::I16)?;

  println!("points_short: {points_short:?}");

  // Causes a run-time error
  // let product = points_64.broadcast_mul(&points_short)?;

  let product = points_64.broadcast_mul(&points_short.to_dtype(DType::F64)?)?;

  println!("product: {product:?}");

  let a = Tensor::ones((3, 2), DType::F32, &device)?;

  let a_t = a.transpose(0, 1)?;

  // page 54

  println!("a.shape: {:?}", a.shape());

  println!("a_t.shape: {:?}", a_t.shape());

  Ok(())
}
