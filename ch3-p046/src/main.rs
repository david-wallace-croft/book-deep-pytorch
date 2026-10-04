#![doc = include_str!("../../README.md")]

use ::candle_core::{Device, IndexOp, Result, Shape, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 46 section 3.3

  let some_list: Tensor = Tensor::arange(0, 6, &device)?;

  println!("some_list: {some_list}\n");

  println!("some_list[:]: {}\n", some_list.i(..)?);

  println!("some_list[1:4]: {}\n", some_list.i(1..4)?);

  println!("some_list[1:]: {}\n", some_list.i(1..)?);

  println!("some_list[:4]: {}\n", some_list.i(..4)?);

  // println!("some_list[:-1]: {}", some_list.i(..-1)?);

  // println!("some_list[1:4:2]: {}", some_list.i(((1..4).step_by(2)))?);

  let points: Tensor = Tensor::new(
    &[
      [
        4f32, 1f32,
      ],
      [
        5f32, 3f32,
      ],
      [
        2f32, 1f32,
      ],
    ],
    &device,
  )?;

  println!("points: {points}\n");

  println!("points[1:]: {}\n", points.i(1..)?);

  println!("points[1:, :]: {}\n", points.i((1.., ..))?);

  println!("points[1:, 0]: {}\n", points.i((1.., 0))?);

  let shape: &Shape = points.shape();

  let dimension_sizes: &[usize] = shape.dims();

  println!("dimension sizes: {:#?}\n", dimension_sizes);

  let unsqueezed_points: Tensor = points.unsqueeze(0)?;

  println!("points[None]: {unsqueezed_points}\n");

  let shape: &Shape = unsqueezed_points.shape();

  let dimension_sizes: &[usize] = shape.dims();

  println!("dimension sizes: {:#?}\n", dimension_sizes);

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() -> Result<()> {
    let device: Device = Device::Cpu;

    let points: Tensor = Tensor::new(
      &[
        [
          4f32, 1f32,
        ],
        [
          5f32, 3f32,
        ],
        [
          2f32, 1f32,
        ],
      ],
      &device,
    )?;

    let shape: &Shape = points.shape();

    let dimension_sizes: &[usize] = shape.dims();

    assert_eq!(
      dimension_sizes,
      &[
        3, 2,
      ],
    );

    let unsqueezed_points: Tensor = points.unsqueeze(0)?;

    let shape: &Shape = unsqueezed_points.shape();

    let dimension_sizes: &[usize] = shape.dims();

    assert_eq!(
      dimension_sizes,
      &[
        1, 3, 2,
      ],
    );

    Ok(())
  }
}
