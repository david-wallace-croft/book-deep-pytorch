use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let points: Tensor = Tensor::zeros(6, DType::F32, &device)?;

  println!("points: {points}\n");

  let value: Tensor = Tensor::new(
    &[
      4f32, 1f32, 5f32, 3f32, 2f32, 1f32,
    ],
    &device,
  )?;

  #[expect(clippy::single_range_in_vec_init)]
  let points: Tensor = points.slice_assign(&[0..6], &value)?;

  println!("points: {points}\n");

  let p0: f32 = points.i(0)?.to_scalar()?;

  let p1: f32 = points.i(1)?.to_scalar()?;

  println!("float(points[0]), float(points[1]): ({p0}, {p1})\n");

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

  let shape: &Shape = points.shape();

  println!("points.shape: {shape:?}\n");

  let points = Tensor::zeros((3, 2), DType::F32, &device)?;

  println!("points: {points}\n");

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

  let p01: Tensor = points.i((0, 1))?;

  println!("points[0, 1]: {p01}\n");

  let p0: Tensor = points.i(0)?;

  println!("points[0]: {p0}\n");

  // TODO: update unit test

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() -> Result<()> {
    let device: Device = Device::cuda_if_available(0)?;

    println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

    let points: Tensor = Tensor::zeros(6, DType::F32, &device)?;

    println!("points: {points}\n");

    let value: Tensor = Tensor::new(
      &[
        4f32, 1f32, 5f32, 3f32, 2f32, 1f32,
      ],
      &device,
    )?;

    let points: Tensor = points.slice_assign(&[0..6], &value)?;

    println!("points: {points}\n");

    let expected_tensor: Tensor = Tensor::new(
      &[
        4f32, 1f32, 5f32, 3f32, 2f32, 1f32,
      ],
      &device,
    )?;

    let expected: Vec<f32> = expected_tensor.to_vec1()?;

    let actual: Vec<f32> = points.to_vec1()?;

    assert_eq!(actual, expected);

    Ok(())
  }
}
