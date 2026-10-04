use ::candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 44

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

  // page 45

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

  // page 46

  let p0: Tensor = points.i(0)?;

  println!("points[0]: {p0}\n");

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_p44() -> Result<()> {
    let device: Device = Device::Cpu;

    let points: Tensor = Tensor::zeros(6, DType::F32, &device)?;

    let value: Tensor = Tensor::new(
      &[
        4f32, 1f32, 5f32, 3f32, 2f32, 1f32,
      ],
      &device,
    )?;

    let points: Tensor = points.slice_assign(&[0..6], &value)?;

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

  #[test]
  fn test_p45() -> Result<()> {
    let device: Device = Device::Cpu;

    let points: Tensor = Tensor::new(
      &[
        4f32, 1f32, 5f32, 3f32, 2f32, 1f32,
      ],
      &device,
    )?;

    let p0: f32 = points.i(0)?.to_scalar()?;

    let p1: f32 = points.i(1)?.to_scalar()?;

    assert_eq!(p0, 4f32);

    assert_eq!(p1, 1f32);

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

    let (dim0, dim1) = shape.dims2()?;

    assert_eq!(dim0, 3);

    assert_eq!(dim1, 2);

    let points = Tensor::zeros((3, 2), DType::F32, &device)?;

    let actual: Vec<Vec<f32>> = points.to_vec2()?;

    let expected: Vec<Vec<f32>> = vec![
      vec![
        0f32, 0f32,
      ],
      vec![
        0f32, 0f32,
      ],
      vec![
        0f32, 0f32,
      ],
    ];

    assert_eq!(actual, expected);

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

    let p01: f32 = points.i((0, 1))?.to_scalar()?;

    assert_eq!(p01, 1f32);

    Ok(())
  }
  #[test]
  fn test_p46() -> Result<()> {
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

    let p0: Tensor = points.i(0)?;

    let actual: Vec<f32> = p0.to_vec1()?;

    let expected: Vec<f32> = vec![
      4f32, 1f32,
    ];

    assert_eq!(actual, expected);

    Ok(())
  }
}
