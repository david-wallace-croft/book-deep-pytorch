use candle_core::{DType, Device, Result, Tensor};

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

  let points: Tensor = points.slice_assign(&[0..6], &value)?;

  println!("points: {points}\n");

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
