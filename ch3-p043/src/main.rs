use ::candle_core::{DType, Device, IndexOp, Result, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  let a: Tensor = Tensor::ones(3, DType::F32, &device)?;

  println!("a: {a}\n");

  let a1: Tensor = a.i(1)?;

  println!("a[1]: {a1}\n");

  let a1f: f32 = a1.to_scalar::<f32>()?;

  println!("float(a[1]): {a1f}\n");

  let value: Tensor = Tensor::new(&[2f32], &device)?;

  #[expect(clippy::single_range_in_vec_init)]
  let a: Tensor = a.slice_assign(&[2..3], &value)?;

  println!("a: {a}\n");

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() -> Result<()> {
    let device: Device = Device::Cpu;

    let a: Tensor = Tensor::ones(3, DType::F32, &device)?;

    let a1: Tensor = a.i(1)?;

    let a1f: f32 = a1.to_scalar::<f32>()?;

    assert_eq!(a1f, 1f32);

    let value: Tensor = Tensor::new(&[2f32], &device)?;

    let a: Tensor = a.slice_assign(&[2..3], &value)?;

    let expected_tensor: Tensor = Tensor::new(
      &[
        1f32, 1., 2.,
      ],
      &device,
    )?;

    let expected: Vec<f32> = expected_tensor.to_vec1()?;

    let actual: Vec<f32> = a.to_vec1()?;

    assert_eq!(actual, expected);

    Ok(())
  }
}
