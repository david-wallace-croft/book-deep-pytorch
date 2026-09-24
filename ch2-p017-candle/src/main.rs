use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("device.is_cuda(): {}", device.is_cuda());

  let a: Tensor = Tensor::new(
    &[
      3f32, 3f32,
    ],
    &device,
  )?;

  println!("{a}");

  let b: Tensor = Tensor::new(
    &[
      3f32, 3f32,
    ],
    &device,
  )?;

  println!("{b}");

  let c: Tensor = (&a + &b)?;

  println!("{c}");

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() -> Result<()> {
    let device: Device = Device::Cpu;

    let a: Tensor = Tensor::new(
      &[
        3f32, 3f32,
      ],
      &device,
    )?;

    let b: Tensor = Tensor::new(
      &[
        3f32, 3f32,
      ],
      &device,
    )?;

    let actual: Tensor = (&a + &b)?;

    assert_eq!(actual.to_vec1::<f32>()?, vec![6.0; 2]);

    Ok(())
  }
}
