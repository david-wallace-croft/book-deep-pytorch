#![expect(unused)]

use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};

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

  println!("short_points.dtype: {:?}: ", short_points.dtype());

  Ok(())
}
