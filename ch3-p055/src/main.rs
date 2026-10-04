#![expect(unused)]

use ::candle_core::{
  DType, Device, IndexOp, Layout, Result, Shape, Storage, Tensor,
};
use ::std::sync::RwLockReadGuard;
use candle_core::{CpuStorage::F32, Storage::Cpu};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 55 section 3.8

  let points = Tensor::new(
    &[
      [
        4f32, 1.,
      ],
      [
        5., 3.,
      ],
      [
        2., 1.,
      ],
    ],
    &device,
  )?;

  let (points_storage, points_layout): (RwLockReadGuard<'_, Storage>, &Layout) =
    points.storage_and_layout();

  // page 56

  println!("points_storage: {:?}", points_storage);

  println!("points_layout: {points_layout:?}");

  if let Cpu(F32(vec_f32)) = &*points_storage {
    let p0: f32 = vec_f32[0];

    println!("p0: {p0}");

    let p1: f32 = vec_f32[1];

    println!("p1: {p1}");
  }

  // Probably cannot write directly to Storage

  Ok(())
}
