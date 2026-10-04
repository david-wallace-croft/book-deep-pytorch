#![expect(unused)]

use ::candle_core::{
  CpuStorage::F32, DType, Device, IndexOp, Layout, Result, Storage,
  Storage::Cpu, Tensor,
};
use ::std::sync::RwLockReadGuard;

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 58 section 3.9.1

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

  println!("points:\n{points}\n");

  let second_point = points.i(1)?;

  println!("second_point:\n{second_point}\n");

  let (storage, layout) = second_point.storage_and_layout();

  println!("second_point offset: {}", layout.start_offset());

  println!("second_point dimensions: {:?}", second_point.dims());

  println!("second_point shape: {:?}", second_point.shape());

  let (storage, layout) = points.storage_and_layout();

  println!("points stride: {:?}", layout.stride());

  // page 59

  let (storage, layout) = second_point.storage_and_layout();

  println!("\nsecond_point dimensions: {:?}", second_point.dims());

  println!("second_point offset: {}", layout.start_offset());

  println!("second_point stride: {:?}", layout.stride());

  let updated_second_point: Tensor =
    second_point.slice_assign(&[0..1], &Tensor::new(&[10f32], &device)?)?;

  println!("\nupdated_second_point:\n{updated_second_point}\n");

  let unsqueezed_second_point: Tensor = updated_second_point.unsqueeze(0)?;

  println!("unsqueezed_second_point:\n{unsqueezed_second_point}\n");

  let updated_points = points.slice_assign(
    &[
      1..2,
      0..2,
    ],
    &unsqueezed_second_point,
  )?;

  println!("updated_points:\n{updated_points}\n");

  // page 60

  let points_t = points.t()?;

  println!("points_t:\n{points_t}\n");

  let (points_storage, points_layout) = points.storage_and_layout();

  if let Cpu(F32(vec_f32)) = &*points_storage {
    println!("{:p}", vec_f32.as_ptr());
  }

  let (points_t_storage, points_t_layout) = points_t.storage_and_layout();

  if let Cpu(F32(vec_f32)) = &*points_t_storage {
    println!("{:p}", vec_f32.as_ptr());
  }

  println!("\npoints stride: {:?}", points_layout.stride());

  println!("points_t stride: {:?}", points_t_layout.stride());

  // page 61

  let some_t = Tensor::ones((3, 4, 5), DType::F32, &device)?;

  println!("\nsome_t.shape: {:?}", some_t.shape());

  let transpose_t = some_t.transpose(0, 2)?;

  println!("transpose_t.shape: {:?}", transpose_t.shape());

  println!("\nsome_t.stride: {:?}", some_t.stride());

  // page 62

  println!("transpose_t.stride: {:?}", transpose_t.stride());

  println!("\npoints.is_contiguous: {}", points.is_contiguous());

  println!("points_t.is_contiguous: {}", points_t.is_contiguous());

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

  let points_t = points.t()?;

  println!("\npoints_t:\n{points_t}\n");

  let (storage, layout) = points_t.storage_and_layout();

  println!("points_t storage: {:?}", storage);

  println!("points_t stride: {:?}", points_t.stride());

  let points_t_cont = points_t.contiguous()?;

  println!("\npoints_t_cont: \n{points_t_cont}\n");

  println!("points_t_cont stride: {:?}", points_t_cont.stride());

  let (storage, layout) = points_t_cont.storage_and_layout();

  println!("points_t_cont storage: {:?}", storage);

  Ok(())
}
