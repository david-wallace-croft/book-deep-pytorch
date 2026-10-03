#![expect(unused)]

use candle_core::{Device, IndexOp, Result, Shape, Tensor};

fn main() -> Result<()> {
  let device: Device = Device::cuda_if_available(0)?;

  println!("\ndevice.is_cuda(): {}\n", device.is_cuda());

  // page 47 section 3.4

  let x = Tensor::new(
    &[[
      1f32, 2., 3.,
    ]],
    &device,
  )?;

  let y = Tensor::new(&[10f32], &device)?;

  let result = x.broadcast_add(&y)?;

  println!("result:\n{result}\n");

  let a = Tensor::new(
    &[[
      1f32, 2., 3.,
    ]],
    &device,
  )?;

  let b = Tensor::new(
    &[
      [10f32],
      [20.],
      [30.],
    ],
    &device,
  )?;

  let result = a.broadcast_mul(&b)?;

  println!("result:\n{result}\n");

  // page 48

  let a = Tensor::new::<Vec<Vec<f32>>>(
    vec![
      vec![
        1., 2., 3.,
      ],
      vec![
        4., 5., 6.,
      ],
    ],
    &device,
  )?;

  println!("a as 2x3:\n{a}\n");

  let a_1x2x3 = a.unsqueeze(0)?;

  println!("a as 1x2x3:\n{a_1x2x3}\n");

  let a_2x2x3 = a_1x2x3.broadcast_as(&[
    2usize, 2, 3,
  ])?;

  println!("a as 2x2x3:\n{a_2x2x3}\n");

  let b = Tensor::new::<&[[[f32; 3]; 1]; 2]>(
    &[
      [[
        10., 20., 30.,
      ]],
      [[
        40., 50., 60.,
      ]],
    ],
    &device,
  )?;

  println!("b as 2x1x3:\n{b}\n");

  let b_2x2x3 = b.broadcast_as(&[
    2usize, 2, 3,
  ])?;

  println!("b as 2x2x3:\n{b_2x2x3}\n");

  let result = a.broadcast_add(&b)?;

  println!("result:\n{result}\n");

  Ok(())
}
