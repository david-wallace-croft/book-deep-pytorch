use ::tch::{Device, Kind, Tensor};

fn main() {
  let device: Device = Device::cuda_if_available();

  println!("device.is_cuda(): {}", device.is_cuda());

  let a = Tensor::ones(
    &[
      3, 3,
    ],
    (Kind::Float, device),
  );

  println!("{a}");

  let b = Tensor::ones(
    &[
      3, 3,
    ],
    (Kind::Float, device),
  );

  println!("{b}");

  let c = a + b;

  println!("{c}");
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let a = Tensor::ones(
      &[
        3, 3,
      ],
      (Kind::Float, Device::Cpu),
    );

    let b = Tensor::ones(
      &[
        3, 3,
      ],
      (Kind::Float, Device::Cpu),
    );

    let actual = a + b;

    let expected = Tensor::ones(
      &[
        3, 3,
      ],
      (Kind::Float, Device::Cpu),
    ) * 2.0;

    assert_eq!(actual, expected);
  }
}
