use ::candle_core::{DType, Device, Result, Tensor};
use ::std::fs;
use ::std::path::{Path, PathBuf};
use ::std::str::Chars;

fn main() -> Result<()> {
  // page 97

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let bikes_path: PathBuf = Path::new(cargo_manifest_dir)
    .join("jane-austen")
    .join("1342-0.txt");

  let text: String = fs::read_to_string(bikes_path)?;

  // page 98

  let lines: Vec<&str> = text.lines().collect();

  let line: &str = lines[200];

  println!("line:\n{line}\n");

  let device: Device = Device::Cpu;

  let chars: Chars<'_> = line.chars();

  let char_count: usize = chars.count();

  let letter_t: Tensor = Tensor::zeros((char_count, 128), DType::U8, &device)?;

  println!("letter_t.shape:\n{:?}\n", letter_t.shape());

  let chars: Chars<'_> = line.chars();

  let indices_vec: Vec<u8> = chars
    .map(|letter| {
      if letter < 128 as char {
        letter.to_ascii_lowercase() as u8
      } else {
        0
      }
    })
    .collect();

  let indices = Tensor::new(indices_vec, &device)?.unsqueeze(1)?;

  println!("indices:\n{indices}\n");

  let source = Tensor::ones((char_count, 1), DType::U8, &device)?;

  println!("source:\n{source}\n");

  // https://docs.pytorch.org/docs/main/generated/torch.Tensor.scatter_.html

  let letter_t = letter_t.scatter(&indices, &source, 1)?;

  println!("letter_t:\n{letter_t}\n");

  Ok(())
}
