use ::candle_core::{DType, Device, Result, Tensor};
use ::std::collections::HashMap;
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

  // page 99

  let words_in_line = clean_words(line);

  println!("line:\n{line}\n");

  println!("words_in_line:\n{words_in_line:?}\n");

  let mut word_list: Vec<String> = clean_words(&text);

  word_list.sort();

  word_list.dedup();

  let first_200: &[String] = &word_list[..200];

  println!("first_200:\n{first_200:?}\n");

  let word2index_dict: HashMap<String, usize> = word_list
    .iter()
    .enumerate()
    .map(|(i, word)| (word.clone(), i))
    .collect();

  println!("word2index_dict.len:\n{}\n", word2index_dict.len());

  println!(
    "word2index_dict.get(\"impossible\"):\n{:?}\n",
    word2index_dict.get("impossible")
  );

  // page 100

  let words_in_line_count = words_in_line.len();

  let word_t = Tensor::zeros(
    (words_in_line_count, word2index_dict.len()),
    DType::F32,
    &device,
  )?;

  let indices_vec: Vec<u32> = words_in_line
    .iter()
    .enumerate()
    .map(|(index, word)| {
      let word_index: u32 = *word2index_dict.get(word).unwrap() as u32;

      println!("{index:2} {word_index:4} {word}");

      word_index
    })
    .collect();

  let indices = Tensor::new(indices_vec, &device)?.unsqueeze(1)?;

  println!("indices:\n{indices}\n");

  let source = Tensor::ones((words_in_line_count, 1), DType::F32, &device)?;

  println!("source:\n{source}\n");

  // https://docs.pytorch.org/docs/main/generated/torch.Tensor.scatter_.html

  let word_t = word_t.scatter(&indices, &source, 1)?;

  println!("word_t:\n{word_t}\n");

  Ok(())
}

fn clean_words(input_str: &str) -> Vec<String> {
  let punctuation: [char; 12] = [
    '\'', '.', ',', ';', ':', '"', '!', '?', '”', '“', '_', '-',
  ];

  input_str
    .split_whitespace()
    .map(|word: &str| word.to_lowercase())
    .map(|word: String| {
      word.chars().filter(|c| !punctuation.contains(c)).collect()
    })
    .collect()
}
