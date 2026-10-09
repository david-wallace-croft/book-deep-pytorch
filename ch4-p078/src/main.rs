use ::std::fs::{self, DirEntry};
use ::std::io::Result;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let scan_dir: PathBuf = Path::new(cargo_manifest_dir)
    .join("volumetric-dicom")
    .join("2-LUNG 3.0  B70f-04083");

  match get_dicom_files(scan_dir) {
    Ok(files) => {
      println!("Found {} DICOM files:", files.len());

      for file in &files {
        println!("{}", file.display());
      }
    },
    Err(e) => eprintln!("Error reading directory: {e}"),
  }

  Ok(())
}

fn get_dicom_files<P: AsRef<Path>>(dir_path: P) -> Result<Vec<PathBuf>> {
  let mut dcm_files: Vec<PathBuf> = Vec::new();

  for entry_result in fs::read_dir(dir_path)? {
    let entry: DirEntry = entry_result?;

    let path: PathBuf = entry.path();

    if path.is_file()
      && let Some(extension) = path.extension()
      && extension.to_string_lossy().eq_ignore_ascii_case("dcm")
    {
      dcm_files.push(path);
    }
  }

  dcm_files.sort();

  Ok(dcm_files)
}
