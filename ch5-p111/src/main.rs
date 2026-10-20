#![expect(unused)]

use ::anyhow::Result;
use ::candle_core::{DType, Device, Tensor};
use ::plotters::prelude::*;
use ::std::collections::HashMap;
use ::std::fs;
use ::std::path::{Path, PathBuf};
use ::std::str::Chars;

fn main() -> Result<()> {
  // page 111

  let tc: [f32; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let tu: [f32; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let data = [
    (1.0, 2.0),
    (2.3, 3.1),
    (3.0, 2.5),
    (4.5, 5.0),
    (5.2, 4.2),
    (6.1, 7.8),
  ];

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let scatter_plot_path: PathBuf =
    Path::new(cargo_manifest_dir).join("scatter-plot.png");

  let root =
    BitMapBackend::new(&scatter_plot_path, (640, 480)).into_drawing_area();

  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
    .caption(
      "X-Y Measurements Scatter Plot",
      ("sans-serif", 25).into_font(),
    )
    .margin(10)
    .x_label_area_size(30)
    .y_label_area_size(30)
    .build_cartesian_2d(0.0..7.0, 0.0..9.0)?;

  chart.configure_mesh().draw()?;

  chart.draw_series(
    data
      .iter()
      .map(|&(x, y)| Circle::new((x, y), 5, RED.filled())),
  )?;

  root.present()?;

  Ok(())
}
