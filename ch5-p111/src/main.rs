use ::anyhow::Result;
use ::plotters::prelude::*;
use ::std::path::{Path, PathBuf};

fn main() -> Result<()> {
  // page 111

  let tc: [f32; 11] = [
    0.5, 14., 15., 28., 11., 8., 3., -4., 6., 13., 21.,
  ];

  let tu: [f32; 11] = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
  ];

  let data: Vec<(f64, f64)> = tc
    .iter()
    .zip(tu.iter())
    .map(|(c, u)| (*c as f64, *u as f64))
    .collect();

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let scatter_plot_path: PathBuf =
    Path::new(cargo_manifest_dir).join("scatter-plot.png");

  let root =
    BitMapBackend::new(&scatter_plot_path, (640, 480)).into_drawing_area();

  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
    .caption(
      "Celsius-Fahrenheit Scatter Plot",
      ("sans-serif", 25).into_font(),
    )
    .margin(10)
    .x_label_area_size(30)
    .y_label_area_size(30)
    .build_cartesian_2d(-5.0..30., 15.0..85.)?;

  chart.configure_mesh().draw()?;

  chart.draw_series(
    data
      .iter()
      .map(|&(x, y)| Circle::new((x, y), 5, RED.filled())),
  )?;

  root.present()?;

  Ok(())
}
