use anyhow::{Context, Result, anyhow};
use llm_code_sdk::lcs::{
    DaemonConfig, DaemonMode, build_site, generate_ipfs_ha_site, generate_jetpack_site,
    generate_lcs_site, run_daemon,
};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

fn usage() -> &'static str {
    "Usage:
  cargo run -p llm-code-sdk --bin lcs -- site <output-dir>
  cargo run -p llm-code-sdk --bin lcs -- jetpack <source-dir> <output-dir>
  cargo run -p llm-code-sdk --bin lcs -- ipfs-ha <source-dir> <output-dir>
  cargo run -p llm-code-sdk --bin lcs -- build <site-dir>
  cargo run -p llm-code-sdk --bin lcs -- all <jetpack-source> <lcs-output> <jetpack-output>
  cargo run -p llm-code-sdk --bin lcs -- daemon lcs <watch-dir> <output-dir> [--build] [--poll-seconds N]
  cargo run -p llm-code-sdk --bin lcs -- daemon jetpack <source-dir> <output-dir> [--build] [--poll-seconds N]
  cargo run -p llm-code-sdk --bin lcs -- daemon ipfs-ha <source-dir> <output-dir> [--build] [--poll-seconds N]
"
}

fn next_path(args: &mut impl Iterator<Item = String>, label: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("missing {label}\n\n{}", usage()))
}

fn parse_daemon_options(args: &mut impl Iterator<Item = String>) -> Result<(bool, Duration)> {
    let mut build = false;
    let mut poll_seconds = 2.0_f64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--build" => build = true,
            "--poll-seconds" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --poll-seconds\n\n{}", usage()))?;
                poll_seconds = value
                    .parse::<f64>()
                    .with_context(|| format!("invalid --poll-seconds value: {value}"))?;
            }
            other => {
                return Err(anyhow!("unknown daemon option: {other}\n\n{}", usage()));
            }
        }
    }

    if poll_seconds <= 0.0 {
        return Err(anyhow!("--poll-seconds must be > 0"));
    }

    Ok((build, Duration::from_secs_f64(poll_seconds)))
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let command = args
        .next()
        .ok_or_else(|| anyhow!("missing command\n\n{}", usage()))?;

    match command.as_str() {
        "site" => {
            let output = next_path(&mut args, "site output dir")?;
            generate_lcs_site(&output)?;
            println!("generated LCS site at {}", output.display());
        }
        "jetpack" => {
            let source = next_path(&mut args, "jetpack source dir")?;
            let output = next_path(&mut args, "jetpack output dir")?;
            generate_jetpack_site(&source, &output)?;
            println!(
                "generated Jetpack docs site from {} into {}",
                source.display(),
                output.display()
            );
        }
        "ipfs-ha" => {
            let source = next_path(&mut args, "ipfs-ha source dir")?;
            let output = next_path(&mut args, "ipfs-ha output dir")?;
            generate_ipfs_ha_site(&source, &output)?;
            println!(
                "generated IPFS HA docs site from {} into {}",
                source.display(),
                output.display()
            );
        }
        "build" => {
            let output = next_path(&mut args, "site dir")?;
            build_site(&output)?;
            println!("built Hugo site at {}", output.display());
        }
        "all" => {
            let jetpack_source = next_path(&mut args, "jetpack source dir")?;
            let lcs_output = next_path(&mut args, "LCS output dir")?;
            let jetpack_output = next_path(&mut args, "Jetpack output dir")?;

            generate_lcs_site(&lcs_output).context("failed to generate LCS site")?;
            generate_jetpack_site(&jetpack_source, &jetpack_output)
                .context("failed to generate Jetpack site")?;
            build_site(&lcs_output).context("failed to build LCS site")?;
            build_site(&jetpack_output).context("failed to build Jetpack site")?;
            println!(
                "generated and built LCS at {} and Jetpack docs at {}",
                lcs_output.display(),
                jetpack_output.display()
            );
        }
        "daemon" => {
            let mode = args
                .next()
                .ok_or_else(|| anyhow!("missing daemon mode\n\n{}", usage()))?;
            let (mode, output) = match mode.as_str() {
                "lcs" => {
                    let watch = next_path(&mut args, "watch dir")?;
                    let output = next_path(&mut args, "site output dir")?;
                    (DaemonMode::Lcs { watch }, output)
                }
                "jetpack" => {
                    let source = next_path(&mut args, "jetpack source dir")?;
                    let output = next_path(&mut args, "jetpack output dir")?;
                    (DaemonMode::Jetpack { source }, output)
                }
                "ipfs-ha" => {
                    let source = next_path(&mut args, "ipfs-ha source dir")?;
                    let output = next_path(&mut args, "ipfs-ha output dir")?;
                    (DaemonMode::IpfsHa { source }, output)
                }
                _ => {
                    return Err(anyhow!("unknown daemon mode: {mode}\n\n{}", usage()));
                }
            };
            let (build, poll_interval) = parse_daemon_options(&mut args)?;
            run_daemon(DaemonConfig {
                mode,
                output,
                poll_interval,
                build,
            })?;
        }
        _ => {
            return Err(anyhow!("unknown command: {command}\n\n{}", usage()));
        }
    }

    Ok(())
}
