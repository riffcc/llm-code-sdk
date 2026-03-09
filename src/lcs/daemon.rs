use crate::lcs::{build_site, generate_ipfs_ha_site, generate_jetpack_site, generate_lcs_site};
use anyhow::{Context, Result, anyhow};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub enum DaemonMode {
    Lcs { watch: PathBuf },
    Jetpack { source: PathBuf },
    IpfsHa { source: PathBuf },
}

#[derive(Clone, Debug)]
pub struct DaemonConfig {
    pub mode: DaemonMode,
    pub output: PathBuf,
    pub poll_interval: Duration,
    pub build: bool,
}

impl DaemonConfig {
    pub fn watch_roots(&self) -> Vec<PathBuf> {
        match &self.mode {
            DaemonMode::Lcs { watch } => vec![watch.clone()],
            DaemonMode::Jetpack { source } => vec![source.clone()],
            DaemonMode::IpfsHa { source } => vec![source.clone()],
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            DaemonMode::Lcs { .. } => "lcs",
            DaemonMode::Jetpack { .. } => "jetpack",
            DaemonMode::IpfsHa { .. } => "ipfs-ha",
        }
    }
}

fn should_skip(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".git" | "target" | "node_modules" | "public" | "dist" | ".direnv")
        )
    })
}

fn walk_files(root: &Path, current: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if should_skip(current) {
        return Ok(());
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(current)
        .with_context(|| format!("failed to read {}", current.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .collect();
    entries.sort();

    for path in entries {
        if should_skip(&path) {
            continue;
        }
        if path.is_dir() {
            walk_files(root, &path, out)?;
        } else if path.is_file() {
            out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
        }
    }

    Ok(())
}

fn fingerprint_root(root: &Path) -> Result<u64> {
    let mut files = Vec::new();
    walk_files(root, root, &mut files)?;
    files.sort();

    let mut hasher = DefaultHasher::new();
    root.display().to_string().hash(&mut hasher);

    for relative in files {
        let full = root.join(&relative);
        let metadata = fs::metadata(&full)
            .with_context(|| format!("failed to read metadata for {}", full.display()))?;
        relative.display().to_string().hash(&mut hasher);
        metadata.len().hash(&mut hasher);

        let modified = metadata
            .modified()
            .with_context(|| format!("failed to read mtime for {}", full.display()))?
            .duration_since(UNIX_EPOCH)
            .map_err(|err| anyhow!("failed to fingerprint {}: {err}", full.display()))?;
        modified.as_secs().hash(&mut hasher);
        modified.subsec_nanos().hash(&mut hasher);
    }

    Ok(hasher.finish())
}

fn fingerprint_roots(roots: &[PathBuf]) -> Result<u64> {
    let mut hasher = DefaultHasher::new();
    for root in roots {
        root.display().to_string().hash(&mut hasher);
        fingerprint_root(root)?.hash(&mut hasher);
    }
    Ok(hasher.finish())
}

fn regenerate(config: &DaemonConfig) -> Result<()> {
    match &config.mode {
        DaemonMode::Lcs { .. } => generate_lcs_site(&config.output)
            .with_context(|| format!("failed to regenerate {}", config.output.display()))?,
        DaemonMode::Jetpack { source } => generate_jetpack_site(source, &config.output)
            .with_context(|| format!("failed to regenerate {}", config.output.display()))?,
        DaemonMode::IpfsHa { source } => generate_ipfs_ha_site(source, &config.output)
            .with_context(|| format!("failed to regenerate {}", config.output.display()))?,
    }

    if config.build {
        build_site(&config.output)
            .with_context(|| format!("failed to build {}", config.output.display()))?;
    }

    Ok(())
}

pub fn run_daemon(config: DaemonConfig) -> Result<()> {
    let watch_roots = config.watch_roots();
    println!(
        "starting lcs daemon: mode={} output={} poll={}s build={}",
        config.mode_name(),
        config.output.display(),
        config.poll_interval.as_secs_f32(),
        config.build
    );
    for root in &watch_roots {
        println!("watching {}", root.display());
    }

    let mut previous = None;
    loop {
        let current = fingerprint_roots(&watch_roots)?;
        if previous != Some(current) {
            regenerate(&config)?;
            println!(
                "regenerated {} at {}",
                config.mode_name(),
                config.output.display()
            );
            previous = Some(current);
        }
        thread::sleep(config.poll_interval);
    }
}
