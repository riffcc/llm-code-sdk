use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HEXTA_THEME_CANDIDATES: &[&str] = &[
    "/home/wings/projects/palace-old/docs/wiki/themes/hextra",
    "/mnt/riffcastle/castle/garage/palace-old/docs/wiki/themes/hextra",
];

pub fn ensure_site_dirs(output: &Path) -> Result<()> {
    for relative in [
        "content",
        "content/docs",
        "layouts/partials/custom",
        "layouts/_partials/scripts",
        "static/css",
        "data",
        "themes",
    ] {
        fs::create_dir_all(output.join(relative))
            .with_context(|| format!("failed to create {}", output.join(relative).display()))?;
    }
    Ok(())
}

pub fn write_text(output: &Path, relative: &str, contents: &str) -> Result<()> {
    let path = output.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(&path, contents).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn write_json(output: &Path, relative: &str, value: &serde_json::Value) -> Result<()> {
    let rendered = serde_json::to_string_pretty(value)?;
    write_text(output, relative, &rendered)
}

pub fn ensure_hextra_theme(output: &Path) -> Result<Option<PathBuf>> {
    let theme_link = output.join("themes/hextra");
    if theme_link.exists() {
        return Ok(Some(theme_link));
    }

    #[cfg(unix)]
    {
        for candidate in HEXTA_THEME_CANDIDATES {
            let candidate_path = Path::new(candidate);
            if candidate_path.exists() {
                std::os::unix::fs::symlink(candidate_path, &theme_link).with_context(|| {
                    format!(
                        "failed to create theme symlink {} -> {}",
                        theme_link.display(),
                        candidate_path.display()
                    )
                })?;
                return Ok(Some(theme_link));
            }
        }
    }

    Ok(None)
}

pub fn build_site(output: &Path) -> Result<()> {
    let status = Command::new("hugo")
        .arg("--source")
        .arg(output)
        .arg("--destination")
        .arg(output.join("public"))
        .status()
        .context("failed to launch hugo")?;

    if !status.success() {
        bail!("hugo build failed for {}", output.display());
    }

    Ok(())
}
