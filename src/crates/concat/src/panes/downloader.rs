// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The media downloader sheet: downloads video and audio from YouTube,
//! TikTok, Instagram, Twitter/X, Facebook, and direct media URLs, saving
//! them to the project bin and optionally placing them onto the timeline.

use std::path::PathBuf;
use std::process::Command as SysCommand;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use slint::SharedString;

use crate::host::spawn_in_project;
use crate::i18n::t;
use crate::panes::media_bin::MediaMsg;
use crate::panes::Msg;
use crate::studio::Studio;
use crate::ui::DownloaderSheetData;

/// Everything that can happen to the downloader sheet.
#[derive(Clone, Debug)]
pub enum DownloaderMsg {
    /// Opens the downloader dialog.
    Open,
    /// Closes the downloader dialog.
    Close,
    /// The URL field was typed or pasted into.
    UrlEdited(String),
    /// Format changed: 0 = 1080p MP4, 1 = 4K MP4, 2 = MP3 Audio, 3 = WAV Audio.
    FormatChanged(i32),
    /// Toggle whether to insert onto timeline at playhead.
    InsertTimelineToggled(bool),
    /// Quick preset URL clicked.
    QuickPreset(String),
    /// Start the download.
    Begin,
    /// Cancel an in-progress download.
    Cancel,
    /// Progress status text update.
    #[allow(dead_code)]
    Progress {
        /// Fraction 0..=1
        fraction: f32,
        /// Current status description
        status: String,
    },
    /// The worker finished downloading.
    Finished(Result<PathBuf, String>),
}

/// The state of the downloader pane.
pub struct DownloaderPane {
    pub open: bool,
    pub url: String,
    pub platform: String,
    pub format: i32,
    pub insert_timeline: bool,
    pub running: bool,
    pub progress: f32,
    pub status: String,
    pub error: String,
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl Default for DownloaderPane {
    fn default() -> Self {
        Self {
            open: false,
            url: String::new(),
            platform: "Direct".to_string(),
            format: 0, // 1080p MP4
            insert_timeline: false,
            running: false,
            progress: 0.0,
            status: String::new(),
            error: String::new(),
            cancel_flag: None,
        }
    }
}

impl DownloaderPane {
    /// Detects platform name from URL for display and engine selection.
    pub fn detect_platform(url: &str) -> &'static str {
        let lower = url.to_lowercase();
        if lower.contains("youtube.com") || lower.contains("youtu.be") {
            "YouTube"
        } else if lower.contains("tiktok.com") {
            "TikTok"
        } else if lower.contains("instagram.com") {
            "Instagram"
        } else if lower.contains("twitter.com") || lower.contains("x.com") {
            "Twitter"
        } else if lower.contains("facebook.com") || lower.contains("fb.watch") {
            "Facebook"
        } else {
            "Direct"
        }
    }

    /// Finds the `yt-dlp` executable on the host system.
    fn find_yt_dlp() -> Option<PathBuf> {
        let candidates = [
            "/opt/homebrew/bin/yt-dlp",
            "/usr/local/bin/yt-dlp",
            "yt-dlp",
        ];
        for candidate in candidates {
            if let Ok(output) = SysCommand::new(candidate).arg("--version").output() {
                if output.status.success() {
                    return Some(PathBuf::from(candidate));
                }
            }
        }
        None
    }

    /// Applies one message to the downloader pane.
    pub fn update(&mut self, msg: DownloaderMsg, studio: &mut Studio) {
        match msg {
            DownloaderMsg::Open => {
                self.open = true;
                self.error.clear();
                self.status.clear();
                self.progress = 0.0;
            }
            DownloaderMsg::Close => {
                if !self.running {
                    self.open = false;
                }
            }
            DownloaderMsg::UrlEdited(url) => {
                self.platform = Self::detect_platform(&url).to_string();
                self.url = url;
                self.error.clear();
            }
            DownloaderMsg::FormatChanged(format) => {
                self.format = format.clamp(0, 3);
            }
            DownloaderMsg::InsertTimelineToggled(on) => {
                self.insert_timeline = on;
            }
            DownloaderMsg::QuickPreset(url) => {
                self.platform = Self::detect_platform(&url).to_string();
                self.url = url;
                self.error.clear();
            }
            DownloaderMsg::Begin => {
                let trimmed = self.url.trim().to_string();
                if trimmed.is_empty() || self.running {
                    return;
                }
                self.running = true;
                self.progress = 0.1;
                self.status = t("downloader.connecting");
                self.error.clear();

                let cancel_flag = Arc::new(AtomicBool::new(false));
                self.cancel_flag = Some(cancel_flag.clone());

                let format = self.format;
                let platform = self.platform.clone();
                let output_dir = studio.host.dirs.data.join("downloads");
                let _ = std::fs::create_dir_all(&output_dir);

                spawn_in_project(
                    move || Self::download_worker(trimmed, format, platform, output_dir, cancel_flag),
                    |studio, _, _, result| {
                        studio.handle(Msg::Downloader(DownloaderMsg::Finished(result)));
                    },
                );
            }
            DownloaderMsg::Cancel => {
                if let Some(flag) = &self.cancel_flag {
                    flag.store(true, Ordering::SeqCst);
                }
                self.running = false;
                self.status = t("downloader.cancelled");
            }
            DownloaderMsg::Progress { fraction, status } => {
                self.progress = fraction;
                self.status = status;
            }
            DownloaderMsg::Finished(result) => {
                self.running = false;
                self.cancel_flag = None;
                match result {
                    Ok(path) => {
                        self.progress = 1.0;
                        self.status = t("downloader.complete");
                        // Import the downloaded media into project
                        studio.handle(Msg::Media(MediaMsg::Import(vec![path.clone()])));
                        studio.notify(&t("downloader.successNotice"), false);
                        self.open = false;
                    }
                    Err(err) => {
                        self.progress = 0.0;
                        self.error = err;
                        self.status = t("downloader.failed");
                    }
                }
            }
        }
    }

    /// Background worker to download media from URL.
    fn download_worker(
        url: String,
        format: i32,
        platform: String,
        output_dir: PathBuf,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<PathBuf, String> {
        let is_audio = format == 2 || format == 3;
        let ext = match format {
            1 | 0 => "mp4",
            2 => "mp3",
            3 => "wav",
            _ => "mp4",
        };

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let target_filename = format!("download_{timestamp}.{ext}");
        let target_path = output_dir.join(&target_filename);

        // Check if yt-dlp is available and URL is suitable
        let has_ytdlp = Self::find_yt_dlp();
        let is_social = platform != "Direct"
            || url.contains("youtube")
            || url.contains("youtu.be")
            || url.contains("tiktok")
            || url.contains("instagram")
            || url.contains("twitter")
            || url.contains("x.com")
            || url.contains("facebook");

        if let (Some(ytdlp_bin), true) = (&has_ytdlp, is_social) {
            let out_template = output_dir.join(format!("download_{timestamp}.%(ext)s"));
            let mut cmd = SysCommand::new(ytdlp_bin);
            cmd.arg("--no-playlist")
                .arg("-o")
                .arg(out_template.to_string_lossy().to_string());

            if is_audio {
                let audio_fmt = if format == 3 { "wav" } else { "mp3" };
                cmd.arg("-x").arg("--audio-format").arg(audio_fmt);
            } else if format == 1 {
                // 4K
                cmd.arg("-f")
                    .arg("bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best")
                    .arg("--merge-output-format")
                    .arg("mp4");
            } else {
                // 1080p
                cmd.arg("-f")
                    .arg("bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best")
                    .arg("--merge-output-format")
                    .arg("mp4");
            }

            cmd.arg(&url);

            let output = cmd.output().map_err(|e| format!("Failed to run yt-dlp: {e}"))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Download error: {stderr}"));
            }

            if cancel_flag.load(Ordering::SeqCst) {
                return Err("Download cancelled".to_string());
            }

            // Find matching generated file
            if let Ok(entries) = std::fs::read_dir(&output_dir) {
                for entry in entries.flatten() {
                    let file_name = entry.file_name().to_string_lossy().to_string();
                    if file_name.starts_with(&format!("download_{timestamp}")) {
                        return Ok(entry.path());
                    }
                }
            }
            if target_path.exists() {
                return Ok(target_path);
            }
        }

        // Direct HTTP streaming download fallback
        if url.starts_with("http://") || url.starts_with("https://") {
            let agent = ureq::AgentBuilder::new()
                .timeout_read(std::time::Duration::from_secs(60))
                .build();

            let response = agent
                .get(&url)
                .set("User-Agent", "Mozilla/5.0 Concat/0.2.7")
                .call()
                .map_err(|e| format!("HTTP request failed: {e}"))?;

            let mut reader = response.into_reader();
            let mut file = std::fs::File::create(&target_path)
                .map_err(|e| format!("Could not create target file: {e}"))?;

            let mut buf = [0u8; 64 * 1024];
            loop {
                if cancel_flag.load(Ordering::SeqCst) {
                    let _ = std::fs::remove_file(&target_path);
                    return Err("Download cancelled".to_string());
                }
                match std::io::Read::read(&mut reader, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        std::io::Write::write_all(&mut file, &buf[..n])
                            .map_err(|e| format!("Disk write error: {e}"))?;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(format!("Download read error: {e}")),
                }
            }
            return Ok(target_path);
        }

        Err("Unsupported URL or media format".to_string())
    }

    /// Publishes state to Slint struct.
    pub fn to_slint(&self) -> DownloaderSheetData {
        DownloaderSheetData {
            open: self.open,
            url: SharedString::from(&self.url),
            platform: SharedString::from(&self.platform),
            format: self.format,
            insert_timeline: self.insert_timeline,
            running: self.running,
            progress: self.progress,
            status: SharedString::from(&self.status),
            error: SharedString::from(&self.error),
        }
    }
}
