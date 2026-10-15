// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Kurdish AI Audio Transcription and Translation using Google Gemini AI,
//! with failover API key rotation, SRT/ASS import and export, and word-highlighting.

use std::fmt::Write as _;
use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::transcribe::Segment;
use crate::JobStage;

/// Default API keys with automatic failover rotation from trplugin.
pub const DEFAULT_GEMINI_KEYS: &[&str] = &[
    "AIzaSyDa0ievh4bRuaXlEMGFMRsb8nGz2MHlrR8",
    "AIzaSyAZT5M9uvD30XHtIqGzb6lLGGZefBU4aws",
    "AIzaSyCpL8ZoOMYiN9mxmaBfTiOsbIay5efKGYQ",
];

/// The default Gemini model used for Kurdish transcription & translation.
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";

/// Supported transcription languages for AI transcription.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KurdishLanguage {
    /// Automatic language detection (any spoken language)
    #[default]
    Auto,
    /// Central Kurdish / Sorani dialect (ckb)
    CentralKurdish,
    /// Arabic (including Iraqi Arabic and dialects)
    Arabic,
    /// English
    English,
    /// Persian (Farsi)
    Persian,
    /// Turkish
    Turkish,
    /// German
    German,
    /// French
    French,
    /// Spanish
    Spanish,
    /// Iraqi Arabic dialect (ar-IQ) - preserved for backward compatibility
    IraqiArabic,
}

impl KurdishLanguage {
    /// Returns the prompt instruction for the selected language.
    pub fn prompt_instruction(self) -> &'static str {
        match self {
            KurdishLanguage::Auto => {
                "Transcribe this audio/video accurately. Auto-detect the language being spoken (e.g. Kurdish, Arabic, English, Persian, Turkish, German, French, Spanish, etc.) and transcribe in that detected language using its native alphabet and spelling."
            }
            KurdishLanguage::CentralKurdish => {
                "Transcribe this audio/video to Central Kurdish (Sorani Kurdish, language code: ckb). Use proper Kurdish script and letters: ئ، ە، ڕ، ڵ، ڤ، ۆ، ێ، ی، و."
            }
            KurdishLanguage::Arabic | KurdishLanguage::IraqiArabic => {
                "Transcribe this audio/video to Arabic (including Iraqi, Levantine, Gulf, Egyptian, or Modern Standard Arabic). Use proper Arabic script."
            }
            KurdishLanguage::English => {
                "Transcribe this audio/video in English. Maintain proper punctuation and capitalization."
            }
            KurdishLanguage::Persian => {
                "Transcribe this audio/video in Persian (Farsi). Use proper Persian script with letters: گ، چ، پ، ژ."
            }
            KurdishLanguage::Turkish => {
                "Transcribe this audio/video in Turkish. Use proper Turkish alphabet: ç, ğ, ı, ö, ş, ü."
            }
            KurdishLanguage::German => {
                "Transcribe this audio/video in German. Maintain proper German orthography."
            }
            KurdishLanguage::French => {
                "Transcribe this audio/video in French. Maintain proper accents and punctuation."
            }
            KurdishLanguage::Spanish => {
                "Transcribe this audio/video in Spanish. Maintain proper accents and punctuation."
            }
        }
    }
}

/// Convert 16kHz mono float32 PCM audio samples to 16-bit PCM RIFF WAV in memory.
pub fn pcm_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 1 channel, 16 bits per sample (2 bytes)
    let data_len = num_samples * 2;
    let file_len = 36 + data_len;

    let mut wav = Vec::with_capacity((44 + data_len) as usize);

    // RIFF chunk descriptor
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_len.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // "fmt " sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size (16 for PCM)
    wav.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat 1 = PCM
    wav.extend_from_slice(&1u16.to_le_bytes());  // NumChannels 1 = mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());  // BlockAlign = channels * bits / 8
    wav.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample = 16

    // "data" sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0) as i16;
        wav.extend_from_slice(&sample_i16.to_le_bytes());
    }

    wav
}

/// Transcribes audio using Google Gemini AI, specially tuned for Central Kurdish and Iraqi Arabic.
pub fn transcribe_kurdish(
    samples: &[f32],
    sample_rate: u32,
    lang: KurdishLanguage,
    custom_keys: &[String],
    model: Option<&str>,
    mut progress: impl FnMut(JobStage),
) -> Result<Vec<Segment>, String> {
    progress(JobStage::Reading);
    let wav_bytes = pcm_to_wav_bytes(samples, sample_rate);
    let base64_audio = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    let language_instruction = lang.prompt_instruction();
    let prompt = format!(
        "{language_instruction}\n\n\
CRITICAL ACCURACY REQUIREMENTS - Follow these rules EXACTLY:\n\n\
1. TIMESTAMPS:\n\
   - Use [MM:SS] format (minutes:seconds), NOT [HH:MM:SS]\n\
   - Timestamps MUST be EXACTLY when each word/phrase is spoken\n\
   - Listen carefully and mark the PRECISE start time of each subtitle segment\n\
   - If a word starts at 00:01.5, round to nearest second: [00:02]\n\n\
2. TRANSCRIPTION ACCURACY (MOST IMPORTANT):\n\
   - Transcribe EVERY word exactly as spoken - do NOT skip any words\n\
   - Do NOT paraphrase or summarize - write the EXACT words spoken\n\
   - Do NOT correct grammar or pronunciation - write what you hear\n\
   - Include ALL filler words (اممم، ئەەە، etc.) if clearly spoken\n\
   - If a word is unclear, write your best guess - do NOT skip it\n\
   - Pay attention to dialect-specific vocabulary and pronunciation\n\n\
3. SUBTITLE FORMAT:\n\
   - MAXIMUM 5 WORDS PER SUBTITLE LINE - This is critical!\n\
   - Break long sentences into multiple subtitle segments\n\
   - Each segment must have its own accurate timestamp\n\
   - Ensure natural reading flow - don't break mid-phrase awkwardly\n\n\
4. LANGUAGE-SPECIFIC RULES:\n\
   - For Kurdish (Sorani): Use proper Kurdish script and spelling conventions\n\
   - For Arabic (Iraqi): Use Iraqi dialect vocabulary, not Modern Standard Arabic\n\
   - Preserve the speaker's actual dialect and word choices\n\
   - Include proper Kurdish letters: ئ، ە، ڕ، ڵ، ڤ، ۆ، ێ، ی، و\n\n\
5. QUALITY CHECKS:\n\
   - Double-check each timestamp by mentally replaying that moment\n\
   - Ensure no words are missing from the transcription\n\
   - Verify subtitle segments flow naturally when read in sequence\n\n\
Example correct format:\n\
[00:00] سڵاو چۆنی تۆ\n\
[00:03] من باشم سوپاس بۆ\n\
[00:05] پرسیارەکەت ئەوەندەم باشە\n\n\
Return ONLY the transcription with timestamps. No explanations or extra text."
    );

    let body = serde_json::json!({
        "contents": [{
            "parts": [
                { "text": prompt },
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": base64_audio
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 65536
        }
    });

    progress(JobStage::Loading);
    let chosen_model = model.unwrap_or(DEFAULT_GEMINI_MODEL);

    // Merge custom keys and fallback keys
    let mut keys: Vec<String> = custom_keys.iter().cloned().collect();
    for k in DEFAULT_GEMINI_KEYS {
        if !keys.iter().any(|existing| existing == *k) {
            keys.push((*k).to_string());
        }
    }

    let mut last_error = String::new();
    let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;

    progress(JobStage::Working { done: 0.2, next: 0.8 });

    for (idx, key) in keys.iter().enumerate() {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={key}"
        );

        let response = ureq::post(&url)
            .timeout(Duration::from_secs(90))
            .set("Content-Type", "application/json")
            .send_string(&body_str);

        match response {
            Ok(res) => {
                let status = res.status();
                if status == 200 {
                    let text = res.into_string().map_err(|e| e.to_string())?;
                    let json: serde_json::Value =
                        serde_json::from_str(&text).map_err(|e| e.to_string())?;
                    if let Some(candidate) = json["candidates"]
                        .as_array()
                        .and_then(|c| c.first())
                    {
                        if let Some(content_text) = candidate["content"]["parts"]
                            .as_array()
                            .and_then(|p| p.first())
                            .and_then(|part| part["text"].as_str())
                        {
                            let total_duration = samples.len() as f64 / sample_rate as f64;
                            progress(JobStage::Working { done: 0.9, next: 1.0 });
                            let segments = parse_transcription(content_text, total_duration);
                            return Ok(segments);
                        }
                    }
                    last_error = format!("invalid candidate response from API: {text}");
                } else {
                    last_error = format!("HTTP {status} using key {}", idx + 1);
                }
            }
            Err(e) => {
                last_error = format!("request failed with key {}: {e}", idx + 1);
            }
        }
    }

    Err(format!(
        "All Google AI Gemini API keys ({}) failed. Last error: {last_error}",
        keys.len()
    ))
}

/// Translates existing subtitle segments to any target language.
pub fn translate_captions(
    captions: &[Segment],
    target_lang: &str,
    custom_keys: &[String],
    model: Option<&str>,
) -> Result<Vec<Segment>, String> {
    if captions.is_empty() {
        return Ok(Vec::new());
    }

    let target_name = match target_lang.trim() {
        "ckb" => "Central Kurdish (Sorani Kurdish)",
        "en" => "English",
        "ar" | "ar-IQ" => "Arabic",
        "fa" => "Persian (Farsi)",
        "tr" => "Turkish",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "it" => "Italian",
        "ru" => "Russian",
        "zh" => "Chinese (Simplified)",
        "ja" => "Japanese",
        "nl" => "Dutch",
        "sv" => "Swedish",
        "pt" => "Portuguese",
        other => other,
    };

    let mut keys: Vec<String> = custom_keys.iter().cloned().collect();
    for k in DEFAULT_GEMINI_KEYS {
        if !keys.iter().any(|existing| existing == *k) {
            keys.push((*k).to_string());
        }
    }
    let chosen_model = model.unwrap_or(DEFAULT_GEMINI_MODEL);

    // Chunk into batches of up to 30 lines for 100% reliable 1:1 line translation
    let chunk_size = 30;
    let mut result = Vec::with_capacity(captions.len());

    for chunk in captions.chunks(chunk_size) {
        let chunk_translated = translate_caption_chunk(chunk, target_name, &keys, chosen_model)?;
        result.extend(chunk_translated);
    }

    Ok(result)
}

/// Backwards-compatible alias for `translate_captions`.
pub fn translate_to_kurdish(
    captions: &[Segment],
    target_lang: &str,
    custom_keys: &[String],
    model: Option<&str>,
) -> Result<Vec<Segment>, String> {
    translate_captions(captions, target_lang, custom_keys, model)
}

fn translate_caption_chunk(
    chunk: &[Segment],
    target_name: &str,
    keys: &[String],
    chosen_model: &str,
) -> Result<Vec<Segment>, String> {
    let count = chunk.len();
    let caption_texts: Vec<String> = chunk
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, c.text))
        .collect();
    let joined_input = caption_texts.join("\n");

    let prompt = format!(
        "Translate the following subtitle lines to {target_name}.\n\n\
CRITICAL REQUIREMENTS:\n\
1. EXACT LINE COUNT: You MUST return EXACTLY {count} lines of translated text, corresponding 1-to-1 to each input line.\n\
2. PRESERVE MEANING: Keep natural subtitle flow, length, and accurate meaning.\n\
3. Kurdish script: If translating to Kurdish, use proper Sorani Kurdish script (ئ، ە، ڕ، ڵ، ڤ، ۆ، ێ، ی، و).\n\
4. FORMAT: Output ONLY the translated text, one subtitle per line. Do NOT include numbers, timestamps, markdown, or commentary.\n\n\
INPUT ({count} lines):\n\
{joined_input}\n\n\
OUTPUT (return exactly {count} lines):"
    );

    let body = serde_json::json!({
        "contents": [{
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 65536
        }
    });

    let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;
    let mut last_error = String::new();

    for (idx, key) in keys.iter().enumerate() {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={key}"
        );

        let response = ureq::post(&url)
            .timeout(Duration::from_secs(90))
            .set("Content-Type", "application/json")
            .send_string(&body_str);

        match response {
            Ok(res) if res.status() == 200 => {
                let text = res.into_string().map_err(|e| e.to_string())?;
                let json: serde_json::Value =
                    serde_json::from_str(&text).map_err(|e| e.to_string())?;
                if let Some(content_text) = json["candidates"]
                    .as_array()
                    .and_then(|c| c.first())
                    .and_then(|cand| cand["content"]["parts"].as_array())
                    .and_then(|p| p.first())
                    .and_then(|part| part["text"].as_str())
                {
                    let raw_lines: Vec<&str> = content_text
                        .lines()
                        .map(|l| l.trim())
                        .filter(|l| !l.is_empty() && !l.starts_with("```"))
                        .collect();

                    let mut translated = Vec::with_capacity(chunk.len());
                    for (i, orig) in chunk.iter().enumerate() {
                        let text = if let Some(line) = raw_lines.get(i) {
                            clean_line(line)
                        } else {
                            orig.text.clone()
                        };
                        translated.push(Segment {
                            start: orig.start,
                            end: orig.end,
                            text,
                        });
                    }
                    return Ok(translated);
                }
            }
            Ok(res) => {
                last_error = format!("HTTP {} with key {}", res.status(), idx + 1);
            }
            Err(e) => {
                last_error = format!("Key {} failed: {e}", idx + 1);
            }
        }
    }

    Err(format!(
        "Translation failed across all API keys. Error: {last_error}"
    ))
}

fn clean_line(line: &str) -> String {
    // Strip "1. ", "1 - ", "[00:00]", etc.
    let mut s = line.trim();
    if let Some(pos) = s.find(". ") {
        if s[..pos].chars().all(|c| c.is_ascii_digit()) {
            s = &s[pos + 2..];
        }
    } else if let Some(pos) = s.find("- ") {
        if s[..pos].chars().all(|c| c.is_ascii_digit() || c.is_whitespace()) {
            s = &s[pos + 2..];
        }
    }
    // Also remove wrapping quotes if present
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('“') && s.ends_with('”')) {
        if s.len() >= 2 {
            return s[1..s.len() - 1].trim().to_owned();
        }
    }
    s.to_owned()
}

/// Parses Gemini's raw transcription output into subtitle `Segment`s.
pub fn parse_transcription(raw: &str, total_duration: f64) -> Vec<Segment> {
    let mut items: Vec<(f64, String)> = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("```") {
            continue;
        }

        // Try matching bracketed or parenthesized timestamps: [MM:SS] or (MM:SS) or [HH:MM:SS] or [MM:SS - MM:SS]
        let mut matched = false;
        for (open, close) in [('[', ']'), ('(', ')')] {
            if let Some(start_idx) = trimmed.find(open) {
                if let Some(end_offset) = trimmed[start_idx..].find(close) {
                    let end_idx = start_idx + end_offset;
                    let time_str = &trimmed[start_idx + 1..end_idx];
                    let text = clean_line(&trimmed[end_idx + 1..]);

                    if let Some(seconds) = parse_timestamp(time_str) {
                        if !text.is_empty() {
                            items.push((seconds, text));
                            matched = true;
                            break;
                        }
                    }
                }
            }
        }

        // If no brackets, check if line starts with timestamp like "00:05 text"
        if !matched {
            if let Some(first_word) = trimmed.split_whitespace().next() {
                if let Some(seconds) = parse_timestamp(first_word) {
                    let text = clean_line(&trimmed[first_word.len()..]);
                    if !text.is_empty() {
                        items.push((seconds, text));
                    }
                }
            }
        }
    }

    let mut segments = Vec::with_capacity(items.len());
    for i in 0..items.len() {
        let start = items[i].0;
        let next_start = items.get(i + 1).map(|next| next.0);

        let end = match next_start {
            Some(ns) if ns > start + 0.3 => ns,
            _ => {
                let word_count = items[i].1.split_whitespace().count();
                let duration = (word_count as f64 * 0.38).clamp(1.5, 5.0);
                let est_end = start + duration;
                if total_duration > 0.0 && est_end > total_duration {
                    total_duration.max(start + 0.5)
                } else {
                    est_end
                }
            }
        };

        segments.push(Segment {
            start,
            end,
            text: items[i].1.clone(),
        });
    }

    segments
}

/// Parses a timestamp string (e.g. "01:23", "01:23.456", "01:23,456", or range "01:23 - 01:26") into seconds.
pub fn parse_timestamp(ts: &str) -> Option<f64> {
    // If range like "00:02 - 00:05" or "00:02 --> 00:05", take the start
    let ts = ts.split(|c| c == '-' || c == '~' || c == '—' || c == '>')
        .next()
        .unwrap_or(ts)
        .trim();
    // Normalize commas to dots: "00:02,500" -> "00:02.500"
    let ts = ts.replace(',', ".");
    let parts: Vec<&str> = ts.split(':').collect();
    match parts.len() {
        2 => {
            let mins: f64 = parts[0].trim().parse().ok()?;
            let secs: f64 = parts[1].trim().parse().ok()?;
            Some(mins * 60.0 + secs)
        }
        3 => {
            let hrs: f64 = parts[0].trim().parse().ok()?;
            let mins: f64 = parts[1].trim().parse().ok()?;
            let secs: f64 = parts[2].trim().parse().ok()?;
            Some(hrs * 3600.0 + mins * 60.0 + secs)
        }
        _ => None,
    }
}

/// Export subtitle segments to standard SubRip (.srt) format.
pub fn export_srt(captions: &[Segment]) -> String {
    let mut out = String::new();
    for (i, c) in captions.iter().enumerate() {
        let _ = writeln!(out, "{}", i + 1);
        let _ = writeln!(
            out,
            "{} --> {}",
            format_srt_time(c.start),
            format_srt_time(c.end)
        );
        let _ = writeln!(out, "{}", c.text);
        let _ = writeln!(out);
    }
    out
}

fn format_srt_time(seconds: f64) -> String {
    let secs = seconds.max(0.0);
    let hrs = (secs / 3600.0).floor() as u32;
    let mins = ((secs % 3600.0) / 60.0).floor() as u32;
    let s = (secs % 60.0).floor() as u32;
    let ms = ((secs - secs.floor()) * 1000.0).round() as u32;
    format!("{hrs:02}:{mins:02}:{s:02},{ms:03}")
}

/// Parse standard SubRip (.srt) text into `Vec<Segment>`.
pub fn parse_srt(content: &str) -> Result<Vec<Segment>, String> {
    let mut segments = Vec::new();
    let blocks: Vec<&str> = content.split("\n\n").collect();

    for block in blocks {
        let lines: Vec<&str> = block.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        if lines.len() >= 2 {
            // Find line with "-->"
            if let Some(arrow_line) = lines.iter().find(|l| l.contains("-->")) {
                let parts: Vec<&str> = arrow_line.split("-->").collect();
                if parts.len() == 2 {
                    if let (Some(start), Some(end)) = (parse_srt_stamp(parts[0].trim()), parse_srt_stamp(parts[1].trim())) {
                        let text_lines: Vec<&str> = lines
                            .iter()
                            .skip_while(|l| !l.contains("-->"))
                            .skip(1)
                            .cloned()
                            .collect();
                        let text = text_lines.join(" ");
                        if !text.is_empty() {
                            segments.push(Segment { start, end, text });
                        }
                    }
                }
            }
        }
    }

    Ok(segments)
}

fn parse_srt_stamp(s: &str) -> Option<f64> {
    // 00:01:23,456 or 00:01:23.456
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 3 {
        let h: f64 = parts[0].trim().parse().ok()?;
        let m: f64 = parts[1].trim().parse().ok()?;
        let sec: f64 = parts[2].trim().parse().ok()?;
        Some(h * 3600.0 + m * 60.0 + sec)
    } else {
        None
    }
}

/// ASS subtitle transition animation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubtitleAnimation {
    /// No animation
    #[default]
    None,
    /// Smooth fade in/out
    Fade,
    /// Slide up from bottom
    SlideUp,
    /// Slide down from top
    SlideDown,
    /// Slide in from left
    SlideLeft,
    /// Slide in from right
    SlideRight,
    /// Zoom in from center
    ZoomIn,
    /// Zoom out from center
    ZoomOut,
    /// Dynamic bouncy overshoot
    Bounce,
    /// Stamp drop from top with spring bounce
    DropStamp,
    /// Fast glitch jitter and flash
    GlitchShake,
    /// Gentle pulse scale beat
    Pulse,
    /// Floating cinematic drift
    Drift,
}

/// Highlight shape style for word-by-word highlighting
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HighlightShape {
    /// No background shape
    #[default]
    None,
    /// Rounded rectangle background plate
    Rounded,
    /// Square / box background plate
    Square,
    /// Underline accent bar
    Underline,
    /// High-contrast yellow highlight box
    Highlight,
    /// Translucent frosted liquid glass white shape
    LiquidGlassWhite,
}

/// Advanced SubStation Alpha (.ass) style and highlight options
#[derive(Clone, Debug)]
pub struct AssSubtitleStyle {
    /// Font family name (e.g. Rabar_021)
    pub font_family: String,
    /// Font size in pixels
    pub font_size: u32,
    /// Primary text fill color hex
    pub text_color: String,
    /// Secondary background plate color hex
    pub bg_color: String,
    /// Stroke / outline color hex
    pub stroke_color: String,
    /// Stroke outline thickness
    pub stroke_width: u32,
    /// Selected transition animation
    pub animation: SubtitleAnimation,
    /// Selected highlight shape
    pub highlight_shape: HighlightShape,
    /// Highlight background plate color hex
    pub highlight_bg_color: String,
    /// Highlight text fill color hex
    pub highlight_text_color: String,
    /// Highlight opacity (0.0 - 1.0)
    pub highlight_opacity: f32,
    /// Highlight scale multiplier
    pub highlight_scale: f32,
    /// Letter and word spacing factor
    pub word_spacing: f32,
    /// Video canvas reference width
    pub video_width: u32,
    /// Video canvas reference height
    pub video_height: u32,
    /// Vertical placement percentage from top
    pub y_percent: f64,
}

impl Default for AssSubtitleStyle {
    fn default() -> Self {
        Self {
            font_family: "Rabar_021".to_owned(),
            font_size: 48,
            text_color: "#ffffff".to_owned(),
            bg_color: "#000000".to_owned(),
            stroke_color: "#000000".to_owned(),
            stroke_width: 2,
            animation: SubtitleAnimation::Fade,
            highlight_shape: HighlightShape::None,
            highlight_bg_color: "#ffff00".to_owned(),
            highlight_text_color: "#000000".to_owned(),
            highlight_opacity: 0.85,
            highlight_scale: 1.1,
            word_spacing: 1.0,
            video_width: 1920,
            video_height: 1080,
            y_percent: 0.88,
        }
    }
}

/// Export subtitle segments to Advanced SubStation Alpha (.ass) format with full styling,
/// RTL direction tags, animations, and drawing mode shapes.
pub fn export_ass(captions: &[Segment], style: &AssSubtitleStyle) -> String {
    let mut ass = format!(
        "[Script Info]\n\
; Script generated by Concat Kurdish Subtitle Studio\n\
Title: Kurdish Subtitles\n\
ScriptType: v4.00+\n\
WrapStyle: 0\n\
ScaledBorderAndShadow: yes\n\
YCbCr Matrix: None\n\
PlayResX: {}\n\
PlayResY: {}\n\n\
[V4+ Styles]\n\
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
Style: Default,{},{},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{},0,2,10,10,10,1\n\n\
[Events]\n\
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        style.video_width,
        style.video_height,
        style.font_family,
        style.font_size,
        style.stroke_width
    );

    let x_pos = (style.video_width as f64 * 0.5).round() as i32;
    let y_pos = (style.video_height as f64 * style.y_percent).round() as i32;

    for caption in captions {
        let start_time = format_ass_time(caption.start);
        let end_time = format_ass_time(caption.end);
        let text = caption.text.replace('\n', "\\N");

        let mut tags = format!("{{\\an2\\pos({x_pos},{y_pos})");

        match style.animation {
            SubtitleAnimation::Fade => {
                tags.push_str("\\fad(300,300)");
            }
            SubtitleAnimation::SlideUp => {
                let start_y = (y_pos + 40).min(style.video_height as i32);
                tags = format!("{{\\an2\\move({x_pos},{start_y},{x_pos},{y_pos},0,300)\\fad(200,200)");
            }
            SubtitleAnimation::SlideDown => {
                let start_y = (y_pos - 40).max(0);
                tags = format!("{{\\an2\\move({x_pos},{start_y},{x_pos},{y_pos},0,300)\\fad(200,200)");
            }
            SubtitleAnimation::SlideLeft => {
                let start_x = (x_pos - 120).max(0);
                tags = format!("{{\\an2\\move({start_x},{y_pos},{x_pos},{y_pos},0,250)\\fad(150,150)");
            }
            SubtitleAnimation::SlideRight => {
                let start_x = (x_pos + 120).min(style.video_width as i32);
                tags = format!("{{\\an2\\move({start_x},{y_pos},{x_pos},{y_pos},0,250)\\fad(150,150)");
            }
            SubtitleAnimation::ZoomIn => {
                tags.push_str("\\fscx60\\fscy60\\t(0,300,\\fscx100\\fscy100)\\fad(200,200)");
            }
            SubtitleAnimation::ZoomOut => {
                tags.push_str("\\fscx140\\fscy140\\t(0,250,\\fscx100\\fscy100)\\fad(150,150)");
            }
            SubtitleAnimation::Bounce => {
                tags.push_str("\\fscx40\\fscy40\\t(0,180,\\fscx115\\fscy115)\\t(180,300,\\fscx100\\fscy100)\\fad(150,150)");
            }
            SubtitleAnimation::DropStamp => {
                let start_y = (y_pos - 90).max(0);
                tags = format!("{{\\an2\\move({x_pos},{start_y},{x_pos},{y_pos},0,220)\\t(0,220,\\fscx112\\fscy112)\\t(220,320,\\fscx100\\fscy100)");
            }
            SubtitleAnimation::GlitchShake => {
                tags.push_str("\\t(0,60,\\fscx112)\\t(60,120,\\fscx94)\\t(120,180,\\fscx100)\\fad(100,100)");
            }
            SubtitleAnimation::Pulse => {
                tags.push_str("\\t(0,350,\\fscx108\\fscy108)\\t(350,700,\\fscx100\\fscy100)");
            }
            SubtitleAnimation::Drift => {
                let end_y = (y_pos - 20).max(0);
                tags = format!("{{\\an2\\move({x_pos},{y_pos},{x_pos},{end_y})\\fad(250,250)");
            }
            SubtitleAnimation::None => {}
        }

        match style.highlight_shape {
            HighlightShape::LiquidGlassWhite => {
                tags.push_str("\\1c&H2A170F&\\3c&HFFFFFF&\\4c&H30FFFFFF&\\bord3\\shad1");
            }
            HighlightShape::Highlight => {
                tags.push_str("\\1c&H000000&\\3c&H00E5FF&\\4c&H00E5FF&\\bord4");
            }
            _ => {}
        }
        tags.push('}');

        let _ = writeln!(
            ass,
            "Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{tags}{text}"
        );
    }

    ass
}

fn format_ass_time(seconds: f64) -> String {
    let secs = seconds.max(0.0);
    let hrs = (secs / 3600.0).floor() as u32;
    let mins = ((secs % 3600.0) / 60.0).floor() as u32;
    let s = (secs % 60.0).floor() as u32;
    let cs = ((secs - secs.floor()) * 100.0).round() as u32;
    format!("{hrs}:{mins:02}:{s:02}.{cs:02}")
}
