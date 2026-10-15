// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Plain text and timing from UTF-8 SubRip and WebVTT files. Styling and
//! WebVTT positioning are intentionally left to the editor's caption style.

/// One subtitle, in seconds from the start of its timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    /// Inclusive start time.
    pub start: f64,
    /// Exclusive end time.
    pub end: f64,
    /// Visible words, with line breaks preserved and cue markup removed.
    pub text: String,
}

/// A malformed subtitle file. No partial result is returned.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("Line {line}: {reason}")]
pub struct ParseError {
    /// One-based source line.
    pub line: usize,
    /// What needs fixing at that line.
    pub reason: &'static str,
}

/// Reads SRT or WebVTT, detecting the format from the WEBVTT header.
/// Cue identifiers, WebVTT settings and metadata blocks are accepted.
/// Invalid timing, empty cues and files without cues are rejected. Times
/// must fit the project's 100-hour limit and minimum 1/60-second clip length
/// so adding a cue cannot silently clamp its timing.
pub fn parse(input: &str) -> Result<Vec<Cue>, ParseError> {
    let normalized = input
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let lines: Vec<&str> = normalized.lines().collect();
    let mut at = 0;
    while at < lines.len() && lines[at].trim().is_empty() {
        at += 1;
    }
    let webvtt = lines.get(at).is_some_and(|line| {
        *line == "WEBVTT" || line.starts_with("WEBVTT ") || line.starts_with("WEBVTT\t")
    });
    if webvtt {
        // Header metadata ends at the first blank line.
        at += 1;
        while at < lines.len() && !lines[at].trim().is_empty() {
            if lines[at].contains("-->") {
                return Err(error(at, "Expected a blank line after the WEBVTT header"));
            }
            at += 1;
        }
    }
    let mut cues = Vec::new();
    while at < lines.len() {
        if lines[at].trim().is_empty() {
            at += 1;
            continue;
        }
        let first = at;
        while at < lines.len() && !lines[at].trim().is_empty() {
            at += 1;
        }
        let block = &lines[first..at];
        let head = block[0].trim();
        if webvtt
            && (head == "NOTE"
                || head.starts_with("NOTE ")
                || head.starts_with("NOTE\t")
                || head == "STYLE"
                || head == "REGION")
        {
            continue;
        }
        let timing = usize::from(!head.contains("-->"));
        if timing == 1 && !webvtt && !head.bytes().all(|b| b.is_ascii_digit()) {
            return Err(error(first, "Expected a subtitle number or timing line"));
        }
        let line = first + timing;
        let (start, rest) = block
            .get(timing)
            .and_then(|s| s.split_once("-->"))
            .ok_or_else(|| error(line, "Expected start --> end timestamps"))?;
        let mut end_and_settings = rest.split_whitespace();
        let end = end_and_settings.next().unwrap_or_default();
        if !webvtt && end_and_settings.next().is_some() {
            return Err(error(line, "Unexpected text after the end timestamp"));
        }
        let start = timestamp(start.trim(), webvtt)
            .ok_or_else(|| error(line, "Invalid start timestamp"))?;
        let end = timestamp(end, webvtt).ok_or_else(|| error(line, "Invalid end timestamp"))?;
        if end <= start {
            return Err(error(line, "End time must be after start time"));
        }
        if end > crate::model::ranges::MAX_TIME {
            return Err(error(line, "Subtitle exceeds the 100-hour timeline limit"));
        }
        if end - start < crate::model::ranges::MIN_CLIP_DURATION {
            return Err(error(line, "Subtitle must last at least 1/60 of a second"));
        }
        let body = &block[timing + 1..];
        if let Some(offset) = body.iter().position(|line| line.contains("-->")) {
            return Err(error(
                line + 1 + offset,
                "Expected a blank line between subtitles",
            ));
        }
        let text = plain_text(&body.join("\n"));
        if text.trim().is_empty() {
            return Err(error(line + 1, "Subtitle text is empty"));
        }
        cues.push(Cue { start, end, text });
    }
    if cues.is_empty() {
        return Err(error(0, "No subtitles found"));
    }
    cues.sort_by(|a, b| a.start.total_cmp(&b.start));
    Ok(cues)
}

fn error(line: usize, reason: &'static str) -> ParseError {
    ParseError {
        line: line + 1,
        reason,
    }
}

fn timestamp(value: &str, webvtt: bool) -> Option<f64> {
    let parts: Vec<&str> = value.split(':').collect();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [hours, minutes, seconds] if hours.len() >= 2 => (*hours, *minutes, *seconds),
        [minutes, seconds] if webvtt => ("00", *minutes, *seconds),
        _ => return None,
    };
    let separator = if webvtt { '.' } else { ',' };
    let (seconds, millis) = seconds.split_once(separator)?;
    if minutes.len() != 2 || seconds.len() != 2 || millis.len() != 3 {
        return None;
    }
    let number = |s: &str| -> Option<u64> {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse().ok())
            .flatten()
    };
    let (hours, minutes, seconds, millis) = (
        number(hours)?,
        number(minutes)?,
        number(seconds)?,
        number(millis)?,
    );
    if minutes >= 60 || seconds >= 60 {
        return None;
    }
    let total = hours
        .checked_mul(3600)?
        .checked_add(minutes * 60 + seconds)?
        .checked_mul(1000)?
        .checked_add(millis)?;
    Some(total as f64 / 1000.0)
}

fn plain_text(input: &str) -> String {
    // Strip only known cue markup, keeping literal comparisons such as
    // "2 < 3". Decode entities afterwards so escaped tags remain literal.
    let mut out = String::new();
    let mut rest = input;
    while let Some(at) = rest.find('<') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        if let Some(end) = rest.find('>') {
            let tag = &rest[1..end];
            let name = tag
                .trim_start_matches('/')
                .split([' ', '\t', '.'])
                .next()
                .unwrap_or_default();
            if matches!(
                name,
                "b" | "i" | "u" | "s" | "font" | "c" | "v" | "lang" | "ruby" | "rt"
            ) || timestamp(tag, true).is_some()
            {
                rest = &rest[end + 1..];
                continue;
            }
        }
        out.push('<');
        rest = &rest[1..];
    }
    out.push_str(rest);
    let mut decoded = String::new();
    let mut rest = out.as_str();
    while let Some(at) = rest.find('&') {
        decoded.push_str(&rest[..at]);
        rest = &rest[at..];
        if let Some(end) = rest.find(';').filter(|end| *end <= 12) {
            let entity = &rest[1..end];
            let value = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                "lrm" => Some('\u{200e}'),
                "rlm" => Some('\u{200f}'),
                _ => entity
                    .strip_prefix("#x")
                    .and_then(|n| u32::from_str_radix(n, 16).ok())
                    .or_else(|| entity.strip_prefix('#').and_then(|n| n.parse().ok()))
                    .and_then(char::from_u32),
            };
            if let Some(value) = value {
                decoded.push(value);
                rest = &rest[end + 1..];
                continue;
            }
        }
        decoded.push('&');
        rest = &rest[1..];
    }
    decoded.push_str(rest);
    decoded.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_preserves_unicode_multiline_and_millisecond_timing() {
        let cues = parse("\u{feff}1\r\n00:00:01,125 --> 00:00:03,456\r\nسڵاو\r\nمرحبا\r\n\r\n2\r\n00:01:00,000 --> 00:01:01,000\r\nHello").unwrap();
        assert_eq!(
            cues[0],
            Cue {
                start: 1.125,
                end: 3.456,
                text: "سڵاو\nمرحبا".into()
            }
        );
        assert_eq!(cues[1].start, 60.0);
    }

    #[test]
    fn webvtt_accepts_metadata_identifiers_settings_and_markup() {
        let cues = parse("WEBVTT - Captions\nLanguage: en\n\nNOTE a comment\nignored\n\nSTYLE\n::cue { color: red; }\n\nREGION\nid:main\n\nintro\n00:01.250 --> 00:03.500 align:start position:10%\n<v Speaker><b>Hello</b> &amp; <c.green>goodbye</c> <00:02.000>world\n").unwrap();
        assert_eq!(
            cues[0],
            Cue {
                start: 1.25,
                end: 3.5,
                text: "Hello & goodbye world".into()
            }
        );
    }

    #[test]
    fn unsorted_and_overlapping_cues_keep_their_times() {
        let cues = parse(
            "2\n00:00:02,000 --> 00:00:04,000\nSecond\n\n1\n00:00:01,000 --> 00:00:03,000\nFirst",
        )
        .unwrap();
        assert_eq!(
            cues.iter().map(|c| (c.start, c.end)).collect::<Vec<_>>(),
            [(1.0, 3.0), (2.0, 4.0)]
        );
    }

    #[test]
    fn invalid_cues_fail_the_entire_file_with_a_line_number() {
        let prefix = "1\n00:00:01,000 --> 00:00:02,000\nValid\n\n2\n";
        for invalid in [
            "00:60:00,000 --> 00:61:00,000\nBad",
            "00:00:03,000 --> 00:00:02,000\nBad",
            "00:00:02,000 --> 00:00:02,000\nBad",
            "-1:00:01,000 --> 00:00:02,000\nBad",
            "00:00:01,00 --> 00:00:02,000\nBad",
            "00:00:01,000 --> 00:00:02,000 garbage\nBad",
            "00:00:01,000 --> 00:00:01,001\nBad",
            "100:00:00,000 --> 100:00:01,000\nBad",
            "999999999999999999:00:00,000 --> 00:00:02,000\nBad",
        ] {
            assert_eq!(parse(&format!("{prefix}{invalid}")).unwrap_err().line, 6);
        }
        assert!(parse(&format!("{prefix}00:00:03,000 --> 00:00:04,000\n<i></i>")).is_err());
    }

    #[test]
    fn rejects_empty_files_and_missing_cue_separators() {
        for input in [
            "",
            "\u{feff}\r\n",
            "WEBVTT\n\nNOTE only a comment",
            "not subtitles",
            "WEBVTT\n00:01.000 --> 00:02.000\nHello",
            "00:00:01,000 --> 00:00:02,000\nOne\n00:00:03,000 --> 00:00:04,000\nTwo",
        ] {
            assert!(parse(input).is_err(), "{input}");
        }
    }

    #[test]
    fn text_cleanup_preserves_literals_and_decodes_entities_once() {
        assert_eq!(
            plain_text("<i>2 < 3</i> &lt;b&gt; &#x627; &#39; &amp;lt; &unknown;"),
            "2 < 3 <b> ا ' &lt; &unknown;"
        );
    }
}
