// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The captions sheet: the tray's Captions tool, as a form and then as a
//! progress report.
//!
//! One switch says what it captions: "Use voice from timeline", on, takes
//! the words from a clip with sound, chosen in the sheet from the clips on
//! the timeline that have any; off, from a script typed in it. It opens on
//! when a clip with sound is selected, or when a clip's menu opened it on
//! that clip, and off otherwise. Either lands as one batch of title clips,
//! one undo step.
//! The transcriber runs on a worker and reports as
//! [`CaptionsMsg::Progress`], ending as [`CaptionsMsg::Finished`].

use std::sync::Arc;

use concat_project::Command;
use concat_project::model::{ClipKey, KeyEase, KeyProperty, TextStyle};
use concat_speech::JobStage;
use concat_speech::transcribe::Segment;

use crate::host::{on_ui_in_project, spawn_in_project};
use crate::i18n::{t, tf};
use crate::panes::Msg;
use crate::panes::pace::{Pace, Stage, Timings};
use crate::panes::settings::installed;
use crate::studio::Studio;
use crate::ui::CaptionsSheetData;

/// Where a caption sits, by the sheet's row: a frame-height fraction from
/// the centre, positive down. Bottom, centre, top.
const CAPTION_OFFSETS: [f64; 3] = [0.35, 0.0, -0.35];
/// A caption's cap height by the sheet's row, as a fraction of the frame.
const CAPTION_SIZES: [f64; 3] = [0.04, 0.05, 0.065];
/// A rough speaking rate, for a script's timing and the speech sheet's
/// estimate.
pub const CHARS_PER_SECOND: f32 = 14.0;

/// Spoken language choices for audio transcription.
pub const SPOKEN_LANGUAGES: &[(&str, &str)] = &[
    ("auto", "Auto Detect (خۆکار / تلقائي)"),
    ("ar", "Arabic / عەرەبی (العربية الفصحى)"),
    ("ar-IQ", "Arabic (Iraqi) / عەرەبی عێراقی"),
    ("ckb", "Kurdish / کوردی (Sorani)"),
    ("en", "English / ئینگلیزی"),
    ("fa", "Persian / فارسی"),
    ("tr", "Turkish / تورکی"),
    ("de", "German / ئەڵمانی"),
    ("fr", "French / فەرەنسی"),
    ("es", "Spanish / ئیسپانی"),
];

/// Target language choices for subtitle translation.
pub const TARGET_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic / عەرەبی (العربية الفصحى)"),
    ("ar-IQ", "Arabic (Iraqi) / عەرەبی عێراقی"),
    ("ckb", "Kurdish / کوردی (Sorani)"),
    ("en", "English / ئینگلیزی"),
    ("fa", "Persian / فارسی"),
    ("tr", "Turkish / تورکی"),
    ("de", "German / ئەڵمانی"),
    ("fr", "French / فەرەنسی"),
    ("es", "Spanish / ئیسپانی"),
    ("it", "Italian / ئیتاڵی"),
    ("ru", "Russian / ڕووسی"),
    ("zh", "Chinese / چینی"),
    ("ja", "Japanese / ژاپۆنی"),
    ("nl", "Dutch / هۆڵەندی"),
    ("sv", "Swedish / سویدی"),
    ("pt", "Portuguese / پورتوگالی"),
];

/// Everything that can happen to the captions sheet.
#[derive(Clone, Debug)]
pub enum CaptionsMsg {
    /// The tray's Captions tool: the selected clip's voice when it has
    /// sound, the script otherwise.
    Open,
    /// A clip's menu: the sheet on that clip's voice.
    OpenOnClip(String),
    Close,
    /// The "Use voice from timeline" switch.
    VoiceToggled(bool),
    /// Row in the list of clips with sound.
    SourceChanged(i32),
    TextEdited(String),
    ModelChanged(i32),
    PlacementChanged(i32),
    SizeChanged(i32),
    /// Kurdish AI toggle (Gemini vs Whisper)
    KurdishAiToggled(bool),
    /// Language: 0 Kurdish (Sorani), 1 Arabic (Iraqi), 2 Auto
    KurdishLanguageChanged(i32),
    /// Translate captions toggle
    TranslateToggled(bool),
    /// Target language index for translation
    TargetLanguageChanged(i32),
    /// Animation: 0 None, 1 Fade, 2 Slide Up, 3 Slide Down, 4 Zoom In, 5 Bounce
    AnimationChanged(i32),
    /// Highlight style: 0 Off, 1 Rounded, 2 Square, 3 Underline, 4 Highlight
    HighlightChanged(i32),
    /// Import a UTF-8 SRT or WebVTT file at its original timeline times.
    ImportSubtitles,
    /// Export timeline captions to SRT
    ExportSrt,
    /// Export timeline captions to ASS
    ExportAss,
    /// Translate existing captions on timeline to target language
    TranslateTimelineCaptions,
    /// Run the pass the sheet describes.
    Begin,
    Cancel,
    /// The transcriber's worker reporting the stage it is in.
    Progress(JobStage),
    /// The transcriber's worker is done: what was said, or why not.
    Finished(Result<Vec<Segment>, String>),
}

/// The clip a transcription is running over, as the finished segments
/// need it: where it starts on the timeline and how fast it plays.
#[derive(Clone, Debug)]
struct Subject {
    start: f64,
    speed: f64,
    /// The look chosen when the run began.
    look: (f64, f64),
    /// The transcriber model, and the thirty-second windows whisper reads
    /// the sound in: what the run's timing is kept by.
    model: String,
    windows: f32,
    animation: usize,
    highlight_style: usize,
    is_kurdish: bool,
}

/// Seconds whisper takes over one thirty-second window, per megabyte of
/// model, before a run on the model has been timed: about a second for
/// base and ten for medium on eight cores of a recent CPU.
const WINDOW_SECONDS_PER_MB: f32 = 1.0 / 160.0;

/// The captions sheet's state.
pub struct CaptionsPane {
    pub open: bool,
    /// "Use voice from timeline": the words come from [`Self::clip`]'s sound
    /// when on, from [`Self::text`] when off.
    pub from_voice: bool,
    /// The clip whose voice is transcribed: one with sound, chosen from the
    /// timeline's. Kept while the switch is off, so turning it back on
    /// returns to the same clip.
    pub clip: Option<String>,
    /// The script's words.
    pub text: String,
    /// Row in the installed transcriber list.
    pub model: usize,
    /// 0 bottom, 1 centre, 2 top.
    pub placement: usize,
    /// 0 small, 1 medium, 2 large.
    pub size: usize,
    /// Kurdish AI (Gemini) enabled
    pub is_kurdish_ai: bool,
    /// 0 Central Kurdish, 1 Iraqi Arabic, 2 Auto
    pub kurdish_language: usize,
    /// Translate captions toggle
    pub translate_to_kurdish: bool,
    /// Target language index for translation (from TARGET_LANGUAGES)
    pub target_language: usize,
    /// 0 None, 1 Fade, 2 Slide Up, 3 Slide Down, 4 Zoom In, 5 Bounce
    pub animation: usize,
    /// 0 Off, 1 Rounded, 2 Square, 3 Underline, 4 Highlight
    pub highlight_style: usize,
    pub running: bool,
    /// How far the running transcription looks.
    pub pace: Pace,
    /// Why the last run failed, when it did.
    pub message: String,
    /// The clip under the running transcription.
    subject: Option<Subject>,
    /// How long each model took this session, for the next run's bar.
    timings: Timings,
    /// Cached sound sources from timeline for fast UI rendering.
    pub sources_cache: Vec<Source>,
}

impl Default for CaptionsPane {
    fn default() -> Self {
        Self {
            open: false,
            from_voice: false,
            clip: None,
            text: String::new(),
            model: 0,
            placement: 0,
            size: 1,
            is_kurdish_ai: true,
            kurdish_language: 0,
            translate_to_kurdish: false,
            target_language: 0,
            animation: 0,
            highlight_style: 0,
            running: false,
            pace: Pace::default(),
            message: String::new(),
            subject: None,
            timings: Timings::default(),
            sources_cache: Vec::new(),
        }
    }
}

impl CaptionsPane {
    /// Applies one message. The studio is the rest of the window; while
    /// this runs the studio's copy of the pane is a blank it must not read.
    pub fn update(&mut self, msg: CaptionsMsg, studio: &mut Studio) {
        match msg {
            CaptionsMsg::Open => {
                let selected = studio.sole_selection();
                self.open_on(studio, selected, false);
            }
            CaptionsMsg::OpenOnClip(id) => self.open_on(studio, Some(id), true),
            CaptionsMsg::Close => self.open = false,
            CaptionsMsg::VoiceToggled(on) => {
                self.from_voice = on;
                self.message.clear();
                let sources = sources(studio);
                if on
                    && !self
                        .clip
                        .as_ref()
                        .is_some_and(|id| sources.iter().any(|s| &s.id == id))
                {
                    self.clip = sources.first().map(|source| source.id.clone());
                }
            }
            CaptionsMsg::SourceChanged(index) => {
                if let Some(source) = usize::try_from(index)
                    .ok()
                    .and_then(|i| self.sources_cache.get(i))
                {
                    self.clip = Some(source.id.clone());
                    self.message.clear();
                }
            }
            CaptionsMsg::TextEdited(text) => self.text = text,
            CaptionsMsg::ModelChanged(index) => self.model = index.max(0) as usize,
            CaptionsMsg::PlacementChanged(index) => {
                self.placement = (index.max(0) as usize).min(2);
            }
            CaptionsMsg::SizeChanged(index) => self.size = (index.max(0) as usize).min(2),
            CaptionsMsg::KurdishAiToggled(on) => self.is_kurdish_ai = on,
            CaptionsMsg::KurdishLanguageChanged(index) => {
                self.kurdish_language =
                    (index.max(0) as usize).min(SPOKEN_LANGUAGES.len().saturating_sub(1));
            }
            CaptionsMsg::TranslateToggled(on) => self.translate_to_kurdish = on,
            CaptionsMsg::TargetLanguageChanged(index) => {
                self.target_language =
                    (index.max(0) as usize).min(TARGET_LANGUAGES.len().saturating_sub(1));
            }
            CaptionsMsg::AnimationChanged(index) => {
                self.animation = (index.max(0) as usize).min(12);
            }
            CaptionsMsg::HighlightChanged(index) => {
                self.highlight_style = (index.max(0) as usize).min(6);
            }
            CaptionsMsg::ImportSubtitles => self.import_subtitles(studio),
            CaptionsMsg::ExportSrt => self.export_srt_captions(studio),
            CaptionsMsg::ExportAss => self.export_ass_captions(studio),
            CaptionsMsg::TranslateTimelineCaptions => self.translate_timeline_captions(studio),
            CaptionsMsg::Begin => {
                if self.from_voice {
                    if self.is_kurdish_ai {
                        self.run_kurdish_sound(studio);
                    } else {
                        self.run_sound(studio);
                    }
                } else {
                    self.run_script(studio);
                }
            }
            CaptionsMsg::Cancel => {
                studio.host.transcriber.cancel();
                self.running = false;
                self.open = false;
            }
            CaptionsMsg::Progress(stage) => self.pace.report(stage),
            CaptionsMsg::Finished(result) => {
                self.running = false;
                let subject = self.subject.take();
                if let (Ok(_), Some(subject)) = (&result, &subject) {
                    self.timings
                        .keep(&subject.model, self.pace.learned(subject.windows));
                }
                match result {
                    Ok(segments) => {
                        let Some(subject) = subject else {
                            return;
                        };
                        let look = subject.look;
                        let base = studio.prefs.title_style.clone();
                        let commands: Vec<Command> = segments
                            .into_iter()
                            .filter_map(|segment| {
                                let text = segment.text.trim().to_owned();
                                (!text.is_empty()).then(|| {
                                    caption_clip(
                                        text,
                                        subject.start + segment.start / subject.speed,
                                        ((segment.end - segment.start) / subject.speed).max(0.2),
                                        look,
                                        base.as_ref(),
                                        subject.animation,
                                        subject.highlight_style,
                                        subject.is_kurdish,
                                    )
                                })
                            })
                            .collect();
                        let count = commands.len();
                        self.open = false;
                        if count == 0 {
                            studio.notify(&t("captions.nothingSaidClip"), true);
                        } else {
                            studio.apply(Command::Batch { commands });
                            studio.notify(&tf("captions.addedCaptions", &[&count]), false);
                        }
                    }
                    // Asked for: the sheet is already on its way down.
                    Err(error) if error.contains("cancel") => self.open = false,
                    Err(error) => self.message = error,
                }
            }
        }
    }

    /// The sheet opened afresh. `clip` is the clip it was opened on, if
    /// any: the switch starts on when that clip has sound, or when a clip's
    /// menu asked for its voice (`voice`), in which case a clip without
    /// sound falls to the first clip that has some.
    fn open_on(&mut self, studio: &Studio, clip: Option<String>, voice: bool) {
        let installed = installed(&studio.settings.transcribers);
        let model = installed.iter().position(|model| model.active).unwrap_or(0);
        let sources = sources(studio);
        let sounding = clip.filter(|id| sources.iter().any(|source| &source.id == id));
        let from_voice = voice || sounding.is_some() || !sources.is_empty();
        let clip = sounding.or_else(|| {
            from_voice
                .then(|| sources.first().map(|source| source.id.clone()))
                .flatten()
        });
        *self = CaptionsPane {
            open: true,
            from_voice,
            clip,
            model,
            placement: 0,
            size: 1,
            is_kurdish_ai: self.is_kurdish_ai,
            kurdish_language: self.kurdish_language,
            translate_to_kurdish: self.translate_to_kurdish,
            target_language: self.target_language,
            animation: self.animation,
            highlight_style: self.highlight_style,
            timings: std::mem::take(&mut self.timings),
            sources_cache: sources,
            ..CaptionsPane::default()
        };
    }

    /// A caption's look, by the sheet's rows: where it sits and its size.
    fn look(&self) -> (f64, f64) {
        (
            CAPTION_OFFSETS[self.placement.min(2)],
            CAPTION_SIZES[self.size.min(2)],
        )
    }

    /// The script as titles, one after another from the playhead.
    fn run_script(&mut self, studio: &mut Studio) {
        let lines = script_captions(&self.text);
        if lines.is_empty() {
            self.message = t("captions.nothingToCaptionYet");
            return;
        }
        let look = self.look();
        let mut at = f64::from(studio.playhead);
        let base = studio.prefs.title_style.clone();
        let commands: Vec<Command> = lines
            .into_iter()
            .map(|(text, seconds)| {
                let command = caption_clip(
                    text,
                    at,
                    seconds,
                    look,
                    base.as_ref(),
                    self.animation,
                    self.highlight_style,
                    self.is_kurdish_ai,
                );
                at += seconds;
                command
            })
            .collect();
        let count = commands.len();
        self.open = false;
        studio.apply(Command::Batch { commands });
        studio.notify(&tf("captions.addedCaptions", &[&count]), false);
    }

    /// Run Kurdish transcription and translation using Google Gemini AI.
    fn run_kurdish_sound(&mut self, studio: &mut Studio) {
        let Some(clip) = self.clip.as_ref().and_then(|id| studio.clip(id)).cloned() else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        let Some(media) = studio.project().media_by_id(&clip.media_id).cloned() else {
            self.message = t("captions.clipHasNoFile");
            return;
        };

        let lang = match self.kurdish_language {
            1 => concat_speech::gemini::KurdishLanguage::Arabic,
            2 => concat_speech::gemini::KurdishLanguage::IraqiArabic,
            3 => concat_speech::gemini::KurdishLanguage::CentralKurdish,
            4 => concat_speech::gemini::KurdishLanguage::English,
            5 => concat_speech::gemini::KurdishLanguage::Persian,
            6 => concat_speech::gemini::KurdishLanguage::Turkish,
            7 => concat_speech::gemini::KurdishLanguage::German,
            8 => concat_speech::gemini::KurdishLanguage::French,
            9 => concat_speech::gemini::KurdishLanguage::Spanish,
            _ => concat_speech::gemini::KurdishLanguage::Auto,
        };
        let translate = self.translate_to_kurdish;
        let target_lang = TARGET_LANGUAGES
            .get(self.target_language)
            .map(|(code, _)| *code)
            .unwrap_or("ar")
            .to_owned();
        let is_target_arabic_or_kurdish =
            target_lang == "ar" || target_lang == "ar-IQ" || target_lang == "ckb";
        let custom_keys = studio.prefs.gemini_api_keys.clone();
        let model = studio.prefs.gemini_model.clone();

        let source_start = clip.source_start;
        let window = clip.duration * clip.speed;
        let audio_stream = clip.audio_stream;
        let media_path = media.path.clone();

        self.pace = Pace::new(vec![
            (Stage::Reading, 0.5),
            (Stage::Loading, 1.0),
            (Stage::Working, (window as f32 / 10.0).max(2.0)),
        ]);

        self.subject = Some(Subject {
            start: clip.start,
            speed: clip.speed,
            look: self.look(),
            model: "gemini-2.5-flash".to_owned(),
            windows: 1.0,
            animation: self.animation,
            highlight_style: self.highlight_style,
            is_kurdish: if translate {
                is_target_arabic_or_kurdish
            } else {
                self.kurdish_language <= 3
            },
        });

        self.running = true;
        self.message.clear();
        let epoch = crate::host::project_epoch();

        spawn_in_project(
            move || {
                let mut decoder = concat_media::AudioDecoder::open(
                    &media_path,
                    &concat_media::AudioOptions {
                        start: Some(source_start),
                        duration: Some(window),
                        filters: Vec::new(),
                        rate: 16_000,
                        channels: 1,
                        format: concat_media::SampleFormat::F32,
                        stream: audio_stream.map(|index| index as usize),
                    },
                )
                .map_err(|error| error.to_string())?;

                let samples = decoder.collect_f32().map_err(|error| error.to_string())?;
                if samples.is_empty() {
                    return Err("No audio found in clip".to_owned());
                }

                let mut segments = concat_speech::gemini::transcribe_kurdish(
                    &samples,
                    16_000,
                    lang,
                    &custom_keys,
                    model.as_deref(),
                    move |stage| {
                        on_ui_in_project(epoch, move |studio, _, _| {
                            studio.handle(Msg::Captions(CaptionsMsg::Progress(stage)));
                        });
                    },
                )?;

                if translate {
                    segments = concat_speech::gemini::translate_captions(
                        &segments,
                        &target_lang,
                        &custom_keys,
                        model.as_deref(),
                    )?;
                }

                Ok(segments)
            },
            |studio, _, _, result| studio.handle(Msg::Captions(CaptionsMsg::Finished(result))),
        );
    }

    /// The sheet's clip through the transcriber on a worker, reporting into
    /// the sheet as it goes.
    fn run_sound(&mut self, studio: &mut Studio) {
        let Some(clip) = self.clip.as_ref().and_then(|id| studio.clip(id)).cloned() else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        let Some(media) = studio.project().media_by_id(&clip.media_id).cloned() else {
            self.message = t("captions.clipHasNoFile");
            return;
        };
        let Some((model, megabytes)) = installed(&studio.settings.transcribers)
            .get(self.model)
            .map(|model| (model.id.clone(), model.megabytes))
        else {
            self.message = t("captions.downloadTranscriberModelFirst");
            return;
        };
        let request = concat_speech::transcribe::TranscribeRequest {
            path: media.path.clone(),
            audio_stream: clip.audio_stream,
            source_start: clip.source_start,
            window: clip.duration * clip.speed,
            model_id: model.clone(),
        };
        let dirs = studio.host.dirs.clone();
        let transcriber = Arc::clone(&studio.host.transcriber);
        let windows = (request.window / 30.0).ceil().max(1.0) as f32;
        let learned = self.timings.get(&model);
        let loaded = studio.host.transcriber.loaded_model().as_deref() == Some(model.as_str());
        let load = if loaded {
            0.0
        } else {
            learned.load.unwrap_or(0.5 + megabytes / 400.0)
        };
        let per_window = learned
            .per_unit
            .unwrap_or(0.3 + megabytes * WINDOW_SECONDS_PER_MB);
        self.pace = Pace::new(vec![
            (Stage::Reading, 0.2 + request.window as f32 / 400.0),
            (Stage::Loading, load),
            (Stage::Working, windows * per_window),
        ]);
        self.subject = Some(Subject {
            start: clip.start,
            speed: clip.speed,
            look: self.look(),
            model,
            windows,
            animation: self.animation,
            highlight_style: self.highlight_style,
            is_kurdish: false,
        });
        self.running = true;
        self.message.clear();
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                transcriber.transcribe(&dirs, &request, move |stage| {
                    on_ui_in_project(epoch, move |studio, _, _| {
                        studio.handle(Msg::Captions(CaptionsMsg::Progress(stage)));
                    });
                })
            },
            |studio, _, _, result| studio.handle(Msg::Captions(CaptionsMsg::Finished(result))),
        );
    }

    /// Pick and parse off the UI thread, then add the entire file as one edit.
    fn import_subtitles(&mut self, studio: &Studio) {
        if self.running || studio.session.is_none() {
            return;
        }
        self.message.clear();
        let epoch = crate::host::project_epoch();
        let timeline = studio.timeline().id.clone();
        let look = self.look();
        let base = studio.prefs.title_style.clone();
        let animation = self.animation;
        let highlight = self.highlight_style;
        crate::host::pick_files_then(
            t("captions.importSubtitles"),
            Some((t("captions.subtitleFiles"), &["srt", "vtt"])),
            move |studio, _, _, paths| {
                if crate::host::project_epoch() != epoch {
                    return;
                }
                if paths.len() != 1 {
                    studio.captions.message = t("captions.chooseOneSubtitle");
                    return;
                }
                let path = paths[0].clone();
                spawn_in_project(
                    move || {
                        let content =
                            std::fs::read_to_string(path).map_err(|err| err.to_string())?;
                        let cues = concat_project::subtitles::parse(&content)
                            .map_err(|err| err.to_string())?;
                        Ok::<_, String>(
                            cues.into_iter()
                                .map(|cue| {
                                    caption_clip(
                                        cue.text,
                                        cue.start,
                                        cue.end - cue.start,
                                        look,
                                        base.as_ref(),
                                        animation,
                                        highlight,
                                        false,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    },
                    move |studio, _, _, result| {
                        if studio.timeline().id != timeline {
                            studio.notify(&t("captions.importTimelineChanged"), true);
                            return;
                        }
                        match result {
                            Ok(commands) => {
                                let count = commands.len();
                                if studio.apply(Command::Batch { commands }).is_some() {
                                    studio.captions.open = false;
                                    studio.notify(&tf("captions.addedCaptions", &[&count]), false);
                                }
                            }
                            Err(error) => {
                                let message = tf("captions.importFailed", &[&error]);
                                studio.captions.message = message.clone();
                                studio.notify(&message, true);
                            }
                        }
                    },
                );
            },
        );
    }

    /// Export timeline captions to an SRT file.
    fn export_srt_captions(&mut self, studio: &mut Studio) {
        let mut segments: Vec<Segment> = studio
            .timeline()
            .clips
            .iter()
            .filter_map(|clip| {
                clip.text.as_ref().map(|text| Segment {
                    start: clip.start,
                    end: clip.start + clip.duration,
                    text: text.content.clone(),
                })
            })
            .collect();
        if segments.is_empty() {
            studio.notify("No captions on timeline to export", true);
            return;
        }
        segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        let srt_content = concat_speech::gemini::export_srt(&segments);

        let default_dir = std::env::current_dir().unwrap_or_default();
        crate::host::after_dialog(
            move || {
                crate::platform::save_file(
                    "Export Subtitles (SRT)",
                    &default_dir,
                    "subtitles.srt",
                    ("SubRip Subtitle", &["srt"]),
                )
            },
            move |studio, _, _, path| match std::fs::write(&path, srt_content) {
                Ok(_) => studio.notify(&format!("Saved SRT to {}", path.display()), false),
                Err(err) => studio.notify(&format!("Failed to save SRT: {err}"), true),
            },
        );
    }

    /// Export timeline captions to an ASS file with styling and animation.
    fn export_ass_captions(&mut self, studio: &mut Studio) {
        let mut segments: Vec<Segment> = studio
            .timeline()
            .clips
            .iter()
            .filter_map(|clip| {
                clip.text.as_ref().map(|text| Segment {
                    start: clip.start,
                    end: clip.start + clip.duration,
                    text: text.content.clone(),
                })
            })
            .collect();
        if segments.is_empty() {
            studio.notify("No captions on timeline to export", true);
            return;
        }
        segments.sort_by(|a, b| a.start.total_cmp(&b.start));

        let anim = match self.animation {
            1 => concat_speech::gemini::SubtitleAnimation::Fade,
            2 => concat_speech::gemini::SubtitleAnimation::SlideUp,
            3 => concat_speech::gemini::SubtitleAnimation::SlideDown,
            4 => concat_speech::gemini::SubtitleAnimation::SlideLeft,
            5 => concat_speech::gemini::SubtitleAnimation::SlideRight,
            6 => concat_speech::gemini::SubtitleAnimation::ZoomIn,
            7 => concat_speech::gemini::SubtitleAnimation::ZoomOut,
            8 => concat_speech::gemini::SubtitleAnimation::Bounce,
            9 => concat_speech::gemini::SubtitleAnimation::DropStamp,
            10 => concat_speech::gemini::SubtitleAnimation::GlitchShake,
            11 => concat_speech::gemini::SubtitleAnimation::Pulse,
            12 => concat_speech::gemini::SubtitleAnimation::Drift,
            _ => concat_speech::gemini::SubtitleAnimation::None,
        };

        let shape = match self.highlight_style {
            1 => concat_speech::gemini::HighlightShape::Rounded,
            2 => concat_speech::gemini::HighlightShape::Square,
            3 => concat_speech::gemini::HighlightShape::Underline,
            4 => concat_speech::gemini::HighlightShape::Highlight,
            5 => concat_speech::gemini::HighlightShape::LiquidGlassWhite,
            _ => concat_speech::gemini::HighlightShape::None,
        };

        let style = concat_speech::gemini::AssSubtitleStyle {
            font_family: "Rabar_021".to_owned(),
            animation: anim,
            highlight_shape: shape,
            y_percent: 0.88,
            ..Default::default()
        };

        let ass_content = concat_speech::gemini::export_ass(&segments, &style);

        let default_dir = std::env::current_dir().unwrap_or_default();
        crate::host::after_dialog(
            move || {
                crate::platform::save_file(
                    "Export Advanced Subtitles (ASS)",
                    &default_dir,
                    "subtitles.ass",
                    ("Advanced SubStation Alpha", &["ass"]),
                )
            },
            move |studio, _, _, path| match std::fs::write(&path, ass_content) {
                Ok(_) => studio.notify(&format!("Saved ASS to {}", path.display()), false),
                Err(err) => studio.notify(&format!("Failed to save ASS: {err}"), true),
            },
        );
    }

    /// Translate all captions currently on the timeline to the selected target language.
    fn translate_timeline_captions(&mut self, studio: &mut Studio) {
        let targets: Vec<(String, TextStyle)> = studio
            .timeline()
            .clips
            .iter()
            .filter_map(|clip| {
                clip.text
                    .as_ref()
                    .map(|text| (clip.id.clone(), text.clone()))
            })
            .collect();

        if targets.is_empty() {
            studio.notify("No captions on timeline to translate", true);
            return;
        }

        let segments: Vec<Segment> = targets
            .iter()
            .map(|(_, text)| Segment {
                start: 0.0,
                end: 1.0,
                text: text.content.clone(),
            })
            .collect();

        let target_lang = TARGET_LANGUAGES
            .get(self.target_language)
            .map(|(code, _)| *code)
            .unwrap_or("ar")
            .to_owned();
        let target_name = TARGET_LANGUAGES
            .get(self.target_language)
            .map(|(_, label)| *label)
            .unwrap_or("Arabic")
            .to_owned();
        let custom_keys = studio.prefs.gemini_api_keys.clone();
        let model = studio.prefs.gemini_model.clone();

        self.running = true;
        self.message.clear();
        self.pace = Pace::new(vec![
            (Stage::Loading, 0.5),
            (Stage::Working, (segments.len() as f32 / 10.0).max(1.5)),
        ]);

        let is_rtl = target_lang == "ar"
            || target_lang == "ar-IQ"
            || target_lang == "ckb"
            || target_lang == "fa";

        spawn_in_project(
            move || {
                let translated = concat_speech::gemini::translate_captions(
                    &segments,
                    &target_lang,
                    &custom_keys,
                    model.as_deref(),
                )?;
                Ok((targets, translated, is_rtl, target_name))
            },
            |studio, _, _, result: Result<(Vec<(String, TextStyle)>, Vec<Segment>, bool, String), String>| {
                studio.captions.running = false;
                match result {
                    Ok((targets, translated, is_rtl, target_name)) => {
                        let mut commands = Vec::new();
                        for (i, (clip_id, mut style)) in targets.into_iter().enumerate() {
                            if let Some(trans) = translated.get(i) {
                                style.content = trans.text.clone();
                                if is_rtl {
                                    style.font_family = "Rabar_021".to_owned();
                                }
                                commands.push(Command::UpdateClip {
                                    clip_id,
                                    patch: concat_project::commands::ClipPatch {
                                        text: Some(Some(style)),
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                        let count = commands.len();
                        studio.apply(Command::Batch { commands });
                        studio.notify(&format!("Translated {count} captions to {target_name}"), false);
                    }
                    Err(err) => {
                        studio.captions.message = err;
                    }
                }
            },
        );
    }

    /// The sheet as Slint shows it.
    pub fn data(&self, studio: &Studio) -> CaptionsSheetData {
        let source = self
            .clip
            .as_ref()
            .and_then(|id| {
                self.sources_cache
                    .iter()
                    .position(|source| &source.id == id)
            })
            .map_or(-1, |index| index as i32);
        let labels: Vec<slint::SharedString> = self
            .sources_cache
            .iter()
            .map(|source| source.label.as_str().into())
            .collect();
        CaptionsSheetData {
            open: self.open,
            from_voice: self.from_voice,
            sources: std::rc::Rc::new(slint::VecModel::from(labels)).into(),
            source,
            text: self.text.as_str().into(),
            model: self.model as i32,
            placement: self.placement as i32,
            size: self.size as i32,
            running: self.running,
            progress: self.pace.floor,
            ceiling: self.pace.ceiling,
            step: self.pace.step,
            left: self.pace.left,
            stage: match self.pace.stage() {
                Some(Stage::Loading) => t("captions.loadingTheModel"),
                Some(Stage::Working) => t("captions.transcribing"),
                _ => t("captions.readingTheSound"),
            }
            .into(),
            ready: !installed(&studio.settings.transcribers).is_empty(),
            message: self.message.as_str().into(),
            is_kurdish_ai: self.is_kurdish_ai,
            kurdish_language: self.kurdish_language as i32,
            translate_to_kurdish: self.translate_to_kurdish,
            animation: self.animation as i32,
            highlight_style: self.highlight_style as i32,
            spoken_languages: std::rc::Rc::new(slint::VecModel::from(
                SPOKEN_LANGUAGES
                    .iter()
                    .map(|(_, label)| (*label).into())
                    .collect::<Vec<slint::SharedString>>(),
            ))
            .into(),
            target_languages: std::rc::Rc::new(slint::VecModel::from(
                TARGET_LANGUAGES
                    .iter()
                    .map(|(_, label)| (*label).into())
                    .collect::<Vec<slint::SharedString>>(),
            ))
            .into(),
            target_language: self.target_language as i32,
        }
    }
}

/// A clip on the timeline whose voice can be captioned, as the sheet's
/// list shows it.
#[derive(Clone, Debug, Default)]
pub struct Source {
    pub id: String,
    pub label: String,
}

/// The clips on the current timeline that have sound, in the order they
/// play: an audio clip, or a video whose file has an audio stream.
fn sources(studio: &Studio) -> Vec<Source> {
    let project = studio.project();
    let mut clips: Vec<_> = studio
        .timeline()
        .clips
        .iter()
        .filter(|clip| studio.clip_has_sound(clip))
        .map(|clip| {
            let name = project
                .media_by_id(&clip.media_id)
                .map_or(clip.name.as_str(), |media| media.name.as_str());
            (clip.start, clip.id.clone(), name.to_owned())
        })
        .collect();
    clips.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    clips
        .into_iter()
        .map(|(start, id, name)| Source {
            id,
            label: source_label(&name, start),
        })
        .collect()
}

/// A clip as the list names it: its file and where it starts on the
/// timeline, so two clips of one file are told apart.
fn source_label(name: &str, start: f64) -> String {
    let seconds = start.max(0.0).floor() as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{name} · {hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{name} · {minutes}:{seconds:02}")
    }
}

/// Generates animation keyframes for the given animation preset and target offset_y.
fn animation_keys(animation: usize, offset_y: f64) -> Vec<ClipKey> {
    match animation {
        1 => {
            // Fade
            vec![
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.15,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.85,
                    value: 1.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 1.0,
                    value: 0.0,
                    ease: KeyEase::IN,
                },
            ]
        }
        2 => {
            // Slide Up
            vec![
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.0,
                    value: (offset_y + 0.07).clamp(-0.5, 0.5),
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.25,
                    value: offset_y,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        3 => {
            // Slide Down
            vec![
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.0,
                    value: (offset_y - 0.07).clamp(-0.5, 0.5),
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.25,
                    value: offset_y,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        4 => {
            // Slide Left
            vec![
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.0,
                    value: -0.12,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.25,
                    value: 0.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        5 => {
            // Slide Right
            vec![
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.0,
                    value: 0.12,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.25,
                    value: 0.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        6 => {
            // Zoom In
            vec![
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 0.45,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.25,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        7 => {
            // Zoom Out
            vec![
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 1.45,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.25,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.2,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        8 => {
            // Bounce Pop
            vec![
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 0.35,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.18,
                    value: 1.25,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.28,
                    value: 0.95,
                    ease: KeyEase::IN_OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.38,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.12,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        9 => {
            // Drop Stamp
            vec![
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.0,
                    value: (offset_y - 0.14).clamp(-0.5, 0.5),
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.18,
                    value: (offset_y + 0.015).clamp(-0.5, 0.5),
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.26,
                    value: offset_y,
                    ease: KeyEase::IN_OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 1.18,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.26,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.1,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        10 => {
            // Glitch Shake
            vec![
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.05,
                    value: -0.018,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.10,
                    value: 0.022,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.15,
                    value: -0.012,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.20,
                    value: 0.008,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetX,
                    at: 0.25,
                    value: 0.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 1.12,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.25,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.08,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        11 => {
            // Pulse Beat
            vec![
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 1.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.25,
                    value: 1.10,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.50,
                    value: 1.0,
                    ease: KeyEase::IN_OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.75,
                    value: 1.08,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 1.0,
                    value: 1.0,
                    ease: KeyEase::IN_OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.12,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
            ]
        }
        12 => {
            // Kinetic Drift
            vec![
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 0.0,
                    value: 1.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Scale,
                    at: 1.0,
                    value: 1.06,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 0.0,
                    value: (offset_y + 0.012).clamp(-0.5, 0.5),
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::OffsetY,
                    at: 1.0,
                    value: (offset_y - 0.012).clamp(-0.5, 0.5),
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.0,
                    value: 0.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.15,
                    value: 1.0,
                    ease: KeyEase::OUT,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 0.85,
                    value: 1.0,
                    ease: KeyEase::LINEAR,
                },
                ClipKey {
                    property: KeyProperty::Opacity,
                    at: 1.0,
                    value: 0.0,
                    ease: KeyEase::IN,
                },
            ]
        }
        _ => Vec::new(),
    }
}

/// One caption as a title clip: its words, when and for how long, and
/// its look.
#[allow(clippy::too_many_arguments)]
fn caption_clip(
    text: String,
    start: f64,
    duration: f64,
    look: (f64, f64),
    base: Option<&TextStyle>,
    animation: usize,
    highlight_style: usize,
    is_kurdish: bool,
) -> Command {
    let (offset_y, font_size) = look;
    let mut style = match base {
        Some(base) => TextStyle {
            content: text,
            font_size,
            max_width: 0.0,
            max_height: 0.0,
            ..base.clone()
        },
        None => TextStyle {
            content: text,
            font_family: if is_kurdish {
                "Rabar_021".to_owned()
            } else {
                "Hanken Grotesk".to_owned()
            },
            font_size,
            font_weight: 700.0,
            color: "#ffffff".to_owned(),
            stroke_width: 0.003,
            stroke_color: "#000000".to_owned(),
            shadow: true,
            ..TextStyle::default()
        },
    };

    if is_kurdish {
        style.font_family = "Rabar_021".to_owned();
    }

    // Apply highlight styles ported from trplugin
    match highlight_style {
        1 => {
            // Rounded background plate
            style.background = "#000000bb".to_owned();
            style.background_radius = 0.015;
            style.background_padding_x = 0.02;
            style.background_padding_y = 0.01;
            style.color = "#ffffff".to_owned();
        }
        2 => {
            // Square / Box background plate
            style.background = "#ffff00f0".to_owned();
            style.background_radius = 0.0;
            style.background_padding_x = 0.02;
            style.background_padding_y = 0.01;
            style.color = "#000000".to_owned();
            style.stroke_width = 0.0;
        }
        3 => {
            // Underline / outline accent
            style.stroke_width = 0.006;
            style.stroke_color = "#ffcc00".to_owned();
            style.shadow = true;
        }
        4 => {
            // High-contrast Highlight box
            style.background = "#ffeb3b".to_owned();
            style.background_radius = 0.008;
            style.background_padding_x = 0.025;
            style.background_padding_y = 0.012;
            style.color = "#000000".to_owned();
            style.stroke_width = 0.0;
        }
        5 => {
            // Liquid Glass White frosted background shape
            style.background = "#ffffffd9".to_owned();
            style.background_radius = 0.035;
            style.background_padding_x = 0.04;
            style.background_padding_y = 0.018;
            style.color = "#0f172a".to_owned();
            style.stroke_width = 0.003;
            style.stroke_color = "#ffffff".to_owned();
            style.shadow = true;
        }
        6 => {
            // Liquid Glass Dark obsidian background shape
            style.background = "#0f172ae0".to_owned();
            style.background_radius = 0.035;
            style.background_padding_x = 0.04;
            style.background_padding_y = 0.018;
            style.color = "#ffffff".to_owned();
            style.stroke_width = 0.003;
            style.stroke_color = "#38bdf888".to_owned();
            style.shadow = true;
        }
        _ => {}
    }

    let keys = animation_keys(animation, offset_y);

    Command::AddTextClip {
        track_id: None,
        above: true,
        start,
        style: Some(style),
        duration: Some(duration),
        offset_y: Some(offset_y),
        keys,
    }
}

/// Longest a caption line gets before it is wrapped: about what two lines
/// of broadcast subtitle hold, and what a reader takes in at a glance.
const CAPTION_CHARS: usize = 42;

/// A script as caption lines, each with how long it stays up: a line's
/// reading time at [`CHARS_PER_SECOND`], held to one second at least so
/// a short word is not a flicker, and seven at most so a long line does
/// not hang. A line break in the script is a break the author asked for;
/// within a paragraph a sentence is a caption, and a long sentence wraps
/// at its words.
fn script_captions(text: &str) -> Vec<(String, f64)> {
    text.lines()
        .flat_map(sentences)
        .flat_map(|sentence| wrap_caption(&sentence))
        .map(|line| {
            let seconds =
                (line.chars().count() as f64 / f64::from(CHARS_PER_SECOND)).clamp(1.0, 7.0);
            (line, seconds)
        })
        .collect()
}

/// A paragraph's sentences. A full stop, question or exclamation mark ends
/// one when it is followed by space or by the end - so "3.5" and "e.g." hold
/// together - and the CJK marks end one on their own.
fn sentences(paragraph: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        let ends = match ch {
            '。' | '！' | '؟' | '?' | '!' => true,
            '.' => chars.peek().is_none_or(|next| next.is_whitespace()),
            _ => false,
        };
        if ends {
            let sentence = current.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_owned());
            }
            current.clear();
        }
    }
    let rest = current.trim();
    if !rest.is_empty() {
        out.push(rest.to_owned());
    }
    out
}

/// A sentence in lines of at most [`CAPTION_CHARS`], broken between words;
/// a word longer than a line, or a run of CJK with no spaces, is broken
/// where it must be.
fn wrap_caption(sentence: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_chars = 0;
    for word in sentence.split_whitespace() {
        let word_chars = word.chars().count();
        if line_chars > 0 && line_chars + 1 + word_chars > CAPTION_CHARS {
            lines.push(std::mem::take(&mut line));
            line_chars = 0;
        }
        if word_chars > CAPTION_CHARS {
            let mut piece = String::new();
            for ch in word.chars() {
                piece.push(ch);
                if piece.chars().count() == CAPTION_CHARS {
                    lines.push(std::mem::take(&mut piece));
                }
            }
            line = piece;
            line_chars = line.chars().count();
            continue;
        }
        if line_chars > 0 {
            line.push(' ');
            line_chars += 1;
        }
        line.push_str(word);
        line_chars += word_chars;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{script_captions, source_label};

    #[test]
    fn imported_captions_keep_timing_style_and_undo_together() {
        use concat_project::{Command, Editor, subtitles};

        let mut editor = Editor::new();
        editor
            .apply(super::caption_clip(
                "Existing title".into(),
                0.0,
                10.0,
                (0.0, 0.04),
                None,
                0,
                0,
                false,
            ))
            .unwrap();
        let cues = subtitles::parse(
            "WEBVTT\n\n00:01.125 --> 00:03.500\nسڵاو\n\n00:02.000 --> 00:02.100\nمرحبا",
        )
        .unwrap();
        let commands = cues
            .iter()
            .map(|cue| {
                super::caption_clip(
                    cue.text.clone(),
                    cue.start,
                    cue.end - cue.start,
                    (0.35, 0.05),
                    None,
                    1,
                    1,
                    false,
                )
            })
            .collect();
        editor.apply(Command::Batch { commands }).unwrap();
        let clips = &editor.project().active().clips;
        assert_eq!(clips.len(), 3);
        for (clip, cue) in clips[1..].iter().zip(&cues) {
            assert_eq!(clip.start, cue.start);
            assert!((clip.duration - (cue.end - cue.start)).abs() < 1e-9);
            assert_eq!(clip.text.as_ref().unwrap().content, cue.text);
            assert_eq!(clip.text.as_ref().unwrap().background, "#000000bb");
            assert!(!clip.keys.is_empty());
            assert_eq!(clip.offset_y, 0.35);
        }
        assert_ne!(clips[1].track_id, clips[2].track_id);
        assert!(editor.undo());
        assert_eq!(editor.project().active().clips.len(), 1);
        assert_eq!(
            editor.project().active().clips[0]
                .text
                .as_ref()
                .unwrap()
                .content,
            "Existing title"
        );
        assert!(editor.redo());
        assert_eq!(editor.project().active().clips.len(), 3);
    }

    /// A clip in the voice list is its file and where it starts, in
    /// minutes and seconds, with hours once there are any.
    #[test]
    fn a_voice_is_named_by_its_file_and_where_it_starts() {
        assert_eq!(source_label("Interview.mp4", 0.0), "Interview.mp4 · 0:00");
        assert_eq!(source_label("Interview.mp4", 72.9), "Interview.mp4 · 1:12");
        assert_eq!(source_label("Talk.wav", 3_725.0), "Talk.wav · 1:02:05");
        assert_eq!(source_label("Talk.wav", -1.0), "Talk.wav · 0:00");
    }

    /// A script becomes one caption per sentence, a hand line break is
    /// kept, a long sentence wraps at its words, and each line is held for
    /// its reading time within one to seven seconds.
    #[test]
    fn a_script_is_cut_into_readable_lines() {
        let lines = script_captions(
            "Hello there. This is version 3.5, mind!\n\nA sentence that runs on for far \
             longer than a caption line has any business running on for. Ok?",
        );
        let text: Vec<&str> = lines.iter().map(|(line, _)| line.as_str()).collect();
        assert_eq!(
            text,
            [
                "Hello there.",
                "This is version 3.5, mind!",
                "A sentence that runs on for far longer",
                "than a caption line has any business",
                "running on for.",
                "Ok?",
            ]
        );
        assert!(
            lines
                .iter()
                .all(|(_, seconds)| (1.0..=7.0).contains(seconds))
        );
        assert_eq!(lines[0].1, 1.0);
        assert!(lines[2].1 > lines[0].1);
        assert!(script_captions("  \n ").is_empty());
        assert_eq!(script_captions("你好。再见！").len(), 2);
    }
}
