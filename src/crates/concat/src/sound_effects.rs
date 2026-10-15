// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Built-in Sound Effects library ported from primertextanimator suite.
//! Provides categorized whooshes, impacts, transitions, foley, and game audio
//! effects embedded and ready to drop onto the timeline or media bin.

use std::path::PathBuf;
use concat_host::AppDirs;

/// One sound effect in the library.
#[derive(Clone, Copy, Debug)]
pub struct BuiltinSfx {
    /// Unique identifier for the sound effect.
    pub id: &'static str,
    /// Display name shown in UI cards.
    pub name: &'static str,
    /// Category grouping (e.g., Transitions, Impacts, Game & UI, Foley).
    pub category: &'static str,
    /// Filename on disk.
    pub filename: &'static str,
    /// Embedded audio binary data.
    pub bytes: &'static [u8],
}

/// All built-in sound effects bundled with Concat.
pub static BUILTIN_SFX: &[BuiltinSfx] = &[
    // Transitions & Whooshes
    BuiltinSfx {
        id: "whoosh-gust",
        name: "Whoosh Wind Gust",
        category: "Transitions",
        filename: "whoosh-gust.wav",
        bytes: include_bytes!("../sound-effects/whoosh-gust.wav"),
    },
    BuiltinSfx {
        id: "whoosh-fire",
        name: "Fire Whoosh",
        category: "Transitions",
        filename: "whoosh-fire.wav",
        bytes: include_bytes!("../sound-effects/whoosh-fire.wav"),
    },
    BuiltinSfx {
        id: "whoosh-breath",
        name: "Airy Breath Swoosh",
        category: "Transitions",
        filename: "whoosh-breath.wav",
        bytes: include_bytes!("../sound-effects/whoosh-breath.wav"),
    },
    BuiltinSfx {
        id: "transition-stinger",
        name: "Dust Storm Stinger",
        category: "Transitions",
        filename: "transition-stinger.wav",
        bytes: include_bytes!("../sound-effects/transition-stinger.wav"),
    },

    // Impacts & Hits
    BuiltinSfx {
        id: "impact-deep",
        name: "Deep Cinematic Impact",
        category: "Impacts",
        filename: "impact-deep.wav",
        bytes: include_bytes!("../sound-effects/impact-deep.wav"),
    },
    BuiltinSfx {
        id: "impact-bass-drop",
        name: "Heavy Bass Drop",
        category: "Impacts",
        filename: "impact-bass-drop.wav",
        bytes: include_bytes!("../sound-effects/impact-bass-drop.wav"),
    },
    BuiltinSfx {
        id: "impact-crisis",
        name: "Crisis Rattling Hit",
        category: "Impacts",
        filename: "impact-crisis.wav",
        bytes: include_bytes!("../sound-effects/impact-crisis.wav"),
    },
    BuiltinSfx {
        id: "impact-sub-bass",
        name: "Stuttering Sub Bass",
        category: "Impacts",
        filename: "impact-sub-bass.wav",
        bytes: include_bytes!("../sound-effects/impact-sub-bass.wav"),
    },
    BuiltinSfx {
        id: "impact-drum-sweeper",
        name: "Cinematic Drum Sweeper",
        category: "Impacts",
        filename: "impact-drum-sweeper.wav",
        bytes: include_bytes!("../sound-effects/impact-drum-sweeper.wav"),
    },
    BuiltinSfx {
        id: "impact-percussion",
        name: "Sub Drop Percussion",
        category: "Impacts",
        filename: "impact-percussion.wav",
        bytes: include_bytes!("../sound-effects/impact-percussion.wav"),
    },

    // Game & UI
    BuiltinSfx {
        id: "game-level-up",
        name: "Level Up Complete",
        category: "Game & UI",
        filename: "game-level-up.wav",
        bytes: include_bytes!("../sound-effects/game-level-up.wav"),
    },
    BuiltinSfx {
        id: "game-fireball",
        name: "Fireball Spell Element",
        category: "Game & UI",
        filename: "game-fireball.wav",
        bytes: include_bytes!("../sound-effects/game-fireball.wav"),
    },
    BuiltinSfx {
        id: "ui-ping",
        name: "Ping Tone Reverberant",
        category: "Game & UI",
        filename: "ui-ping.wav",
        bytes: include_bytes!("../sound-effects/ui-ping.wav"),
    },
    BuiltinSfx {
        id: "foley-marker-sketch",
        name: "Marker Sketch Drawing",
        category: "Game & UI",
        filename: "foley-marker-sketch.wav",
        bytes: include_bytes!("../sound-effects/foley-marker-sketch.wav"),
    },

    // Foley & Real Life
    BuiltinSfx {
        id: "foley-car-skid",
        name: "Car Brake Skid Turn",
        category: "Foley",
        filename: "foley-car-skid.wav",
        bytes: include_bytes!("../sound-effects/foley-car-skid.wav"),
    },
    BuiltinSfx {
        id: "foley-ceramics-shatter",
        name: "Ceramics Shattering Shards",
        category: "Foley",
        filename: "foley-ceramics-shatter.wav",
        bytes: include_bytes!("../sound-effects/foley-ceramics-shatter.wav"),
    },
    BuiltinSfx {
        id: "foley-door-shut",
        name: "Door Shut Slam",
        category: "Foley",
        filename: "foley-door-shut.wav",
        bytes: include_bytes!("../sound-effects/foley-door-shut.wav"),
    },
    BuiltinSfx {
        id: "logo-ethernal",
        name: "Ethernal Brand Opener",
        category: "Foley",
        filename: "logo-ethernal.wav",
        bytes: include_bytes!("../sound-effects/logo-ethernal.wav"),
    },
];

/// Ensures the sound effect file exists on disk in `dirs.data/sound-effects/`
/// and returns its path for probing or importing into project.
pub fn ensure_sfx_on_disk(dirs: &AppDirs, sfx: &BuiltinSfx) -> std::io::Result<PathBuf> {
    let sfx_dir = dirs.data.join("sound-effects");
    std::fs::create_dir_all(&sfx_dir)?;
    let dest = sfx_dir.join(sfx.filename);
    if !dest.exists() || std::fs::metadata(&dest)?.len() != sfx.bytes.len() as u64 {
        std::fs::write(&dest, sfx.bytes)?;
    }
    Ok(dest)
}

/// Finds a built-in sound effect by id.
pub fn find_sfx(id: &str) -> Option<&'static BuiltinSfx> {
    BUILTIN_SFX.iter().find(|sfx| sfx.id == id)
}
