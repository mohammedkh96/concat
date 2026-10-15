// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Lanes: adding, removing and muting or hiding them.
//!
//! One arm per command, exactly as [`super::apply`] routes them here;
//! everything these arms share lives in the parent module.

use super::*;

/// Applies one of this module's commands. Any other is a routing error.
pub(super) fn apply(
    project: &mut Project,
    mint: &mut IdMint,
    command: Command,
) -> Result<Outcome, CommandError> {
    match command {
        Command::AddTrack { bottom } => {
            let timeline = project.active_mut();
            let id = mint.next("t");
            let track = Track {
                id: id.clone(),
                visible: true,
                muted: false,
                extra: Default::default(),
            };
            // The stack is stored bottom first: the last track is the lane
            // drawn on top.
            if bottom {
                timeline.tracks.insert(0, track);
            } else {
                timeline.tracks.push(track);
            }
            Ok(Outcome {
                created_id: Some(id),
                applied: true,
            })
        }

        Command::RemoveTrack { track_id } => {
            let timeline = project.active_mut();
            if timeline.tracks.len() <= 1 {
                return Err(CommandError::LastTrack);
            }
            let track_count = timeline.tracks.len();
            timeline.tracks.retain(|track| track.id != track_id);
            timeline.clips.retain(|clip| clip.track_id != track_id);
            // Clips only ever sit on existing tracks, so an unknown id - the
            // tolerated no-op - removes neither.
            let applied = timeline.tracks.len() != track_count;
            Ok(Outcome {
                created_id: None,
                applied,
            })
        }

        Command::DuplicateTrack { track_id } => {
            let timeline = project.active_mut();
            let Some(idx) = timeline.tracks.iter().position(|t| t.id == track_id) else {
                return Ok(Outcome::default());
            };
            let src = timeline.tracks[idx].clone();
            let new_id = mint.next("t");
            let new_track = Track {
                id: new_id.clone(),
                visible: src.visible,
                muted: src.muted,
                extra: src.extra.clone(),
            };
            // The stack is stored bottom first; insert directly above
            timeline.tracks.insert(idx + 1, new_track);

            let mut new_clips = Vec::new();
            for clip in timeline.clips.iter().filter(|c| c.track_id == track_id) {
                let mut dup = (**clip).clone();
                dup.id = mint.next("c");
                dup.track_id = new_id.clone();
                new_clips.push(std::sync::Arc::new(dup));
            }
            timeline.clips.extend(new_clips);
            Ok(Outcome {
                created_id: Some(new_id),
                applied: true,
            })
        }

        Command::SetTrackFlag {
            track_id,
            flag,
            value,
        } => {
            let timeline = project.active_mut();
            let applied = timeline
                .tracks
                .iter_mut()
                .find(|track| track.id == track_id)
                .is_some_and(|track| match flag {
                    TrackFlag::Visible => assign(&mut track.visible, value),
                    TrackFlag::Muted => assign(&mut track.muted, value),
                });
            Ok(Outcome {
                created_id: None,
                applied,
            })
        }

        _ => unreachable!("commands::apply routes only this module's commands here"),
    }
}
