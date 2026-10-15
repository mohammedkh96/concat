# Importing subtitles

Open **Captions → Import subtitles (SRT / VTT)** and choose one UTF-8 subtitle
file. This works without an AI model, API key or audio clip.

Each subtitle becomes an editable text clip at its file's timestamp, measured
from the start of the active timeline. The playhead does not offset the import.
Overlapping subtitles keep their times and occupy separate lanes. Existing
clips remain in place, and one **Undo** removes the entire import; **Redo**
restores it.

Choose placement, size, animation and highlight in the Captions dialog before
importing. These controls and your default title style determine the result.
File styling, WebVTT positioning, speaker tags and inline timing tags are not
imported. Common formatting tags are removed, text entities are decoded, and
multiline text and Unicode (including Kurdish and Arabic) are preserved.

Supported files include numbered SRT cues, WebVTT cue identifiers and settings,
WebVTT NOTE/STYLE/REGION blocks, a UTF-8 BOM, and Windows or Unix line endings.
Invalid files show an error with a line number and add no clips. Cue durations
must be at least 1/60 second and timestamps must fit within 100 hours, matching
the editor's timing limits. Convert legacy non-UTF-8 files to UTF-8 first.
