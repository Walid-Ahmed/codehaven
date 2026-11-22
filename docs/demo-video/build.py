"""
build.py
Build the narrated demo video (docs/assets/codehaven-demo.mp4) from the slides in slides/.

What it does: records one narration clip per slide, trims the silence around each clip, shows
every slide for as long as its clip lasts, cross-fades between slides, and sets the loudness
to -16 LUFS.

This is a documentation tool, not part of CodeHaven. CodeHaven itself runs offline; this script
does not. It sends the narration text below to OpenAI's speech service (model gpt-4o-mini-tts),
which costs a few US cents for the whole video.

Prerequisites: ffmpeg and ffprobe on PATH, `pip install openai`, and OPENAI_API_KEY set in the
environment.

Run (from the repository root):
    python docs/demo-video/build.py

Expected output: one "NN start ... length ..." line per slide, then
"total 105.0s -> .../docs/assets/codehaven-demo.mp4".

Recorded clips are kept in voice/ (ignored by Git) and reused on the next run, so a rebuild
calls the speech service only for clips that are missing. After changing a line in LINES,
delete that line's clip (voice/NN.wav) so it is recorded again. The number of slides must
equal the number of lines.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "assets" / "codehaven-demo.mp4"
MODEL, VOICE = "gpt-4o-mini-tts", "ash"
STYLE = ("Calm, confident, warm narrator for a software product film. Measured pace, clear diction, "
         "slight pauses at commas. Not salesy.")
# Seconds: cross-fade between slides, pause before and after each clip, shortest time on one slide.
FADE, LEAD, TAIL, SHORTEST = 0.5, 0.6, 0.8, 5.0

# One line per slide, in order. Names are spelled the way they should be spoken.
LINES = [
    "Code Haven is a private assistant that helps you understand code. It runs entirely on your own computer, "
    "works offline, and never changes your files.",
    "Start it, and a page opens in your browser. Choose a project folder, or upload a single file.",
    "Code Haven indexes the project in seconds, then suggests questions to get you started.",
    "Ask in plain English. A model running on your own machine answers, and every point names the file and "
    "the lines it came from.",
    "Click a citation to see the exact lines, highlighted, so you can check the answer yourself.",
    "Ask for a review of a file. Here it finds a function that promises, at or below the threshold, "
    "but the code only checks, less than.",
    "Ask about your uncommitted changes, or your last commit, and it explains the Git diff in plain words.",
    "Ask which functions have no tests, and it lists them, each with its file and lines.",
    "The map shows how your files import each other.",
    "Flashcards turn a file into study questions.",
    "Turn a card over to see the answer, with the source lines behind it.",
    "There is a dark theme too. Your chats stay on your computer.",
    "Find Code Haven on GitHub, at github dot com, slash Walid Ahmed, slash code haven.",
]


def run(*command):
    """Run one program and fail loudly if it does; its output is kept out of the terminal."""
    subprocess.run(command, check=True, capture_output=True)


def seconds(path):
    """The length of an audio or video file in seconds, read with ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out)


def record(missing):
    """Record the narration clips that do not exist yet. `missing` is a list of (line, path)."""
    from openai import OpenAI  # Imported here so a rebuild from saved clips needs neither the library nor a key.
    client = OpenAI()
    for line, clip in missing:
        with client.audio.speech.with_streaming_response.create(
                model=MODEL, voice=VOICE, input=line, instructions=STYLE, response_format="wav") as reply:
            reply.stream_to_file(clip)


def main():
    """Build the demo video: record each narration line, pair it with its slide, and join the clips."""
    slides = sorted((HERE / "slides").glob("*.png"))
    if len(slides) != len(LINES):
        sys.exit(f"{len(slides)} slides but {len(LINES)} narration lines; they must match.")
    voice = HERE / "voice"
    voice.mkdir(exist_ok=True)
    takes = [voice / f"{index:02d}.wav" for index in range(len(LINES))]
    missing = [(line, take) for line, take in zip(LINES, takes) if not take.exists()]
    if missing:
        record(missing)

    lengths, clips = [], []
    for take in takes:
        # The speech service leaves up to two seconds of silence around a clip; cut it so every
        # slide gets the same pause before and after the voice.
        clip = take.with_name(take.stem + "_trim.wav")
        quiet = "silenceremove=start_periods=1:start_threshold=-45dB,areverse"
        run("ffmpeg", "-y", "-v", "error", "-i", str(take), "-af", f"{quiet},{quiet}", str(clip))
        clips.append(clip)
        lengths.append(max(SHORTEST, LEAD + seconds(clip) + TAIL))
    lengths[-1] += 2.0  # Hold the repository address on screen.

    # Each slide starts FADE seconds before the previous one ends, so the two overlap during the cross-fade.
    starts, clock = [], 0.0
    for length in lengths:
        starts.append(clock)
        clock += length - FADE
    total = clock + FADE

    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for slide, length in zip(slides, lengths):
        command += ["-loop", "1", "-framerate", "30", "-t", f"{length:.3f}", "-i", str(slide)]
    for clip in clips:
        command += ["-i", str(clip)]
    count = len(slides)
    parts, last = [], "0:v"
    for index in range(1, count):
        name = f"v{index}"
        parts.append(f"[{last}][{index}:v]xfade=transition=fade:duration={FADE}:offset={starts[index]:.3f}[{name}]")
        last = name
    parts.append(f"[{last}]format=yuv420p[video]")
    for index in range(count):
        delay = int((starts[index] + LEAD) * 1000)
        parts.append(f"[{count + index}:a]aresample=48000,adelay={delay}:all=1[a{index}]")
    parts.append("".join(f"[a{index}]" for index in range(count))
                 + f"amix=inputs={count}:normalize=0,apad,atrim=0:{total:.3f},loudnorm=I=-16:TP=-1.5[audio]")
    command += ["-filter_complex", ";".join(parts), "-map", "[video]", "-map", "[audio]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", "30",
                # loudnorm raises the sample rate to 192 kHz internally; bring it back to 48 kHz.
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                "-t", f"{total:.3f}", "-movflags", "+faststart", str(OUT)]
    subprocess.run(command, check=True)
    for index, (start, length) in enumerate(zip(starts, lengths)):
        print(f"{index:02d} start {start:6.2f} length {length:5.2f}")
    print(f"total {total:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
