# Production diary

Kept by Claude Opus 5.5, 3 November 2026; revision 3 by the agentic loop (critic, producer and
verifier agents, also Claude Opus 5.5). No one in this process can hear the audio. Every
judgement below comes from the score data, the code and the measurements (spectrograms, loudness
curves, arrangement maps, bar-by-bar self-similarity, onset histograms), never from listening.

## Toolchain

The Ableton routes all need a running, licensed Ableton Live, and none renders headless (see
`research/toolchain_audit.md`). So the project uses its own numpy/scipy engine, with ffmpeg as an
independent loudness meter. The decision that mattered most was to make every sound from code: no
sample licensing questions, every timbre editable as a parameter, and every render reproducible
from a seed.

## Revision 1: first complete arrangements

**A. Rivet Logic.** The design: a one-bar syncopated bass cell in E (onsets 0, 3, 6, 7, 10, 13, 14
of 16; the root on the downbeat, the octave and flat seventh on off-beats), with a second cell that
slides G to G sharp to contest the third. A three-note square-wave hook placed 3+3+2, cut to under
half a sixteenth. Muted guitar dyads as percussion. A dry kit with fixed 54% swing. The form is
built by addition and subtraction: drums alone, then bass, then hook and guitar, then the hook
displaced one sixteenth per bar, then a break that removes bass and hats, then one 2/4 bar of
dropped-pitch snare. After that come a lift to A with a square-wave melody, a return to E with
organ fourths, a C / F sharp tritone bridge with a formant pad, and a final section that removes
one element per bar to a dead stop.

Measured: −14.5 LUFS, true peak −1.1 dBTP, no clipping, tempo estimate 112.3 BPM (nominal 112).
What the analysis showed:
- the mix was very dark (centroid 325 Hz, mids 3.3%, presence 1.8%), so the hook and guitar the
  piece depends on were buried;
- it was nearly mono (correlation 0.98);
- the limiter took 4 dB;
- loudness was flat (2.6 LU) apart from the break.

In the score, the bass cell never varied in 40 bars (R3 was written but never called). There was
one fill in 147 seconds, and the lift's melody appeared once and never returned.

**B. Lantern Arithmetic.** Four loops with periods of 11.3, 17.9, 23.3 and 29.1 seconds (R16),
over a pedal that moves four times (Db, Gb lydian, Bb, Ab, Db). At each harmony change the loops
follow by nearest-tone voice leading, so repeating material is re-harmonised rather than replaced.
Each loop repeat may change one thing: a pitch, a timbre or its placement (R18). Sparse FM events
arrive by a Poisson process with a minimum gap, choosing pitches by weighted walk. Composed
treatments: reversed swells landing on each harmony change, two tape-speed dips, a ring-modulation
window in the Bb section, wandering band-passed noise, and a faint 5-in-13 wood-block clock only in
the middle.

Measured: −16.9 LUFS (target −17), no limiting. The arrangement map showed the loops drifting
against each other as intended. Problems: the drone dominated (bass band 74%, centroid 149 Hz,
nothing above 4 kHz), and loudness hardly moved across the piece.

**C. Concrete Tide.** A two-bar bass melody on a Karplus-Strong bass, compressed, driven and
mixed loudest. Its answering bar alternates dorian E and phrygian E flat between sections (R22).
The percussion: kick, a half-time dropped-pitch snare, Euclidean wood blocks (5 in 16), struck
pipes (two 3-in-8 rotations) and shaker. The pads are stacked fifths and sus voicings. The form
alternates dense and space sections (R20). Space 2 removes the bass entirely, leaving a Bb lydian
choir cluster and the lead; the climax adds fuzz power chords and a choir an octave up. The coda
removes one bass onset per bar.

Measured: −13.9 LUFS, LRA 6.4 LU. The self-similarity matrix showed the dense/space blocks
clearly. Problems: space 1 barely dropped (the pad kept sustaining), presence was about 1%, and
the limiter took 3.5 dB.

**Engine bug found through the analysis.** The "mono below 120 Hz" master stage subtracted an IIR
low-pass to get the high band, and its phase shift left a large low residue in that band. B's lows
measured 0.18 correlation where they should have been close to 1. A zero-phase split fixes it (test:
uncorrelated noise below 100 Hz went from −0.13 to 0.94). The fix is opt-in from revision 2, so
revision 1 still renders as it did. A second analysis bug: the chroma was indexed from A and
included the smeared lowest bins. That is fixed, and the self-similarity features are now
standardised per dimension, so sections contrast instead of all sharing the bass-heavy average.

## Revision 2: my own revision

Composition randomness and render noise were seeded with the revision number, so v2 of B would
have been a new random draw, not a revision. Both are now pinned to revision 1's seed. From here
on, a revision differs only by its stated changes.

**A.**
- Mix moved toward the mids: bass −2.5 dB with less sub, hook +3.5 dB and brighter, guitar
  +3.5 dB, hats on their own track panned right and +4 dB, presence shelf.
- The bass cell now mutates every fourth bar from the hook on, one move at a time (a nudge or a
  removed onset, R3).
- The hook is foreshadowed: the four bars before it arrives play it dark and quiet, opening a
  little each bar.
- Fills at each section end.
- The lift's melody returns over E in the final section, so the lift is remembered.

Result: centroid 540 Hz, mids 6.9%. The self-similarity matrix now shows the hook and displace
block, a distinct lift and a distinct bridge, where v1 was nearly uniform. New problems: the
brighter top produced inter-sample overs (true peak −0.3 dBTP), and the limiter took 4.9 dB.

**B.**
- Drone −4 dB and darker; glass, bell and events up 2.5–3 dB, the glass brighter.
- An arc: the swell loop rests through the Bb section, extra high events thicken the Ab section,
  and at home the loops leave one at a time until glass and drone remain.

The arc shows in the arrangement map but barely in the loudness curve, and the top end is still
nearly empty (presence 0.1%).

**C.** Space 1 now opens: the pad rests there, leaving bass, metal and echo tails, with a high pipe
answer each bar. Wood, pipe and plate are lifted for presence and the kick trimmed. LRA rose to
6.6 LU and the limiter fell to 3.25 dB.

**Engine, after revision 2.** An opt-in true-peak limiter detects on a 4x oversampled copy (test:
a near-Nyquist sine at −1.5 dB sample peak measured −0.47 dBTP before and −1.38 after). A
`softclip` insert shaves transient peaks on drum tracks, so loudness comes from balance rather than
limiting.

## Revision 3: the agentic loop, and the first listening

Revision 3 was made by the agentic loop (`workflow/revise.js`). A fresh critic agent read each
piece's v2 code, rules and figures; a producer agent implemented the critique; a verifier was to
check the result. The producers were still iterating when the workflow was stopped, so each v3 is
the producer's latest render rather than a verified final. Each piece file's notes for revision 3
say what changed.

**Jonathan listened to A v3 and called it fire**, the first ear-based judgement in the project.
That render breaks the loop's 3 dB limiting gate by 3.5 dB, and A's producer was running
limiter probes to "fix" it when the workflow was stopped. The heard render is frozen in
`out/a_rivet_logic/v3_heard/`. The lesson is built into the LangGraph agent: limiting and
loudness are advisory gates, every re-render keeps the earlier take, and earlier revisions are
hash-locked.

What made A v3 work, by the code: the hook moved into the bass's gaps (an interlocking 3+3+3
pickup instead of a doubling); every section got its own kick map; the bridge became a valley, so
the return lands; the final is the one place everything plays; major and minor trade by phrase.

B v3 departs further harmonically (an A major ninth middle, C withheld from 72 to 130 seconds),
accretes its loops, leaves a hole in the drone, and develops a four-note motif by chance
fragments. LRA rose from 3.9 to 8.3 LU. C v3 builds each dense section in stages, stops for a
bar before the climax, turns phrygian in the harmony as well as the bass, and walks the fuzz
across the bar. Its limiter takes 4.2 dB, above the old gate.

Process faults found and fixed: analyse-only runs rewrote the MIDI and score from newer code
while keeping the older WAV (it now leaves composition files alone); B and C were re-rendered from
their final v3 code; the B render reproduced bit for bit.

## C revisions 4 to 6: the LangGraph agent, two rejections and a new piece

C v4 was the LangGraph agent's first round: Gemini listened, Claude Opus 5.5 critiqued and
composed. Jonathan rated it 5/10: very basic, no lushness, no hook or groove. He set the aim
that now governs C: a hooky guitar riff, a rhythmic bass, lush textures and a Trent Reznor
influence (principles only, rules R24-R29).

C v5 answered with a double-tracked guitar riff, distorted bass, three bed layers and a loud
wall. A guest rater gave it 4/10 and Jonathan agreed: generic, start from scratch, more
bass, a better drum beat. Measurement showed why. The composer had mixed blind: the riff sat 14
to 16 dB under the bass in its own band, the beds at -22 to -34 dB, and the sub band held 5.8% of
the energy. The composer now has a `render_and_measure` tool that reports every track's level
against the mix, and every render's analysis records those levels.

C v6 is a new composition (agent run with `--fresh`; none of v1-v5's material). It is in E at
88 BPM: a guitar call and answer with a D-to-E slide, played only in the gaps of a sub-heavy
bass locked to the kick; a three-layer backbeat with a tritone metal clang; a bridge that empties
completely and a wall where everything plays, ending in a hard stop and a tape stop. Against v5
the riff now leads its band (within 5 dB of the mix), the sub band rose from 5.8% to 16.5%, and
the loudness range from 8.4 to 14.7 LU. Gemini scored it 6/10, its best so far, but heard the
snare as thin and the wall as congested in the low mids. The limiter still takes 4.7 dB.

Jonathan rated C v6 6.5/10, his best rating for C: the right rhythm and beat, "oozes sex", hooky
and swinging, with the potential of a sleeper dance hit. His one criticism: the lead melody is
too thin in texture. v6 is the version to build on.

C v7 makes only Jonathan's two changes to v6 (agent run, $0.58). Every note, section, drum and
bass part and the master are unchanged; the score and configuration diff confirms it. The hook
guitars each gained a detuned double (9 to 16 cents, 5 to 6 ms late) and a slow chorus, A and B
gained a quiet octave-up layer, and an organ-like body plays the hook's notes under it; the
original guitars came down 3 dB, so the hook group is thicker at about the same level. The pads
and drone came up 3.6 to 3.9 dB. Gemini scored v7 7/10 but still hears the lead as slightly thin.
