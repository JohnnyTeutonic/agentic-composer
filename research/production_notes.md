# Production research notes

These notes record production principles only. They are kept apart from the compositions in
`../pieces/`. No melody, riff, recording or arrangement from any record is transcribed or used.

The observations come from Claude Opus 5.5's background knowledge (3 November 2026). None was
checked against a source in this session, so treat the historical details as commonly reported,
not verified. The rules that follow from them are design choices and stand on their own.

## 1. Observations

**Another Green World (Eno, 1975).** Mostly short instrumentals built in the studio rather than
performed: a few instruments, each with a strongly individual timbre (treated guitar, organ,
fretless bass, synthesiser). The bass often carries the melody. Pieces read as landscapes: a
texture is set up and then lightly disturbed. Chance procedures (the Oblique Strategies cards)
were used to break habits. Most pieces have neither a drum-kit backbeat nor a conventional
climax.

**Low (Bowie, 1977).** Side one sets a tight funk-derived rhythm section (bass, drums, rhythm
guitar) against synthesisers that sound unstable and slightly wrong. The snare is famously
processed through a pitch-shifting harmoniser, so each hit drops in pitch: big and mechanical,
with no natural room. The pieces are short and often start or stop abruptly. Side two is slow
and ambient, with sustained synthesiser chords and sparse melodic events.

**Freedom of Choice (Devo, 1980).** Short, hook-first songs. Synthesiser bass, often doubled by
guitar, plays staccato syncopated figures. The drumming is extremely precise and dry, with little
fill. Parts are clipped: notes stop before the next one starts, leaving air in the groove. The
humour comes from rigidity: machine-like repetition with small jolts.

**Generative practice (Eno, *Discreet Music* 1975, *Music for Airports* 1978).** Long tape delays
feed a phrase back on itself. Several loops of different, unrelated lengths drift in and out of
phase, so the combinations keep changing without anyone composing them.

## 2. Characteristics and the rules they become

Each rule names where it is implemented (`engine/rules.py` unless stated otherwise).

### Bass figures: short, memorable, syncopated
- **R1. One-bar or two-bar cells.** A bass figure has 5–8 onsets in 16 sixteenths. At most two of
  them land on quarter-note beats, and at least one lands on the "a" of a beat (sixteenth 3, 7, 11
  or 15). `syncopation()` scores a cell; cells are written by hand and checked.
- **R2. Octave pops and the flat seventh.** The root is anchored on sixteenth 0. Its upper octave
  and the flat seventh fall on off-beats. This gives a figure that repeats without monotony.
- **R3. Variation by subtraction, not decoration.** A repeated cell changes by removing an onset
  (`thin()`) or by moving one onset a sixteenth (`nudge()`), at most one change per two bars.

### Economical guitar and synthesiser motifs
- **R4. Motifs of 2–4 notes, articulated short** (gate 30–60% of the step). Short gates make
  space for the bass. `clip()` sets the gate.
- **R5. Rhythmic displacement.** A motif is rotated by one sixteenth per repeat over a fixed
  bar (`rotate()`), so a fixed figure walks across the beat: mechanical, angular phasing.
- **R6. Rhythm guitar as percussion.** Muted plucks with the pitch barely audible, on the 16th
  grid with accents in a 3+3+2 grouping.

### Sparse, highly differentiated percussion
- **R7. One spectral region per voice.** Each percussion voice has a clearly separate spectral
  centre (kick < 100 Hz, toms 100–300 Hz, snare body 200 Hz plus noise 2–6 kHz, metal 1–4 kHz,
  hats above 7 kHz). No two voices share a region at the same time.
- **R8. Euclidean placement for the unconventional voices.** `euclid(k, n, rot)` (Bjorklund)
  places k hits in n steps. Coprime cycle lengths (for example 5 in 16 against 3 in 8) give
  patterns that keep realigning.
- **R9. Dry means dry.** In piece A the kit gets no reverb send: decay comes from the envelopes.
  Space is a choice, used elsewhere for contrast.
- **R10. The dropped-pitch snare.** Treating the snare by pitching it down over its decay
  (`drum_snare(drop=...)` in `engine/instruments.py`) gives the big, unnatural hit. A generic
  technique, applied with our own voice.

### Funk grooves with mechanical precision
- **R11. Swing without slop.** The off-beat sixteenths are delayed by a fixed swing ratio
  (52–56%), identical in every bar. The only random timing is ±2 ms on hats and guitar.
  `Grid.t()` applies it.
- **R12. Velocity accents carry the groove,** not timing. Accent maps are fixed per section.

### Unusual synthesiser timbres, modulation and tape texture
- **R13. Every sustained sound moves.** Each sustained voice carries at least one slow modulation
  (filter, pulse width, FM index or pitch drift) at a rate unrelated to the tempo.
- **R14. Tape as an instrument.** A tape stage (saturation, head bump, high-frequency loss, wow
  and flutter, hiss) sits on buses. Varispeed dips are composed events.
- **R15. Treatments over presets.** Ring modulation, sample-and-hold filtering, reversed swells
  and decimation are used as composed events, not left running.

### Generative repetition, gradual variation, controlled unpredictability
- **R16. Incommensurate loops.** Generative layers repeat with periods in roughly golden-ratio or
  prime-like relation (for example 11.3 s, 17.9 s, 23.3 s), so no combination recurs within the
  piece.
- **R17. Seeded chance.** All randomness flows from one seed per piece and version. Pitch
  choices use weighted random walks over a fixed pitch set (`walk()`); events use Poisson
  timing with minimum spacing (`poisson_times()`).
- **R18. Mutation budget.** Each repeat of a phrase may change at most one parameter (pitch,
  timbre or placement), so change is audible as change.

### Arrangement by subtraction
- **R19. Arrangement matrix.** Every piece has an explicit section-by-track matrix (`Song.sections`).
  Before a structural return, one load-bearing element (bass or kick) is removed for 2–4 bars, and
  its return marks the downbeat.
- **R20. Density alternation.** In piece C, dense and spacious sections alternate, and the space
  sections keep at least one sustaining element so that silence is never dead air.

### Harmonic ambiguity, still accessible
- **R21. Pedal plus moving upper structure.** The bass holds a pedal while upper voices move
  through stacked fourths and sus2 shapes, which belong to more than one key.
- **R22. Third withheld or contested.** The third of the tonic is avoided in the bass. When it
  appears above, major and minor forms alternate between sections (modal mixture).
- **R23. Melodic accessibility.** Melodies stay mostly stepwise within a pentatonic or modal set,
  so they remain singable even over ambiguous harmony.

## Jonathan's aim (3 November 2026, after hearing C v4)

"A track that has a hooky guitar riff, a rhythmic bass with lush sonic textures is what we're
aiming for." He heard C v4 as basic: a simple melody, no lushness, no real hook or groove,
little texture or dynamism. He asked for Trent Reznor's influence to be considered as well.
He rated A v3 highly, the piece where the hook and bass interlock and every section has its
own job.

### Industrial and textural production (Reznor-associated principles)

From Claude Opus 5.5's background knowledge (3 November 2026), not checked against sources here.
These are commonly reported traits of Reznor's records and his scores with Atticus Ross: no
riff, melody or sound of his is reproduced.

- **R24. Distortion as texture.** Saturation, bit reduction and filtered fuzz are layered under
  and around a part, not only used to make it louder. One clean element is kept against the
  dirt for contrast.
- **R25. Extreme dynamic contrast.** Near-silence and a dense, compressed wall alternate. The
  quiet passages are intimate (one exposed instrument, close and dry); the loud ones are
  saturated and wide.
- **R26. Layered beds.** A noise or drone bed of several filtered, slowly moving layers runs under
  the rhythm section and changes between sections. It is felt more than heard.
- **R27. Degraded melodic voices.** A simple, memorable melody is carried by a detuned,
  softened or decaying keyboard or guitar tone. The fragility is the hook's colour.
- **R28. Machine drums with weight.** Drums are programmed and precise, then crushed,
  pitched or side-chained so the groove pumps. Industrial hits (metal, noise bursts) take the
  place of cymbals.
- **R29. The riff is the hook.** One short guitar (or guitar-like) riff, 1–2 bars and
  rhythmically locked to the bass, is stated early, kept recognisable and re-orchestrated
  rather than replaced. A convincing guitar needs amp-like asymmetric clipping, cabinet-style
  band-limiting, and double-tracking panned left and right with small timing and pitch
  differences. Pieces may build such voices in their own file (see `engine/song.py`
  `CUSTOM_INSERTS` and `instruments.VOICES`).

## 3. Mix rules
- **M1.** The low end below 120 Hz stays mono. Kick and bass are side-chained against each
  other, or separated by envelope.
- **M2.** The loudness target is about −14 LUFS integrated for A and C and −17 for B (a quiet
  piece stays quiet), with true peak at or below −1 dBTP. Integrated loudness is never
  improved by squashing: the limiter should take at most 3 dB.
- **M3.** Every track has a reason to be audible: an element that never sounds alone and never
  changes is cut.
