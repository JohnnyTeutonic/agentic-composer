export const meta = {
  name: 'instrumentals-revise',
  description: 'Critic -> producer -> verifier revision round for the three instrumentals',
  whenToUse: 'One revision round (rev N -> N+1) of instrumentals/ pieces; pass human_notes to steer it',
  phases: [
    { title: 'Critique', detail: 'fresh critic per piece reads code, analysis and figure' },
    { title: 'Produce', detail: 'producer implements rev-gated changes, renders, checks gates' },
    { title: 'Verify', detail: 'compare old and new revision, flag regressions' },
    { title: 'Repair', detail: 'one repair pass if technical gates fail' },
  ],
}

const ROOT = 'c:/Users/jonat/OneDrive/Documents/research_portfolio_complete'
const PROJ = ROOT + '/instrumentals'
const FROM = args.from_rev
const TO = args.to_rev
const HUMAN = args.human_notes || {}

const CONTEXT = `You are working on an experimental instrumental music project at ${PROJ}.
Three original instrumentals are composed as Python code (numpy synthesis engine in engine/, one
file per piece in pieces/, rendered by render.py). Aesthetic reference points are production
principles associated with 1970s-early-80s art-rock, electronic funk and ambient (Another Green
World, Low, Freedom of Choice) -- never their melodies, riffs or arrangements.
Key files: research/production_notes.md (the rules R1-R23 and M1-M3 the pieces implement),
engine/rules.py, engine/instruments.py (voices and their patch keys), engine/song.py (Track,
inserts: tape, chorus, ringmod, decimate, comp, eq, hp, lp, drive, fuzz, softclip, lp_auto,
varispeed, gate), engine/fx.py.
Each render writes out/<slug>/v<rev>/: the WAV, analysis.json (ffmpeg EBU R128 loudness, true
peak, LRA, band shares, centroid, stereo correlation, tempo estimate, onset grid, self-similarity
stats) and analysis.png (spectrogram with section labels, short-term loudness, arrangement map,
bar-by-bar self-similarity, onset-position histogram, band shares). Read the PNG with the Read tool.
Nobody in this pipeline can hear audio. Never claim to have listened; reason from the score data,
the code and the measurements, and say so. Musical interest is the primary objective; technical
measurements are guard rails, not goals.`

const CRITIQUE_SCHEMA = {
  type: 'object',
  properties: {
    strengths: { type: 'array', items: { type: 'string' }, description: 'what works and must be kept' },
    weaknesses: {
      type: 'array', maxItems: 5,
      items: {
        type: 'object',
        properties: {
          area: { type: 'string', enum: ['development', 'repetition', 'groove', 'sound_design', 'mix', 'harmony', 'form'] },
          observation: { type: 'string' },
          evidence: { type: 'string', description: 'bar numbers, times, metric values, code lines' },
          change: { type: 'string', description: 'concrete change, specific enough to implement' },
          rule: { type: 'string', description: 'R/M rule it serves, or new rule' },
        },
        required: ['area', 'observation', 'evidence', 'change'],
      },
    },
  },
  required: ['strengths', 'weaknesses'],
}

const PRODUCE_SCHEMA = {
  type: 'object',
  properties: {
    implemented: { type: 'array', items: { type: 'string' } },
    skipped: { type: 'array', items: { type: 'string' }, description: 'critique items not done, with reason' },
    renders: { type: 'integer' },
    metrics: { type: 'object', description: 'final I_lufs, true_peak_dbtp, LRA_lu, limiter_max_gr_db, clipped_samples, audible_s, spectral_centroid_hz' },
    gates_pass: { type: 'boolean' },
    engine_requests: { type: 'array', items: { type: 'string' }, description: 'engine capabilities you needed but could not add' },
    rev_note: { type: 'string' },
  },
  required: ['implemented', 'skipped', 'renders', 'metrics', 'gates_pass', 'rev_note'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    addressed: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string' },
          status: { type: 'string', enum: ['done', 'partial', 'not_done'] },
          evidence: { type: 'string' },
        },
        required: ['item', 'status', 'evidence'],
      },
    },
    regressions: { type: 'array', items: { type: 'string' } },
    gates_pass: { type: 'boolean' },
    verdict: { type: 'string', enum: ['accept', 'revise'] },
    next_suggestions: { type: 'array', items: { type: 'string' } },
  },
  required: ['addressed', 'regressions', 'gates_pass', 'verdict', 'next_suggestions'],
}

function gates(p) {
  return `Technical gates for v${TO}: ffmpeg true_peak_dbtp <= -1.0; clipped_samples == 0;
render.limiter_max_gr_db <= 3.0; ffmpeg I_lufs within 0.5 LU of ${p.target}; audible_s between 90 and 180.`
}

function humanBlock(p) {
  const h = HUMAN[p.key]
  return h ? `\nJONATHAN'S CRITIQUE (he has listened; this outranks everything else, including the
critic): ${h}\n` : ''
}

function renderCmd(p) {
  return `cd ${ROOT} && python scripts/cpu_budget.py run --session music --job "instrumentals ${p.key} v${TO}" --cores 1 --resume "cd instrumentals && python render.py ${p.key} ${TO}" -- python instrumentals/render.py ${p.key} ${TO}`
}

const PRODUCER_RULES = (p) => `Rules:
- Edit ONLY ${p.file}. Do not touch engine/, render.py, other pieces, or anything outside ${PROJ}.
  If a capability is missing from the engine, build it inside the piece file (note generation,
  per-note params, automation points) or list it under engine_requests.
- Every change must be gated on rev >= ${TO} so that revisions 1..${FROM} still render exactly as
  before. Existing "rev >= 2"/v2 branches stay active for rev ${TO}. Follow the file's existing
  pattern (v2 flags, REV_NOTES / notes dicts). Add a revision note for rev ${TO} in the file's notes
  dict, in plain Commonwealth-English sentences.
- For rev >= ${TO} set master "true_peak": True and "linear_phase_mono": True. If the limiter takes
  more than 3 dB, prefer the "softclip" insert on transient-heavy tracks (kw shave_db, e.g. 2-3)
  or rebalancing over lowering the loudness target; lowering the target is the last resort.
- Render with exactly this command (it waits for a CPU slot; a render takes 1-3 minutes):
  ${renderCmd(p)}
  At most 3 renders. After each, read out/${p.slug}/v${TO}/analysis.json and analysis.png.
- Use the Edit/Write tools for code changes. Never write Python containing backslashes through a
  Bash heredoc or python -c (backslashes get collapsed).
- Do not git commit. Do not delete earlier versions' outputs.
${gates(p)}`

const results = await pipeline(
  args.pieces,
  (p) => agent(`${CONTEXT}

You are the CRITIC for piece ${p.key.toUpperCase()}, "${p.title}". Brief: ${p.brief}
Piece file: ${PROJ}/${p.file}. Current revision: v${FROM}.
Read the piece file, research/production_notes.md, out/${p.slug}/v${FROM}/analysis.json and
out/${p.slug}/v${FROM}/analysis.png, and out/${p.slug}/v1/analysis.png for where it started.
${humanBlock(p)}
Identify the 3-5 weaknesses that most limit this piece's musical interest: musical development,
repetition, groove, sound design, harmony, form, and mix. At least three must be musical
(composition, arrangement or sound design), not mix. For each, cite evidence (bars, seconds,
metric values, code) and give a concrete change the producer can implement in the piece file with
the existing engine (voices: synth_bass, blip, fm, pluck, pad, lead, organ, metal, breath, drums;
inserts listed above). Also list the strengths that must survive the revision. Aim high: the
target is a piece a serious listener would want to hear twice, not one that passes metrics.`,
    { label: `critic:${p.key}`, phase: 'Critique', schema: CRITIQUE_SCHEMA }),

  (crit, p) => agent(`${CONTEXT}

You are the PRODUCER for piece ${p.key.toUpperCase()}, "${p.title}". Brief: ${p.brief}
Implement revision v${TO} of ${PROJ}/${p.file} from this critique of v${FROM}:
${JSON.stringify(crit, null, 1)}
${humanBlock(p)}
${PRODUCER_RULES(p)}
Work: read the piece file and the engine files you need; implement the changes in priority order;
render; inspect analysis.png and analysis.json; fix problems (musical or technical) and re-render
if needed. Report what you implemented, what you skipped and why, the final metrics, whether every
gate passes, and the revision note text.`,
    { label: `producer:${p.key}`, phase: 'Produce', schema: PRODUCE_SCHEMA })
    .then(prod => ({ crit, prod })),

  (r, p) => agent(`${CONTEXT}

You are the VERIFIER for piece ${p.key.toUpperCase()}, "${p.title}". Be sceptical: check claims
against the files, do not take the producer's report on trust.
Critique of v${FROM}: ${JSON.stringify(r.crit.weaknesses, null, 1)}
Producer's report: ${JSON.stringify(r.prod, null, 1)}
Compare out/${p.slug}/v${FROM}/ and out/${p.slug}/v${TO}/ (analysis.json, analysis.png, score.json,
config.json) and read the rev >= ${TO} code in ${PROJ}/${p.file}. For each critique item, decide
done / partial / not_done with evidence. List regressions: anything musically or technically worse
in v${TO} (lost strengths, new clutter, measurements that got worse). Check that revisions before
v${TO} are still gated unchanged in the code. ${gates(p)}
Verdict: accept, or revise if a gate fails or a regression outweighs the gains. Give up to five
next_suggestions for the following round.`,
    { label: `verifier:${p.key}`, phase: 'Verify', schema: VERIFY_SCHEMA })
    .then(ver => ({ ...r, ver })),

  async (r, p) => {
    if (r.ver.gates_pass && r.ver.verdict === 'accept') return { piece: p.key, ...r }
    const fix = await agent(`${CONTEXT}

You are the REPAIR producer for piece ${p.key.toUpperCase()}, "${p.title}", revision v${TO}.
The verifier asked for revision. Verifier report: ${JSON.stringify(r.ver, null, 1)}
Producer's earlier report: ${JSON.stringify(r.prod, null, 1)}
Fix the failed gates and the regressions in ${PROJ}/${p.file} (rev >= ${TO} code only), keeping the
musical gains. ${PRODUCER_RULES(p)}`,
      { label: `repair:${p.key}`, phase: 'Repair', schema: PRODUCE_SCHEMA })
    return { piece: p.key, ...r, repair: fix }
  },
)

return results.filter(Boolean)
