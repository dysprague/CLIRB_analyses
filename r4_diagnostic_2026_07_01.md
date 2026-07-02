# R4 performance diagnostic — 2026-07-01

Investigated: R4 was hovering ~20 matches/session, then declined after tone was removed (state 1 → state 0). Hypotheses tested against 24 R4 sessions (2026-06-11 → 2026-07-01), with R5 (20 sessions) and R6 (24 sessions) as controls.

---

## TL;DR — what's going wrong

**R4's problem is a combination of (2) port-zone misunderstanding and (4) template difficulty. It is NOT primarily motivation.**

The evidence:

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| 1. Motivation drop | **Partly, minor** | licks/min dropped 5.4 → 4.3 (−20%) state 1 → 0. But she's still trying — 55 attempts/session in state 0 (vs 57 in state 1). Rears/min dropped only 3.7 → 3.2 (−13%). Not the story. |
| 2. Port-zone misunderstanding | **YES, dominant** | 47% of R4's rears in state 0 are **inside** the port exclusion zone, vs 22% (R5) and 26% (R6). She's rearing at the wrong location. |
| 3. Tracking failure | **No** | PC0 is finite 100% of the time on all rats; reasonable values 100%. Tracking is not the issue. |
| 4. Harder template | **YES** | R4's median rear-alignment MSE is **2047**, versus **1156** (R5) and **1273** (R6). Even her *best-aligned* rears sit ~1.8× further from the template than R5's do. Her template is objectively harder to match. |

The tone removal itself is not really the cause — R4's `%matched` dropped only 3 points state 1 → state 0 (32.9% → 29.5%), which is within noise. The real problem is that R5 and R6 respond to state-0 by rearing away from the port and getting matches at 45% and 41.5%; R4 rears **at** the port and gets matched at 22% there and 34% outside.

Note: the port-zone problem is a behavioral pattern (where she chooses to rear), not something the online rule can directly correct. Making the port gate stricter would just suppress more of her rewardable rears without changing where she rears. Only physical relocation or a location-contingent cue actually gives her a signal to change. See the corrected remedies section below.

---

## Detailed findings

### 1. Motivation — mostly intact

R4 state 1 vs state 0 (means over sessions):

| metric | state 1 (n=8) | state 0 (n=12) | Δ |
|---|---|---|---|
| licks/min | 5.42 | 4.34 | −20% |
| rears/min | 3.68 | 3.20 | −13% |
| attempts/session | 57.4 | 55.8 | −3% |
| rears/session | 128.9 | 109.5 | −15% |

She's licking and rearing a bit less, but *attempts* — rears with an actual lick follow-up — are essentially unchanged. She hasn't given up; she's rearing less efficiently. **Increasing water restriction to boost motivation would probably help modestly (~10–15% lift on attempts) but is not the primary lever.**

### 2. Port zone — she's parked near the water port

State-0 head-to-head, pooling across all sessions per rat:

| | % rears inside port zone | match rate inside | match rate outside |
|---|---|---|---|
| R4 | **46.8%** | **22.3%** | 34.5% |
| R5 | 21.5% | 21.9% | **45.2%** |
| R6 | 25.6% | 30.9% | 41.5% |

Two things pop out:

- **R4 rears at the port ~2× more often than R5/R6.** She hasn't learned the exclusion rule.
- **Match rate outside the port zone is R4 = 34.5% vs R5 = 45.2%.** So even when she's in the right place, she's ~25% worse at hitting the template. That's hypothesis 4 (harder template) creeping in.

The consequence for reward count: R4 attempts inside port ×22% match rate = ~10 rewards from inside-port attempts (but suppressed by the port gate anyway); attempts outside port ×34.5% = ~7 rewards. R5 gets ~40+ rewards per session from outside-port attempts alone.

### 3. Tracking — fine

`pc0_reasonable` (fraction of frames with |PC0| < 1000) is 1.00 on every session for every rat. Not a triangulation problem.

### 4. Template difficulty — R4's is genuinely harder

Median rear-alignment MSE (state 0, best 30-frame window per rear):

| | median MSE | p25 MSE |
|---|---|---|
| R4 | **2047** | 1300 |
| R5 | 1156 | 679 |
| R6 | 1273 | 855 |

Interpretation: even when the animal is rearing, the best-aligned 30-frame slice of her PC trajectory is *twice as far from her own template* as R5's is. Something about her R4_template_2 is either (a) noisier, (b) an unusual rear shape, or (c) built off a small number of matches.

R4's template metadata backs this up:

```
R4_template_1.npz: built from  5 matches, PC1 band mean 65.8, PC2 band mean 35.2 (tight)
R4_template_2.npz: built from 27 matches, PC1 band mean 83.3, PC2 band mean 52.4
R5_template_2.npz: built from 51 matches, PC1 band mean 78.5, PC2 band mean 58.6
R6_template_2.npz: built from 36 matches, PC1 band mean 74.0, PC2 band mean 46.7
```

R4's template was built from **27 examples** vs R5's **51** — smaller sample, more idiosyncratic. Her PC2 band is not narrower than R5/R6 but her template *center trajectory* apparently sits in a region she doesn't visit often. See the 1.8f plot in `multi_session.ipynb` — R4's matched-rear mean curves cluster near the template but her *unmatched* rear population is pulled a long way off.

---

## What to try (ranked by expected impact)

### Correction to an earlier draft of this section

An earlier version of this write-up suggested raising `min_port_distance` to "force" R4 away from the port. That was wrong — the gate is invisible to the animal, so tightening it just suppresses more of her rewardable rears without changing where she rears. It would make her worse. The interventions below actually change what she experiences.

### 1. **Change the port location** — biggest behavior lever

R4's 46.8% inside-port rear rate is the single most anomalous number in the dataset. She's built a spatial habit: "the rearing spot is right by the port". The only ways to break that habit are physical or via a signal she can perceive:

- **Physically relocate the water port** to a different arena wall. She learns by proximity — her reward-associated rearing spot follows the port. Behavioral cost is a few disoriented sessions.
- **Add a location-contingent cue.** If the rig can play a tone or light while she's *far enough* from the port, that gives her a signal for where good rears happen — closer to the state-1 semantics she'd already learned, but now spatial. Requires rig support.
- **Shape it with rearless location reward.** If a small bolus can be delivered simply for standing away from the port for a few seconds (pre-rear), she'd learn "far from port = good things", which then combines with her intact rearing habit. Also requires rig support.

If none of the above are practical, this may partially self-correct: R4's inside-port bias may be a legacy of state 2's tone+water pairing at a spot she happened to be near, and state 0 could grind it out over more sessions. Best guess timeline: 5–10 more sessions before you'd expect meaningful drift.

### 2. **Rebuild R4's template with more matches** — second biggest, unrelated to port zone

Her `R4_template_2` was built from only 27 matches, and her median rear-alignment MSE is ~1.8× R5's. Suggested procedure:

- Run 1–2 sessions at `bounds` scalar 1.5. This inflates her match count 3–5× — most of the extra matches will still look rear-shaped.
- Rebuild `R4_template_3.npz` from those matches (target: 100+ examples).
- Drop `bounds` back to 1.0. Keep the port gate as-is.

This is the "increasing bounds" option from your list, but *only as a template-building step*, not a permanent change. Permanent bound-inflation would reward whatever she's doing, including inside-port rears, which is the opposite of what we want.

### 3. **Reduce initial warm-up time** — small lever, cheap

R4's `pct_att_first5` in state 0 is ~22% — a meaningful fraction of her attempts happen in the first 5 min. If your setup has a task-active gate that suppresses rewards for the first 5 min (e.g. context switch at 300 s), you're throwing away real behavior.

- If there's a task-onset delay > 0, drop it or shrink it.
- If the 5-min gate is what you meant by "increasing initial time before training portion of session starts", yes — shortening it helps, but on its own is not enough.

### 4. **Water restriction — modest support**

licks/min dropped 20% state 1 → 0. Real but small. Worth tightening restriction by 10% *only in combination* with template rebuild + port-zone intervention. Don't do it standalone — you'd just be starving her while she keeps rearing in the wrong place.

### What I would NOT do

- **Do not** raise `min_port_distance` and expect it to change her behavior. It only makes the gate stricter; the animal never learns about the gate.
- **Do not** increase reward volume as a first-line intervention. Her problem is not per-reward value.
- **Do not** permanently loosen her template bounds. That removes the training signal.

---

## Recommended action sequence

1. **This session or next**: shorten any task-onset warm-up delay. Cheap, no downside.
2. **Following 1–2 sessions**: run `bounds` scalar 1.5 on R4 to harvest matches. Don't worry about reward count on these.
3. **Rebuild** `R4_template_3.npz` from those matches (100+ examples). Drop `bounds` back to 1.0.
4. **In parallel**, plan a port relocation or a location-contingent cue if the rig supports it. This is the primary behavioral fix.
5. If matched-attempt rate on outside-port rears hasn't recovered to ≥ 40% within a week of the template rebuild + port intervention, tighten water restriction by ~10%.

If any of that runs contrary to your protocol tell me — several of these levers I can't verify from the data alone (whether the port is movable, whether the rig can trigger location-contingent cues, whether `bounds` is easily adjustable per session).
