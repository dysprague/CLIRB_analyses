"""Build the self-contained summary interface (figures embedded as data URIs)."""
import os, sys, base64, json
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
os.chdir(REPO)
OUT = '/tmp/claude-1000/-home-yutaka-sprague/0709c2d8-bcbc-4d7c-b833-a6ea2773aa11/scratchpad/clirb_report.html'


def img(path, alt, cap):
    if not os.path.isfile(path):
        return f'<p class="missing">[missing figure: {path}]</p>'
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return (f'<figure><div class="fig-scroll">'
            f'<img src="data:image/png;base64,{b}" alt="{alt}"></div>'
            f'<figcaption>{cap}</figcaption></figure>')


SP = 'results/figures/spatial_confound'
CV = 'results/figures/corrector_v2'

t2 = pd.read_csv('results/corrector_v2/tier2_summary.csv')
tr = json.load(open('results/corrector_v2/train_results.json'))


def t2row(src, opt, col):
    r = t2[(t2.source == src) & (t2.option == opt)]
    return f'{r[col].iloc[0]:.3g}' if len(r) else '—'


HTML = f"""<title>CLIRB — spatial reward bias and the corrector experiment</title>
<style>
  :root {{
    --ground:#f6f7f9; --panel:#ffffff; --ink:#111820; --ink-2:#3d4854; --ink-3:#697687;
    --line:#dfe3e9; --accent:#146b7a; --accent-soft:#e4f0f2;
    --warn:#8a6410; --warn-soft:#fbf3e2; --bad:#a33a38; --bad-soft:#fbecec;
    --good:#1f6b46; --good-soft:#e6f2ec;
    --serif: ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      --ground:#0e1319; --panel:#151c24; --ink:#e9edf2; --ink-2:#b4bfcc; --ink-3:#8593a3;
      --line:#26303b; --accent:#4fb3c4; --accent-soft:#12303550;
      --warn:#d9a441; --warn-soft:#3a2f1650; --bad:#e0807e; --bad-soft:#3a201f50;
      --good:#6fc296; --good-soft:#16301f50;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#0e1319; --panel:#151c24; --ink:#e9edf2; --ink-2:#b4bfcc; --ink-3:#8593a3;
    --line:#26303b; --accent:#4fb3c4; --accent-soft:#12303550;
    --warn:#d9a441; --warn-soft:#3a2f1650; --bad:#e0807e; --bad-soft:#3a201f50;
    --good:#6fc296; --good-soft:#16301f50;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
         font-size:16.5px; line-height:1.62; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 clamp(16px,4vw,40px) 96px; }}
  .measure {{ max-width:70ch; }}
  header.top {{ border-bottom:1px solid var(--line); padding:clamp(40px,7vw,76px) 0 30px; }}
  .eyebrow {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
              text-transform:uppercase; color:var(--accent); }}
  h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(31px,4.6vw,50px);
        line-height:1.12; margin:.32em 0 .2em; text-wrap:balance; letter-spacing:-.01em; }}
  .sub {{ font-size:clamp(17px,2vw,20px); color:var(--ink-2); max-width:64ch; }}
  .meta {{ font-family:var(--mono); font-size:12px; color:var(--ink-3); margin-top:22px;
           display:flex; flex-wrap:wrap; gap:6px 20px; }}
  h2 {{ font-family:var(--serif); font-weight:600; font-size:clamp(23px,2.7vw,31px);
        margin:0 0 .3em; letter-spacing:-.005em; text-wrap:balance; }}
  h3 {{ font-size:16px; margin:2em 0 .4em; letter-spacing:.005em; }}
  section {{ padding-top:clamp(40px,5.5vw,68px); }}
  .snum {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.14em; color:var(--ink-3);
           text-transform:uppercase; }}
  p {{ margin:0 0 1em; }}
  .verdict {{ background:var(--panel); border:1px solid var(--line);
              border-left:4px solid var(--accent); border-radius:3px;
              padding:26px clamp(18px,3vw,32px); margin-top:30px; }}
  .verdict p:last-child {{ margin-bottom:0; }}
  .lede {{ font-family:var(--serif); font-size:clamp(19px,2.2vw,23px); line-height:1.45;
           color:var(--ink); }}
  table {{ border-collapse:collapse; width:100%; font-size:14px;
           font-variant-numeric:tabular-nums; }}
  .tscroll {{ overflow-x:auto; margin:1.2em 0; border:1px solid var(--line);
              border-radius:3px; background:var(--panel); }}
  th,td {{ text-align:left; padding:9px 14px; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ font-family:var(--mono); font-size:11px; letter-spacing:.07em; text-transform:uppercase;
        color:var(--ink-3); font-weight:500; background:var(--ground); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.num {{ font-family:var(--mono); }}
  .hi {{ color:var(--accent); font-weight:600; }}
  .dim {{ color:var(--ink-3); }}
  figure {{ margin:1.6em 0; background:var(--panel); border:1px solid var(--line);
            border-radius:3px; padding:14px; }}
  .fig-scroll {{ overflow-x:auto; }}
  figure img {{ display:block; width:100%; min-width:640px; height:auto; border-radius:2px; }}
  figcaption {{ font-size:13.5px; color:var(--ink-2); margin-top:12px; line-height:1.5; }}
  .callout {{ border-left:3px solid; padding:16px 20px; border-radius:0 3px 3px 0;
              margin:1.4em 0; font-size:15.5px; }}
  .c-warn {{ border-color:var(--warn); background:var(--warn-soft); }}
  .c-bad  {{ border-color:var(--bad);  background:var(--bad-soft); }}
  .c-good {{ border-color:var(--good); background:var(--good-soft); }}
  .callout .lab {{ font-family:var(--mono); font-size:11px; letter-spacing:.1em;
                   text-transform:uppercase; display:block; margin-bottom:5px; }}
  .c-warn .lab {{ color:var(--warn); }} .c-bad .lab {{ color:var(--bad); }}
  .c-good .lab {{ color:var(--good); }}
  .grid {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
           margin:1.4em 0; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:3px;
           padding:16px 18px; }}
  .stat .k {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.09em;
              text-transform:uppercase; color:var(--ink-3); }}
  .stat .v {{ font-family:var(--serif); font-size:29px; line-height:1.15; margin-top:5px;
              font-variant-numeric:tabular-nums; }}
  .stat .n {{ font-size:13px; color:var(--ink-2); margin-top:3px; }}
  ol.steps {{ list-style:none; counter-reset:s; padding:0; margin:1.4em 0; display:grid; gap:12px; }}
  ol.steps li {{ counter-increment:s; background:var(--panel); border:1px solid var(--line);
                 border-radius:3px; padding:16px 20px 16px 56px; position:relative; }}
  ol.steps li::before {{ content:counter(s); position:absolute; left:20px; top:16px;
        font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:600; }}
  ol.steps b {{ display:block; margin-bottom:3px; }}
  code {{ font-family:var(--mono); font-size:.87em; background:var(--accent-soft);
          padding:1px 5px; border-radius:2px; }}
  .files {{ font-family:var(--mono); font-size:12.5px; color:var(--ink-2);
            background:var(--panel); border:1px solid var(--line); border-radius:3px;
            padding:16px 20px; overflow-x:auto; white-space:pre; line-height:1.75; }}
  hr.rule {{ border:none; border-top:1px solid var(--line); margin:0; }}
</style>

<div class="wrap">
<header class="top">
  <div class="eyebrow">CLIRB · closed-loop behavioural template matching</div>
  <h1>Are the same behaviours rewarded differently in different parts of the arena?</h1>
  <p class="sub">Yes — in two of three animals, and not because of anything a keypoint corrector can fix.
     A full investigation of the tracking confound, plus a corrector rebuilt on fine-tuned SLEAP output.</p>
  <div class="meta"><span>R4 · R5 · R6</span><span>151 sessions · 16,315 rears</span>
    <span>136 dual-modality sessions</span><span>2026-08-05 → 08-06</span></div>
</header>

<section>
  <div class="snum">The answer</div>
  <h2>Same behaviour, different place, different reward</h2>
  <div class="verdict">
    <p class="lede">Holding true behaviour fixed — each rear matched to its nearest twins in
      60-dimensional DANNCE trajectory space — match probability still varies around the arena
      perimeter by <b>0.75 in R5</b> and <b>0.37 in R6</b> (both p&lt;0.0001), with rears defined
      entirely from DANNCE so event selection cannot be contaminated by SLEAP.</p>
    <p>R4's effect (0.28) does <b>not</b> survive that test (p=0.075). The confound is a
      position-dependent <em>bias</em>, not a tracking failure: reprojection residual, confidence and
      FOV margin are all flat across the arena. It is driven by SLEAP 2D keypoint error that varies
      with <b>viewing aspect</b>, and it is a property of the rig — μ(θ) has the same shape in all
      three animals (r = 0.87–0.95).</p>
  </div>
  <div class="grid">
    <div class="stat"><div class="k">R5 · behaviour-matched</div><div class="v">0.75</div>
      <div class="n">spread in P(match) across the perimeter</div></div>
    <div class="stat"><div class="k">Aspect vs calibration</div><div class="v">1.5–1.8×</div>
      <div class="n">aspect-dependent error exceeds the image-position field</div></div>
    <div class="stat"><div class="k">μ(θ) cross-rat agreement</div><div class="v">r = 0.87–0.95</div>
      <div class="n">the bias belongs to the rig, not the animal</div></div>
    <div class="stat"><div class="k">Correctable at keypoint level</div><div class="v">1.6 mm</div>
      <div class="n">systematic part of a ~10 mm mostly-random residual</div></div>
  </div>
</section>

<hr class="rule">

<section>
  <div class="snum">Part one · the confound</div>
  <h2>The arena is a ring, and reward is not uniform around it</h2>
  <div class="measure">
    <p>The arena fits a circle of radius 332 mm centred at (−66, −98), and the animals are
      effectively always against the wall — 96% of rears occur at r &gt; 250 mm. That makes perimeter
      angle θ, not a 2-D grid, the coordinate that matters.</p>
  </div>
  {img(f'{SP}/F1_arena_occupancy.png', 'Rear locations per rat',
       'Every rear, per animal. R5&rsquo;s dense band at +45° is the localization that prompted the question. '
       'The red arc is the 200 mm port gate, where reward is impossible by design and which is excluded throughout.')}
  {img(f'{SP}/F2_pmatch_by_sector.png', 'P(match) by sector',
       'Reward probability varies about twofold around the perimeter. Bars are 95% session-clustered bootstrap CIs.')}
  {img(f'{SP}/F3_nn_matched_delta.png', 'Behaviour-matched delta',
       'The main result. Each rear is compared with its nearest twins in DANNCE trajectory space located in '
       'other sectors, so behaviour is held fixed. For R5 the same rear is ~31 points more likely to be rewarded '
       'at +45° and ~41 points less likely at +15° — adjacent sectors.')}
  <div class="callout c-warn"><span class="lab">Read this as a contrast, not two independent numbers</span>
    These deltas compare each rear against a mixture dominated by whichever sectors are densely
    occupied — for R5, one sector holds 54% of compared rears. The conservative, independently
    estimated figure is the within-quality-bin sector SD: <b>0.083 / 0.148 / 0.060</b> for R4/R5/R6.</div>
</section>

<section>
  <h2>It is not explained by tracking quality</h2>
  {img(f'{SP}/F4_quality_vs_theta.png', 'Quality metrics vs theta',
       'No truth-free quality metric tracks the reward gradient (all p &gt; 0.18). Panels state their full span '
       'because the axes are truncated. These are the corrected v2 numbers, computed after fixing a '
       'frame-basis indexing error described below.')}
  {img(f'{SP}/F5_residual_field.png', 'Residual field mu(theta)',
       'The SLEAP−DANNCE PC0 residual has the same shape in all three animals, differing only by a per-rat '
       'constant (which is a PC-basis artifact, not biology). After removing that offset, cross-rat SD at a given '
       'θ is 1.11 vs a within-rat spread of 4.19 — the spatial pattern is ~4× larger than between-animal disagreement.')}
  <div class="measure">
    <p>Per-rear analysis sharpened this. Tracking quality genuinely predicts the SLEAP–DANNCE
      discrepancy (reprojection residual ρ = +0.25 / +0.38 / +0.38, all p &lt; 1e-31), so the metrics
      are real. But conditioning on all six quality metrics makes the sector effect <em>larger</em>
      (+37% / +64% / +10%), not smaller. Quality is not the mediator.</p>
  </div>
  {img(f'{SP}/F7_cluster_sector_ami.png', 'Template builder clusters vs sector',
       'The template builder&rsquo;s own KMeans clusters segregate by arena sector in 15 of 16 runs. '
       'The mechanism is real — "largest cluster" can mean "most-occupied sector" — though the build sector '
       'does not predict which sector matches best (rank 4.33 vs 4.08 by chance).')}
  {img(f'{SP}/F6_behaviour.png', 'Behavioural adaptation',
       'Only R5 adapts: its rear-location concentration nearly doubles over training, and it stays put after '
       'reward and moves after a miss (p &lt; 1e-4). R4 and R6 show neither.')}
</section>

<hr class="rule">

<section>
  <div class="snum">Part two · the corrector</div>
  <h2>Rebuilt on fine-tuned SLEAP output — and it does not help</h2>
  <div class="measure">
    <p>Because the SLEAP network was fine-tuned on 2026-07-21 but only deployed online on 07-29, the
      stored 2D comes from the <em>old</em> model for all but 16 of 163 sessions. So 2D was re-inferred
      from video with the fine-tuned model, then re-triangulated with the 05_04 calibration and paired
      with DANNCE — a pipeline that consumes exactly what the rig has at inference time.</p>
  </div>
  {img(f'{CV}/V3_sampling.png', 'Balanced sampling',
       'Selection is stratified by rat × training-time third × arena sector × posture, with rears enriched. '
       'Sector occupancy is flattened from a ~20× natural range to ~2.6×, so the corrector cannot simply learn '
       'the occupancy prior.')}
  <div class="grid">
    <div class="stat"><div class="k">Frames re-inferred</div><div class="v">132,912</div>
      <div class="n">2,769 contiguous 48-frame segments, 17 sessions</div></div>
    <div class="stat"><div class="k">Paired &amp; usable</div><div class="v">36,674</div>
      <div class="n">13 sessions passing the alignment gate</div></div>
    <div class="stat"><div class="k">Fine-tuned vs DANNCE</div><div class="v">8.6 mm</div>
      <div class="n">Procrustes residual, vs 14.5 mm for the deployed pose</div></div>
    <div class="stat"><div class="k">Systematic component</div><div class="v">1.6 mm</div>
      <div class="n">on clean frames; sector spread only 2.6 mm</div></div>
  </div>
  {img(f'{CV}/V1_tier1_keypoint_error.png', 'Tier 1 keypoint error',
       'All three architectures are marginally WORSE than uncorrected, at essentially every keypoint. '
       'This is not overfitting — they fail to improve on the training set too (10.04 → 10.22 mm).')}
  <div class="callout c-bad"><span class="lab">Why there is nothing to learn</span>
    On clean frames the raw-triangulation → DANNCE residual is ~10 mm with a per-frame SD of 13.9 mm,
    while the systematic part is only <b>1.6 mm</b> (per-sector spread 2.6 mm). Correcting with a
    global or per-sector median offset changes held-out error by <b>0.0–0.1%</b>. The fine-tuned SLEAP
    model already agrees with DANNCE to within mostly-random error.</div>
  <h3>The raw triangulation is the real problem</h3>
  <div class="measure">
    <p>The unregularised DLT fails catastrophically on a sizeable minority of frames: 16% have a pose
      radius above 500 mm against a true rat radius of ~150 mm, and the worst |SLEAP−DANNCE| reaches
      2.4 × 10⁶ mm. That is exactly what the online bone-length bundle adjustment
      (<code>correct_triangulations</code>) exists to fix — and it does so better than any corrector here.</p>
  </div>
  <div class="tscroll"><table>
    <thead><tr><th>pose source</th><th>median |R|</th><th>|median R|</th><th>per-sector spread</th></tr></thead>
    <tbody>
      <tr><td>Deployed — old model + bone regularisation</td><td class="num">9.77 mm</td>
          <td class="num">1.30 mm</td><td class="num hi">1.60 mm</td></tr>
      <tr><td>New — fine-tuned model + raw triangulation</td><td class="num">10.11 mm</td>
          <td class="num">1.59 mm</td><td class="num">2.63 mm</td></tr>
    </tbody></table></div>
  {img(f'{CV}/V2_tier2_spatial.png', 'Tier 2 spatial metrics',
       'Tier 2, both template options. The one robust signal is μ(θ) range: the deployed pipeline is far better '
       'than raw triangulation, with the correctors intermediate. Match-based metrics are underpowered — see caveat.')}
  <div class="tscroll"><table>
    <thead><tr><th>source</th><th>P(match)</th><th>sector spread</th><th>μ(θ) range</th></tr></thead>
    <tbody>
      <tr><td>deployed (bone-regularised)</td><td class="num">{t2row('deployed','existing','p_match')}</td>
        <td class="num">{t2row('deployed','existing','sector_spread')}</td>
        <td class="num hi">{t2row('deployed','existing','mu_range')}</td></tr>
      <tr><td>uncorrected raw triangulation</td><td class="num">{t2row('uncorrected','existing','p_match')}</td>
        <td class="num">{t2row('uncorrected','existing','sector_spread')}</td>
        <td class="num">{t2row('uncorrected','existing','mu_range')}</td></tr>
      <tr><td>corrector — temporal</td><td class="num">{t2row('temporal','existing','p_match')}</td>
        <td class="num">{t2row('temporal','existing','sector_spread')}</td>
        <td class="num">{t2row('temporal','existing','mu_range')}</td></tr>
      <tr><td>corrector — velacc</td><td class="num">{t2row('velacc','existing','p_match')}</td>
        <td class="num">{t2row('velacc','existing','sector_spread')}</td>
        <td class="num">{t2row('velacc','existing','mu_range')}</td></tr>
      <tr><td>corrector — aspect-aware</td><td class="num">{t2row('aspect','existing','p_match')}</td>
        <td class="num">{t2row('aspect','existing','sector_spread')}</td>
        <td class="num">{t2row('aspect','existing','mu_range')}</td></tr>
    </tbody></table></div>
  <div class="callout c-warn"><span class="lab">Tier 2 is underpowered — do not over-read it</span>
    708 segments yield only ~10–35 matches per source with the existing template, far too few for
    reliable sector statistics, and the rebuilt sector-balanced template is so permissive that
    P(match) saturates near 0.91 for every source. The μ(θ) column is the only column I would quote.
    A proper version needs whole sessions, not 48-frame segments.</div>
</section>

<hr class="rule">

<section>
  <div class="snum">Follow-up · mechanism</div>
  <h2>What causes the viewpoint-specific error — and what it does not explain</h2>
  <div class="measure">
    <p>Four candidate mechanisms were tested against the DANNCE reference reprojected into each SLEAP
      camera. Two survive, two are cleanly out.</p>
  </div>
  <div class="tscroll"><table>
    <thead><tr><th>account</th><th>signature</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>Foreshortening</td><td>error peaks end-on</td>
          <td><b>supported</b> — 12.5&ndash;13.6 px facing the camera vs 10.1&ndash;11.4 px oblique;
              confidence 0.61 vs 0.70</td></tr>
      <tr><td>Lateral self-occlusion</td><td>far side of a bilateral pair worse</td>
          <td><b>supported</b> — far/near ratio 1.26 median (ankle 1.44, foot 1.36, knee 1.36);
              ears show none (0.99)</td></tr>
      <tr><td>Left/right swaps</td><td>swapping would reduce error</td>
          <td><b>rejected</b> — swap better in 1.9% of frames; swapping triples the error</td></tr>
      <tr><td>Aspect-dependent labels</td><td>net confidently wrong</td>
          <td><b>rejected</b> — &rho;(confidence, error) = &minus;0.32 for all 23 keypoints</td></tr>
    </tbody></table></div>
  <div class="callout c-good"><span class="lab">This explains the corrector failure</span>
    The network is <em>appropriately uncertain</em> where it errs. That makes the error missing
    information, not removable bias &mdash; and you cannot subtract away information that is not in
    the image. It is the same fact viewed twice.</div>
  <div class="callout c-bad"><span class="lab">But the causal chain does NOT close</span>
    Pushing the measured 2D error field through the actual triangulation geometry predicts only
    <b>20&ndash;25% of the amplitude</b> of the measured &mu;(&theta;) (predicted range 3.0&ndash;3.6
    vs measured 8.3&ndash;15.4; pooled &rho; = +0.30, p = 0.08). The sign is right, the magnitude is
    3&ndash;4&times; short, and removing my confidence/outlier filters made it worse &mdash; so it is
    not a censoring artifact. <b>Aspect-dependent 2D error is real but is not the main driver of the
    spatial reward bias.</b></div>
  <div class="measure">
    <p>The most likely gap: the field was binned by viewing aspect alone, but 2D error must also
      depend on <em>posture</em> &mdash; a full rear and a crouch present entirely different
      self-occlusion at the same aspect &mdash; and posture varies systematically by sector
      (apex height spans 2.2&ndash;6.9 in R4). Conditioning the error field on pose as well as aspect
      is the next iteration.</p>
    <p>One anomaly worth a physical check: <b>EarR has twice the aspect sensitivity of EarL</b>
      (10.2 vs 4.9 px) and uniquely shows no near/far effect. A symmetric mechanism cannot produce a
      left/right asymmetry, so something asymmetric &mdash; headcap, tether, lighting &mdash; is
      likely, and it is the largest single keypoint contributor.</p>
  </div>
</section>

<hr class="rule">

<section>
  <div class="snum">Corrections</div>
  <h2>Things I got wrong along the way</h2>
  <div class="callout c-bad"><span class="lab">Retracted — the "data integrity bug"</span>
    I reported that <code>sleap_keys_2D.npy</code> did not correspond to <code>triang_keys_3D.npy</code>
    in any of 152 sessions and called it a pipeline bug. <b>That was my indexing error.</b> The 2D is on
    the processed-frame basis, the 3D on the cam0 acquisition basis; bridging them via
    <code>frame_mapping.csv</code> gives 17.4 px reprojection and r = +0.988, versus 253 px and +0.23
    when misaligned. The files were always fine.</div>
  <div class="tscroll"><table>
    <thead><tr><th>claim</th><th>corrected to</th><th>cause</th></tr></thead>
    <tbody>
      <tr><td>Cluster×sector significant in 16/16 runs</td><td>15/16</td><td>rounding a p=0.086 run away; the figure exposed it</td></tr>
      <tr><td>NN-matched effect “all p&lt;0.0001”</td><td>R4 p = 0.005</td><td>original null did not re-run the whole pipeline</td></tr>
      <tr><td>§4 quality metrics are void</td><td>valid, recomputed</td><td>followed from the retracted bug above</td></tr>
      <tr><td>Median cameras with conf ≥ 0.5 is 1</td><td>2</td><td>misaligned frames</td></tr>
      <tr><td>Camera 2 mean confidence 0.31</td><td>0.348 (vs 0.549/0.547)</td><td>misaligned frames</td></tr>
      <tr><td>07_20 calibration is “more correct”</td><td>05_04 is metric (scale 1.0006)</td><td>a degenerate Procrustes fit skewed the first estimate</td></tr>
    </tbody></table></div>
  <div class="callout c-good"><span class="lab">Also resolved</span>
    The two calibrations differ by a <b>180° yaw flip of the world origin</b> plus 0.37% scale — the
    cameras never moved (baselines agree to 4.6–11.2 mm out of 1.9 m). The flip is the signature of a
    checkerboard origin detected at the opposite corner; the scale error explains why 07_20 reprojects
    slightly worse. Use 05_04 everywhere.</div>
</section>

<hr class="rule">

<section>
  <div class="snum">What to do next</div>
  <h2>Recommendations, most actionable first</h2>
  <ol class="steps">
    <li><b>Stop pursuing keypoint-level correction.</b> The systematic component is 1.6 mm inside a
      ~10 mm mostly-random residual, and three architectures plus an oracle offset all fail to beat
      identity. The fine-tuned SLEAP model has already closed this gap. Effort here has a low ceiling.</li>
    <li><b>Fix the template builder — the one clearly actionable lever.</b> In
      <code>refine_templates_mult_session.py</code>, keep the largest cluster's mean as the template but
      compute <code>max_dev</code> bounds over a <em>sector-balanced</em> sample. Current bounds are
      1.5–3.0× tighter than that purely because the pool is spatially lopsided.</li>
    <li><b>Re-run Tier 2 on whole sessions.</b> The 48-frame segments were chosen to fit an SMB I/O
      budget of ~5 MB/s, and they cost the match-based statistics their power. Whole sessions from the
      deployed pose would settle whether the reward gradient moves at all.</li>
    <li><b>Attack the aspect-dependent 2D bias at source.</b> The error is keypoint-specific and varies
      with viewing aspect, concentrated in EarR/EarL, shoulders, feet and TailBase. Balancing SLEAP's
      training labels by viewing aspect targets the cause rather than patching the output.</li>
    <li><b>Report sector-stratified in the behavioural notebooks.</b> Cross-session match-rate trends
      partly track where the animal stood; R5's location concentration nearly doubles over training,
      so its learning curve is confounded.</li>
    <li><b>Consider a fourth SLEAP camera.</b> With three, the median frame has only two confident
      views, which is what leaves the reconstruction sensitive to viewing aspect at all.</li>
  </ol>
</section>

<section>
  <div class="snum">Artifacts</div>
  <h2>Where everything lives</h2>
  <div class="files">CLIRB_analyses/
├─ SPATIAL_CONFOUND_PLAN.md        the analysis plan
├─ SPATIAL_CONFOUND_RESULTS.md     full findings, caveats, retractions
├─ CORRECTOR_V2_PLAN.md            corrector plan + calibration forensics
├─ clirb/                          extracted package (events, geometry, quality, dual)
│   └─ events.py                   THE single copy of rear detector / matcher / gating
├─ scripts/
│   ├─ build_event_table.py        rears.csv  (16,315 rears)
│   ├─ stage1..stage10_*.py        the spatial-confound analyses
│   └─ v2_01..v2_07_*.py           corrector v2 pipeline
└─ results/
    ├─ spatial_confound/           tables, residual fields, NN matching
    ├─ corrector_v2/               segments, dataset, checkpoints, tier2
    └─ figures/                    every figure on this page</div>
  <p class="dim" style="font-size:13.5px; margin-top:18px;">Not yet done: the three behavioural
    notebooks still carry their own copies of the rear detector rather than importing
    <code>clirb.events</code>. The module reproduces the documented invariant exactly
    (R6/2026_07_08_1 → 60 rears / 20 matched), so repointing them is mechanical.</p>
</section>
</div>
"""

open(OUT, 'w').write(HTML)
print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')
