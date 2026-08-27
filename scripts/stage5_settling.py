"""Stage 5 — natural settling vs induced bias, plus the template-swap tests."""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from clirb.spatial.geometry import sector_center

OUT = 'results/spatial_confound'
SECTOR_W = 30
GATE_OFF = ['2026_07_21_1', '2026_07_22_1', '2026_07_23_1']


def circ_concentration(theta_deg):
    """Mean resultant length R in [0,1]; 1 = perfectly concentrated."""
    a = np.radians(np.asarray(theta_deg))
    return float(np.abs(np.exp(1j * a).mean()))


def modal_sector(g):
    return int(g.sector_30.value_counts().idxmax())


def main():
    df = pd.read_csv(f'{OUT}/rears.csv')
    res = {}

    # ------------------------------------------------ 5.4/5.5 gate zone
    print('=' * 78); print('5.4/5.5  PORT-GATE ZONE — does the rat learn it?'); print('=' * 78)
    gz = []
    for (rat, sess), g in df.groupby(['rat', 'session']):
        mpd = g.min_port_distance.iloc[0]
        # use a FIXED 200 mm criterion so gate-off sessions are comparable
        frac_in = float((g.port_dist < 200).mean())
        gz.append(dict(rat=rat, session=sess, session_idx=g.session_idx.iloc[0],
                       mpd=mpd, frac_in200=frac_in, n=len(g),
                       gate_off=sess in GATE_OFF))
    gz = pd.DataFrame(gz)
    for rat, g in gz.groupby('rat'):
        on = g[~g.gate_off]
        r, p = stats.spearmanr(on.session_idx, on.frac_in200)
        pre = g[(~g.gate_off) & (g.session_idx < g[g.gate_off].session_idx.min())]
        off = g[g.gate_off]
        post = g[(~g.gate_off) & (g.session_idx > g[g.gate_off].session_idx.max())]
        print(f'\n{rat}: frac of rears within 200mm of port')
        print(f'   trend across gate-ON sessions: Spearman r={r:+.3f} p={p:.4f} (n={len(on)})')
        print(f'   pre-gate-off  mean={pre.frac_in200.mean():.3f} (n={len(pre)} sessions)')
        print(f'   GATE OFF      mean={off.frac_in200.mean():.3f} (n={len(off)})')
        print(f'   post-gate-off mean={post.frac_in200.mean():.3f} (n={len(post)})')
        res[f'{rat}_gatezone_trend_r'] = float(r); res[f'{rat}_gatezone_trend_p'] = float(p)
        res[f'{rat}_gateoff_delta'] = float(off.frac_in200.mean() - pre.frac_in200.mean())
    gz.to_csv(f'{OUT}/gate_zone_by_session.csv', index=False)

    # ------------------------------------------------ 5.1 occupancy drift
    print('\n' + '=' * 78); print('5.1  OCCUPANCY DRIFT'); print('=' * 78)
    occ = []
    for (rat, sess), g in df.groupby(['rat', 'session']):
        occ.append(dict(rat=rat, session=sess, session_idx=g.session_idx.iloc[0],
                        conc=circ_concentration(g.theta), modal=modal_sector(g),
                        n=len(g), p_match=g.matched.mean(),
                        template=g.template_sha.iloc[0]))
    occ = pd.DataFrame(occ).sort_values(['rat', 'session_idx'])
    occ.to_csv(f'{OUT}/occupancy_by_session.csv', index=False)
    for rat, g in occ.groupby('rat'):
        r, p = stats.spearmanr(g.session_idx, g.conc)
        early = g[g.session_idx < g.session_idx.median()]
        late = g[g.session_idx >= g.session_idx.median()]
        print(f'\n{rat}: rear-location concentration (circular R)')
        print(f'   early mean R={early.conc.mean():.3f}  late mean R={late.conc.mean():.3f}'
              f'   Spearman r={r:+.3f} p={p:.4f}')
        print(f'   modal sector: early={early.modal.mode().iloc[0]} '
              f'({sector_center(early.modal.mode().iloc[0], SECTOR_W):+.0f}deg)  '
              f'late={late.modal.mode().iloc[0]} '
              f'({sector_center(late.modal.mode().iloc[0], SECTOR_W):+.0f}deg)')
        res[f'{rat}_conc_trend_r'] = float(r); res[f'{rat}_conc_trend_p'] = float(p)
        res[f'{rat}_conc_early'] = float(early.conc.mean())
        res[f'{rat}_conc_late'] = float(late.conc.mean())

    # ------------------------------------------------ 5.3 win-stay / lose-shift
    print('\n' + '=' * 78); print('5.3  SPATIAL WIN-STAY / LOSE-SHIFT'); print('=' * 78)
    rows = []
    for (rat, sess), g in df.groupby(['rat', 'session']):
        g = g.sort_values('start_frame')
        sec = g.sector_30.to_numpy(); m = g.matched.to_numpy()
        if len(g) < 10:
            continue
        moved = (sec[1:] != sec[:-1])
        rows.append(dict(rat=rat, session=sess,
                         p_move_after_match=float(moved[m[:-1]].mean()) if m[:-1].any() else np.nan,
                         p_move_after_miss=float(moved[~m[:-1]].mean()) if (~m[:-1]).any() else np.nan,
                         n=len(g)))
    ws = pd.DataFrame(rows)
    for rat, g in ws.groupby('rat'):
        a = g.p_move_after_match.dropna(); b = g.p_move_after_miss.dropna()
        t, p = stats.wilcoxon(g.dropna().p_move_after_match, g.dropna().p_move_after_miss) \
            if len(g.dropna()) > 5 else (np.nan, np.nan)
        print(f'{rat}: P(change sector | previous rear REWARDED)={a.mean():.3f}  '
              f'| NOT rewarded={b.mean():.3f}   Wilcoxon p={p:.4f} (n={len(g.dropna())} sessions)')
        res[f'{rat}_move_after_match'] = float(a.mean())
        res[f'{rat}_move_after_miss'] = float(b.mean())
        res[f'{rat}_winstay_p'] = float(p) if np.isfinite(p) else None
    ws.to_csv(f'{OUT}/winstay_by_session.csv', index=False)

    # ------------------------------------------------ template swaps
    print('\n' + '=' * 78); print('8.4  TEMPLATE SWAPS — occupancy + P(match) around a change')
    print('=' * 78)
    swap_rows = []
    for rat, g in occ.groupby('rat'):
        g = g.sort_values('session_idx').reset_index(drop=True)
        ch = np.where(g.template.values[1:] != g.template.values[:-1])[0] + 1
        for c in ch:
            pre = g.iloc[max(0, c - 3):c]
            post = g.iloc[c:c + 3]
            if len(pre) < 2 or len(post) < 2:
                continue
            swap_rows.append(dict(
                rat=rat, at_session=g.session.iloc[c],
                pre_modal=int(pre.modal.mode().iloc[0]), post_modal=int(post.modal.mode().iloc[0]),
                pre_conc=pre.conc.mean(), post_conc=post.conc.mean(),
                pre_p=pre.p_match.mean(), post_p=post.p_match.mean()))
    sw = pd.DataFrame(swap_rows)
    if not sw.empty:
        sw['modal_moved'] = sw.pre_modal != sw.post_modal
        sw['d_p'] = sw.post_p - sw.pre_p
        sw['d_conc'] = sw.post_conc - sw.pre_conc
        print(sw.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        print(f'\n  modal sector moved at {sw.modal_moved.sum()}/{len(sw)} template swaps')
        print(f'  mean delta P(match) across swaps = {sw.d_p.mean():+.3f}')
        print(f'  mean delta concentration        = {sw.d_conc.mean():+.3f}')
        res['n_swaps'] = int(len(sw))
        res['swaps_modal_moved'] = int(sw.modal_moved.sum())
        res['swap_mean_dP'] = float(sw.d_p.mean())
        sw.to_csv(f'{OUT}/template_swaps.csv', index=False)

    # ------------------------------------------------ 5.6 practice vs geometry
    print('\n' + '=' * 78); print('5.6  PRACTICE — does align_mse in a sector improve with use?')
    print('=' * 78)
    for rat, g in df.groupby('rat'):
        g = g.sort_values(['session_idx', 'start_frame']).copy()
        g['cum_in_sector'] = g.groupby('sector_30').cumcount()
        sub = g[g.cum_in_sector < 3000]
        r, p = stats.spearmanr(sub.cum_in_sector, sub.align_mse)
        print(f'{rat}: Spearman(cumulative rears in sector, align_mse) '
              f'r={r:+.3f} p={p:.2e} (n={len(sub)})')
        res[f'{rat}_practice_r'] = float(r)

    with open(f'{OUT}/stage5_results.json', 'w') as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f'\nwrote {OUT}/stage5_results.json')


if __name__ == '__main__':
    main()
