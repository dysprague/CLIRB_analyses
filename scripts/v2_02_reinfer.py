"""Corrector v2, step 2 — re-run SLEAP inference with the FINE-TUNED model.

The stored sleap_keys_2D comes from the OLD model for all but 16 sessions, so the
2D must be re-inferred from video for the selected frames.

Constraints established during planning:
  * sleap-track's --frames is broken in this build -> write selected frames to a clip
  * lossy re-encode shifts predictions (p95 10 px) -> clips must be LOSSLESS FFV1,
    which is bit-identical to a PNG directory and 3.6x faster to infer
  * random seeks cost 5.5 s, sequential decode runs at 500 FPS -> never seek
  * per-camera frame indices differ by up to 2 -> use cam{c}_frame for camera c

Writes results/corrector_v2/keys2d/<rat>__<session>.npz with
  k        (n,)            processed-frame indices
  keys2d   (n, 3, 23, 3)   new (x, y, score) from the fine-tuned model
"""
import os, sys, glob, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
MODEL = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/models/260721_165454.single_instance.n=9975'
SLEAP = '/home/yutaka-sprague/.local/bin/sleap-track'
SLEAP_PY = '/home/yutaka-sprague/.local/share/uv/tools/sleap/bin/python'
OUT = 'results/corrector_v2/keys2d'
TMP = '/tmp/claude-1000/-home-yutaka-sprague/0709c2d8-bcbc-4d7c-b833-a6ea2773aa11/scratchpad/v2tmp'
CHUNK = 500
os.makedirs(OUT, exist_ok=True)

_READ_SLP = r'''
import sys, numpy as np, sleap_io as sio
L = sio.load_slp(sys.argv[1])
n = len(L.labeled_frames)
out = np.full((int(sys.argv[3]), 23, 3), np.nan, dtype=np.float32)
for lf in L.labeled_frames:
    if lf.instances:
        i = int(lf.frame_idx)
        if 0 <= i < out.shape[0]:
            out[i] = lf.instances[0].numpy(scores=True)
np.save(sys.argv[2], out)
'''


def infer_clip(clip, n_frames):
    """Run the fine-tuned model on one clip -> (n_frames, 23, 3)."""
    slp = clip + '.slp'; npy = clip + '.npy'
    r = subprocess.run([SLEAP, '-m', MODEL, '-o', slp, clip],
                       capture_output=True, text=True, timeout=3600)
    if not os.path.isfile(slp):
        print(f'    inference FAILED for {clip}: {r.stderr[-300:]}')
        return np.full((n_frames, 23, 3), np.nan, np.float32)
    subprocess.run([SLEAP_PY, '-c', _READ_SLP, slp, npy, str(n_frames)],
                   capture_output=True, text=True, timeout=600)
    out = np.load(npy) if os.path.isfile(npy) else np.full((n_frames, 23, 3), np.nan, np.float32)
    for f in (slp, npy):
        if os.path.isfile(f):
            os.remove(f)
    return out


def extract_camera(rat, session, cam, want_frames, tmpdir):
    """DECODE ONLY (I/O bound) -> write lossless chunks. Returns [(path, [frame_idx])].

    Separated from inference so the three cameras can decode CONCURRENTLY (the
    SMB share sustains ~5 MB/s per stream but ~8 MB/s aggregate) while the GPU
    work stays serialised.
    """
    vid = f'{DR}/{rat}/{session}/sleap/Camera{cam}/0.mp4'
    want = np.sort(np.unique(want_frames))
    wset = set(int(x) for x in want)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f'    cannot open {vid}', flush=True)
        return []
    last = int(want.max())
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out, writer, buf_idx, chunk_i = [], None, [], 0
    i = 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wset:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(f'{tmpdir}/c{cam}_{chunk_i:04d}.avi',
                                         fourcc, 20, (w, h))
            writer.write(frame)
            buf_idx.append(i)
            if len(buf_idx) >= CHUNK:
                writer.release()
                out.append((f'{tmpdir}/c{cam}_{chunk_i:04d}.avi', buf_idx))
                chunk_i += 1; writer = None; buf_idx = []
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
        out.append((f'{tmpdir}/c{cam}_{chunk_i:04d}.avi', buf_idx))
    return out


def process_session(rat, session, segs):
    op = f'{OUT}/{rat}__{session}.npz'
    if os.path.isfile(op):
        return 'hit'
    fm = pd.read_csv(f'{DR}/{rat}/{session}/sleap/frame_mapping.csv', skipinitialspace=True)
    cam = {c: fm[f'cam{c}_frame'].to_numpy(int) for c in range(3)}
    K = len(fm)
    ks = set()
    for _, r in segs.iterrows():
        ks.update(range(int(r.k_start), min(int(r.k_end), K)))
    ks = np.array(sorted(ks), dtype=int)
    tmpdir = f'{TMP}/{rat}_{session}'
    os.makedirs(tmpdir, exist_ok=True)
    try:
        keys = np.full((len(ks), 3, 23, 3), np.nan, dtype=np.float32)
        # --- phase A: decode all three cameras CONCURRENTLY (I/O bound)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(extract_camera, rat, session, c, cam[c][ks], tmpdir): c
                    for c in range(3)}
            chunks = {futs[f]: f.result() for f in as_completed(futs)}
        n_ch = sum(len(v) for v in chunks.values())
        print(f'    decode: {n_ch} chunks in {time.time()-t0:.0f}s', flush=True)
        # --- phase B: infer serially on the single GPU
        t1 = time.time()
        for c in range(3):
            res = {}
            for path, idxs in chunks.get(c, []):
                kp = infer_clip(path, len(idxs))
                for j, fi in enumerate(idxs):
                    res[fi] = kp[j]
                if os.path.isfile(path):
                    os.remove(path)
            for j, k in enumerate(ks):
                v = res.get(int(cam[c][k]))
                if v is not None:
                    keys[j, c] = v
        print(f'    infer: {time.time()-t1:.0f}s', flush=True)
        np.savez_compressed(op, k=ks, keys2d=keys)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 'built'


def main():
    segs = pd.read_csv('results/corrector_v2/segments.csv')
    todo = list(segs.groupby(['rat', 'session']))
    # Interleave rats and time-thirds so that stopping early still leaves a
    # BALANCED subset rather than all of R4 and none of R6.
    order = {}
    for (rat, session), g in todo:
        order[(rat, session)] = (int(g.third.iloc[0]), session, rat)
    by_rat = {r: [k for k in order if k[0] == r] for r in ['R4', 'R5', 'R6']}
    for r in by_rat:
        by_rat[r].sort(key=lambda k: order[k])
    inter = []
    for i in range(max(len(v) for v in by_rat.values())):
        for r in ['R4', 'R5', 'R6']:
            if i < len(by_rat[r]):
                inter.append(by_rat[r][i])
    lut = dict(todo)
    todo = [(k, lut[k]) for k in inter]
    print(f'{len(todo)} sessions to process (rat-interleaved)', flush=True)
    for n, ((rat, session), g) in enumerate(todo):
        t0 = time.time()
        st = process_session(rat, session, g)
        print(f'[{n+1}/{len(todo)}] {rat}/{session}: {st} '
              f'({len(g)} segs, {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
