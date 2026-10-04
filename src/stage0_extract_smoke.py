"""Stage 0 engineering smoke: does the VadCLIP feature recipe run on ShanghaiTech frames?

    python src/stage0_extract_smoke.py --videos 01_0014 01_0015 --frames 32 --out-dir <scratch>

The cross-domain plan's Stage 1 needs CLIP ViT-B/16 features for datasets beyond
UCF/XD (SHT first, then UBnormal/MSAD). Local raw frames exist for SHT; this
script proves the extraction path end to end on a couple of videos before
anyone budgets a full extraction run:

  1. read frames from ``E:\\dataset\\shanghaitech\\testing\\frames\\<video>\\``;
  2. VadCLIP's crop.py preprocessing - resize to (340, 256), centre crop
     [16:240, 58:282] (type 0), which is what the official UCF/XD caches used;
  3. CLIP's own preprocess + ViT-B/16 encode_image;
  4. save a (T, 512) float32 array - the cache format every repo here loads.

It also measures CPU encoding speed so the full-extraction estimate in the
Stage 0 report is a measurement, not a guess. The 16-frame subsampling stride
of the official caches is a *list/GT alignment* convention (features at 1/16 of
GT frame rate); a real extraction must reproduce it together with the dataset's
own GT - out of scope for a smoke test, which encodes consecutive frames and
saves nothing into the repo.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip  # noqa: E402  (the vendored src/clip package - same import the repo uses;
#              do NOT put src/clip itself on sys.path, that breaks its relative imports)
import freq_text_options  # noqa: E402

SHT_TEST_FRAMES = os.path.join(freq_text_options.sht_root(), 'testing', 'frames')


def vadclip_center_crop(img_bgr):
    """crop.py's type-0 path: resize (340,256) -> [16:240, 58:282], BGR->RGB."""
    img = cv2.resize(img_bgr, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[16:240, 58:282, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', nargs='+', default=['01_0014', '01_0015'])
    ap.add_argument('--frames', type=int, default=32, help='frames per video to encode')
    ap.add_argument('--out-dir', required=True, help='scratch dir; nothing is written to the repo')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} (CPU is expected locally; CUDA is a bonus)', flush=True)
    model, preprocess = clip.load('ViT-B/16', device=device, jit=False)
    model.eval()

    for vid in args.videos:
        vdir = os.path.join(SHT_TEST_FRAMES, vid)
        if not os.path.isdir(vdir):
            print(f'  [{vid}] MISSING frame dir: {vdir}')
            continue
        names = sorted(f for f in os.listdir(vdir) if f.lower().endswith(('.png', '.jpg')))
        if not names:
            print(f'  [{vid}] no frames in {vdir}')
            continue
        take = names[:args.frames]
        feats = np.zeros((len(take), 512), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i, fn in enumerate(take):
                img = vadclip_center_crop(cv2.imread(os.path.join(vdir, fn)))
                if img is None:
                    raise OSError(f'unreadable frame: {os.path.join(vdir, fn)}')
                tensor = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
                feats[i] = model.encode_image(tensor).float().cpu().numpy()[0]
        dt = time.time() - t0
        out_path = os.path.join(args.out_dir, f'smoke_{vid}.npy')
        np.save(out_path, feats)
        print(f'  [{vid}] {len(names)} frames on disk, encoded {len(take)} -> '
              f'{feats.shape} saved {out_path}  ({dt:.1f}s = {len(take) / dt:.1f} fps)',
              flush=True)

    print("""
  PASS criteria: shapes are (n,512) float32; no unreadable frames; fps measured.
  Full SHT extraction budget = (train+test frames) / fps / 3600 h on this device;
  divide by ~5-10x for a GPU card. The 16-frame stride convention and the GT
  alignment belong to the real Stage-1 extraction, not to this smoke.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
