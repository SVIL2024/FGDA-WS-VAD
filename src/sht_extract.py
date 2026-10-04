"""Extract CLIP ViT-B/16 features for ShanghaiTech (stride 16, VadCLIP recipe).

    python src/sht_extract.py --split both            # resume-by-skip
    python src/sht_extract.py --split test

Output: ``E:\\dataset\\SHTClipFeatures\\{train,test}\\<video>.npy``, one
``(T, 512)`` float32 array per video with ``T = ceil(n_frames / 16)`` (frames
sampled at stride 16 - the 1-feature-per-16-frames convention every repo here
aligns GT against, GT_FRAME_REPEAT=16).

Preprocessing mirrors ``referCode/VadCLIP-main/src/crop.py`` type 0 (resize
340x256, centre crop [16:240, 58:282]) - the recipe the official UCF/XD caches
used, verified by the Stage-0 smoke. Resume-by-skip: an existing output npy is
never rewritten, so the script can be interrupted and relaunched freely.
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
import clip  # noqa: E402  (vendored src/clip package)
import freq_text_options  # noqa: E402

SHT_ROOT = freq_text_options.sht_root()
OUT_ROOT = freq_text_options.sht_feat_root()
SPLITS = {'train': os.path.join(SHT_ROOT, 'training', 'frames'),
          'test': os.path.join(SHT_ROOT, 'testing', 'frames')}
STRIDE = 16


def vadclip_center_crop(img_bgr):
    img = cv2.resize(img_bgr, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[16:240, 58:282, :]


def extract_video(model, preprocess, device, vdir, out_path):
    names = sorted(f for f in os.listdir(vdir) if f.lower().endswith(('.png', '.jpg')))
    picks = names[::STRIDE]
    feats = np.zeros((len(picks), 512), dtype=np.float32)
    with torch.no_grad():
        for i, fn in enumerate(picks):
            img = vadclip_center_crop(cv2.imread(os.path.join(vdir, fn)))
            if img is None:
                raise OSError(f'unreadable frame: {os.path.join(vdir, fn)}')
            tensor = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
            feats[i] = model.encode_image(tensor).float().cpu().numpy()[0]
    np.save(out_path, feats)
    return len(names), len(picks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[sht_extract] device={device} stride={STRIDE}', flush=True)
    model, preprocess = clip.load('ViT-B/16', device=device, jit=False)
    model.eval()

    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    for split in splits:
        root, out_root = SPLITS[split], os.path.join(OUT_ROOT, split)
        os.makedirs(out_root, exist_ok=True)
        videos = sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)))
        t0 = time.time()
        done = 0
        for k, vid in enumerate(videos):
            out_path = os.path.join(out_root, vid + '.npy')
            if os.path.exists(out_path):
                continue
            n_frames, n_feats = extract_video(model, preprocess, device,
                                              os.path.join(root, vid), out_path)
            done += 1
            if done % 20 == 0:
                rate = done / (time.time() - t0)
                print(f'  [{split}] {k + 1}/{len(videos)} videos '
                      f'({rate:.1f} videos/s, last={vid} '
                      f'frames={n_frames} feats={n_feats})', flush=True)
        print(f'[{split}] done: {done} extracted, '
              f'{len(videos) - done} already present, '
              f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
