"""Build libero_obs_3v_n8.npz (and 2v/1v) from the LeRobot-v3 LIBERO dataset.
8 frames, one per task 0..7, mid-episode (first episode of that task).
LIBERO is 2-view; wrist_right duplicates wrist (same convention as the
original fixture / test_thor_calibrate_matrix.py)."""
import glob, json, sys
import numpy as np, pyarrow.parquet as pq, av
from PIL import Image
import argparse
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);args=p.parse_args();root=args.dataset
t = pq.read_table(root + "/data/chunk-000/file-000.parquet").to_pandas()
picks = []
for task in range(8):
    sub = t[t.task_index == task]
    ep = sub.episode_index.min()
    e = sub[sub.episode_index == ep]
    picks.append(int(e.iloc[len(e) // 2]["index"]))
print("picks", picks)
maxi = max(picks)
def grab(key):
    f = root + f"/videos/{key}/chunk-000/file-000.mp4"
    out = {}
    with av.open(f) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i in picks:
                img = fr.to_image().convert("RGB").resize((224, 224), Image.BILINEAR)
                out[i] = np.asarray(img, dtype=np.uint8)
                if len(out) == len(picks): break
    return out
imgs = grab("observation.images.image"); wr = grab("observation.images.wrist_image")
d = {"n": np.int64(8)}
for k, gi in enumerate(picks):
    row = t[t["index"] == gi].iloc[0]
    d[f"img_{k}"] = imgs[gi]; d[f"wrist_{k}"] = wr[gi]; d[f"wrist_right_{k}"] = wr[gi]
    d[f"state_{k}"] = np.asarray(row["observation.state"], dtype=np.float32)
    d[f"global_index_{k}"] = np.int64(gi); d[f"task_{k}"] = np.int64(row.task_index)
for v in (2,):
    np.savez(args.out, **d)
print("mean pix", [int(d[f"img_{k}"].mean()) for k in range(8)], "state0", d["state_0"])
