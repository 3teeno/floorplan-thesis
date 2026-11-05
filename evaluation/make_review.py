"""Review pack for the final 500 run: per-plan summary CSV + input/output pictures by outcome."""
import io, os, sys
import cv2, ezdxf, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

B = "output/final_500"
LIVING = r"^(MH|OH|OH/K|H|AH|HO|TH)\d?$|^(MAKUU|OLO|TAKKA|TYÖ|TYO)?HUONE\d?$"
PER_GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 15

s = pd.read_csv(f"{B}/run_stats.csv").drop_duplicates("image_name", keep="last").set_index("image_name")
s = s[(s.status == "ok") & (s.listed_rooms > 0)]
r = pd.read_csv(f"{B}/run_rooms.csv").drop_duplicates()
r = r[r.image_name.isin(s.index)]
lab = r.room_text.fillna("").str.upper().str.strip()
r["living"] = lab.str.match(LIVING)
s["living_found"] = r.groupby("image_name").living.sum().reindex(s.index).fillna(0).astype(int)
s["error"] = s.living_found - s.listed_rooms.astype(int)
s["group"] = np.select([s.error == 0, s.error == -1, s.error <= -2, s.error >= 1],
                       ["1_exact", "2_one_too_few", "3_two_or_more_too_few", "4_too_many"], default="")
s["rooms_found"] = r.groupby("image_name").apply(
    lambda g: ", ".join(f"{t if isinstance(t, str) else '-'} {a:.1f}" for t, a in zip(g.room_text, g.area_m2) if a >= 4))
cols = ["group", "listed_rooms", "living_found", "error", "listed_room_configuration", "listed_size_m2",
        "num_rooms_total", "num_rooms_shown", "num_rooms_named", "num_text_labels", "num_doors", "num_windows",
        "width_px", "height_px", "rooms_found"]
s[cols].sort_values(["group", "error"]).to_csv(f"{B}/review_summary.csv")
print(s.group.value_counts().sort_index().to_string())

cfg = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.COLOR)
for g, rows in s.groupby("group"):
    out_dir = f"{B}/review/{g}"
    os.makedirs(out_dir, exist_ok=True)
    for n in rows.sample(min(PER_GROUP, len(rows)), random_state=0).index:
        doc = ezdxf.readfile(f"{B}/dxf/{n[:-4]}.dxf")
        fig = plt.figure(figsize=(12, 12)); ax = fig.add_axes([0, 0, 1, 1])
        Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(doc.modelspace(), finalize=True)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig)
        out = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR); H = out.shape[0]
        inp = cv2.imread(f"output/floorplan_images/{n}"); inp = cv2.resize(inp, (int(inp.shape[1] * H / inp.shape[0]), H))
        e = int(rows.loc[n, "error"])
        cv2.imwrite(f"{out_dir}/{e:+d}_{n[:-4]}.png", np.hstack([inp, np.full((H, 30, 3), 255, np.uint8), out]))
print("done")
