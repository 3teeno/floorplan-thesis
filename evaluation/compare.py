"""Before (review_500 run, 8 Oct) vs after (new tracing + rooms) on the same plans."""
import re, sys
import cv2, ezdxf, numpy as np, pandas as pd

LIVING = r"^(MH|OH|OH/K|H|AH|HO|TH)\d?$|^(MAKUU|OLO|TAKKA|TYÖ|TYO)?HUONE\d?$"
names = open(sys.argv[1] if len(sys.argv) > 1 else "output/trace_eval_images.txt").read().split()
runs = {"before": ("output/review_500_stats.csv", "output/review_500_rooms.csv", "output/review_500_dxf"),
        "after": tuple(f"{sys.argv[2] if len(sys.argv) > 2 else 'output/trace_eval40'}/{f}" for f in ("run_stats.csv", "run_rooms.csv", "dxf"))}
AREA_RE = re.compile(r"(\d{1,3}(?:[.,]\d)?)\s*M[2²]", re.I)


def printed_vs_computed(dxf, listed_m2):
    """Pairs (printed m2 on the plan, computed m2) for every computed room containing exactly one
    printed area label."""
    doc = ezdxf.readfile(dxf); msp = doc.modelspace()
    polys = [np.array([(x, y) for x, y, *_ in e.get_points()], np.float32) for e in msp.query("LWPOLYLINE[layer=='Area']")]
    comp = []
    for t in msp.query("TEXT[layer=='Area']"):
        m = re.search(r"([\d.]+) m²", t.dxf.text)
        if m:
            v = t.dxf.align_point if t.dxf.hasattr("align_point") else t.dxf.insert; comp.append((float(m.group(1)), (v.x, v.y)))
    printed = []
    for t in msp.query("TEXT[layer=='Text']"):
        m = AREA_RE.search(t.dxf.text)
        if m:
            printed.append((float(m.group(1).replace(",", ".")), (t.dxf.insert.x, t.dxf.insert.y)))
    pairs = []
    for poly in polys:
        inside_c = [a for a, p in comp if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0]
        inside_p = [a for a, p in printed if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0]
        if len(inside_c) == 1 and len(inside_p) == 1 and 2 <= inside_p[0] < 0.5 * listed_m2:  # not the flat's total
            pairs.append((inside_p[0], inside_c[0]))
    return pairs


rows = []
for run, (sf, rf, dd) in runs.items():
    s = pd.read_csv(sf).drop_duplicates("image_name", keep="last").set_index("image_name")
    s = s.loc[[n for n in names if n in s.index]]
    ok = s[s.status == "ok"]
    r = pd.read_csv(rf).drop_duplicates(); r = r[r.image_name.isin(ok.index)]
    lab = r.room_text.fillna("").str.upper().str.strip()
    living = lab.str.match(LIVING).groupby(r.image_name).sum().reindex(ok.index).fillna(0)
    d = living - ok.listed_rooms
    shown = r[r.shown.astype(bool)]
    pairs = [p for n in ok.index for p in printed_vs_computed(f"{dd}/{n[:-4]}.dxf", ok.loc[n, "listed_size_m2"])]
    err = np.array([c - p for p, c in pairs]); rel = np.array([abs(c - p) / p for p, c in pairs])
    rows.append({
        "run": run, "plans ok": f"{len(ok)}/{len(s)}",
        "living exact": f"{(d == 0).mean():.0%}", "living within ±1": f"{(d.abs() <= 1).mean():.0%}",
        "living MAE": round(d.abs().mean(), 2), "living bias": round(d.mean(), 2),
        "rooms shown/plan": round(ok.num_rooms_shown.mean(), 1),
        "shown rooms named": f"{(~shown.room_text.fillna('').isin(['', 'Other'])).mean():.0%}",
        "rooms <4 m2/plan": round((r.area_m2 < 4).groupby(r.image_name).sum().reindex(ok.index).fillna(0).mean(), 1),
        "printed-area rooms": len(pairs),
        "area MAE m2": round(np.abs(err).mean(), 2) if len(err) else None,
        "area median % err": f"{np.median(rel):.0%}" if len(rel) else None,
        "area within 10%": f"{(rel <= 0.10).mean():.0%}" if len(rel) else None,
        "median time s": ok.processing_time_sec.median(),
    })
print(pd.DataFrame(rows).set_index("run").T.to_string())
