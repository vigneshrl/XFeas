"""Builds the terrain-tile results table (successes per robot and tile)."""
import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

# gen5b terrain table (2026-10-28): per robot x tile, strict successes / 3 + outcomes, with the gen4 canonical row beside it
import json, os, glob

O = xfeas.paths.ROOT + "/data/gen5_terrain"
G4 = {
    x["cell_id"]: x
    for x in json.load(
        open(
            "/tmp/claude-15744551/-p-cral-vignesh-ARL/4b466661-928b-4204-a39c-fceb06a50462/scratchpad/navtable/gen4_cells.json"
        )
    )
}
TILES = [
    ("sand_dunes", "Sand dunes"),
    ("rubble_rocky", "Rocky"),
    ("roots_leaves", "Leaves"),
    ("oil_spill", "Oil"),
    ("ice", "Ice"),
    ("wet_tile", "Wet tile"),
    ("sticky_clay", "Sticky clay"),
]
rows = []
for c, rn in (("spot", "Spot"), ("amo", "G1 (AMO)")):
    for t, tn in TILES:
        oc, tt, cls = [], [], {}
        for k in range(3):
            f = f"{O}/{c}_{t}/t{k}/result.json" if c != "spot" else f"{O}/{c}_{t}/t{k}/{t}_t{k}.json"
            if not os.path.exists(f):
                oc.append("?")
                continue
            r = json.load(open(f))
            oc.append(r["outcome"])
            tt.append(r["t_end"] if r["outcome"] == "success" else None)
            d = f"{O}/{c}_{t}/t{k}/diag"
            for fj in glob.glob(f"{d}/decisions5_t*.jsonl"):
                for l in open(fj):
                    n = json.loads(l)["chosen"]["cls"]
                    cls[n] = cls.get(n, 0) + 1
        k5 = sum(o == "success" for o in oc)
        g = G4.get(f"{c}_{t}", {})
        tot = sum(cls.values()) or 1
        top = ", ".join(f"{n} {100 * v // tot}%" for n, v in sorted(cls.items(), key=lambda x: -x[1])[:2])
        rows.append(dict(robot=rn, tile=tn, gen5b=f"{k5}/3", outcomes=oc, gen4=g.get("SR", "-"), picks=top))
        print(f"| {rn} | {tn} | **{k5}/3** | {g.get('SR', '-')} | {' / '.join(oc)} | {top} |")
json.dump(rows, open(f"{O}/table.json", "w"), indent=1)
