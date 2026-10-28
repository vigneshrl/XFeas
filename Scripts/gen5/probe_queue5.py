#!/usr/bin/env python
"""
gen5 probe queue (plain python): runs probe_transitions5.py jobs on GPU 0 / 1, <= 3 Isaac processes per GPU (ANY user,
nvidia-smi + own CUDA_VISIBLE_DEVICES), thread gate ps -u atk9sb -L < 1600, max_mine slots, hang watchdog (log silent
> 20 min -> kill that job's process group), 2 tries. Jobs: data/probes_v5/queue/jobs.json [{id, base, ctrls, heldout}].
Writes data/probes_v5/MANIFEST.json when every job finished (scripts/gen5/manifest5.py).
    nohup python3 scripts/gen5/probe_queue5.py --max-mine 5 > data/probes_v5/queue/queue.out 2>&1 &
"""
import argparse, json, os, signal, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
Q = os.path.join(ROOT, "data", "probes_v5", "queue"); os.makedirs(Q, exist_ok=True)
LOGD = os.path.join(ROOT, "data", "probes_v5", "logs"); os.makedirs(LOGD, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--max-mine", type=int, default=5)
ap.add_argument("--jobs", default=os.path.join(Q, "jobs.json"))
ap.add_argument("--qdir", default="", help="queue dir (default data/probes_v5/queue); generic jobs: {id, argv: [script, args...]}")
ap.add_argument("--no-manifest", action="store_true")
a = ap.parse_args()
if a.qdir:
    Q = os.path.abspath(a.qdir); os.makedirs(Q, exist_ok=True); LOGD = os.path.join(Q, "logs"); os.makedirs(LOGD, exist_ok=True)
GPUS = ("0", "1"); MAX_PER_GPU = 3; THREAD_CAP = 1600; THREADS_PER_JOB = 260; HANG_S = 1200


def log(*x):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x))
    print(s, flush=True); open(os.path.join(Q, "queue.log"), "a").write(s + "\n")


def isaac_pids():
    P = set()
    for l in subprocess.run(["ps", "-eo", "pid,stat,args"], capture_output=True, text=True).stdout.splitlines():
        f = l.split(None, 2)
        if len(f) == 3 and ("kit/python/bin/python3" in f[2] or "isaaclab_venv/bin/python" in f[2]) and not f[1].startswith("T"):
            P.add(int(f[0]))
    return P


def per_gpu():
    cnt = {g: 0 for g in GPUS}
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
        idx = {u.strip(): i.strip() for i, u in (l.split(",") for l in uu.strip().splitlines())}
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
        ip = isaac_pids()
        for l in apps.strip().splitlines():
            u, p = [x.strip() for x in l.split(",")]
            if int(p) in ip and idx.get(u) in cnt:
                cnt[idx[u]] += 1
    except Exception as e:
        log("per_gpu failed", repr(e))
    own = {g: 0 for g in GPUS}
    for p in isaac_pids():
        try:
            env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
            g = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "0")
            if g in own:
                own[g] += 1
        except Exception:
            pass
    return {g: max(cnt[g], own[g]) for g in GPUS}


def threads():
    try:
        return int(subprocess.run(["bash", "-c", "ps -u atk9sb -L --no-headers | wc -l"], capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        return 99999


def main():
    J = json.load(open(a.jobs))
    st = {j["id"]: dict(status="pending", tries=0) for j in J}
    procs = {}
    log("probe queue start", os.getpid(), len(J), "jobs")
    open(os.path.join(Q, "queue.pid"), "w").write(str(os.getpid()))
    while True:
        now = time.time()
        for jid, (p, lf, t0) in list(procs.items()):
            rc = p.poll()
            if rc is None and now - os.path.getmtime(lf) > HANG_S:
                log("KILL hang", jid);
                try:
                    os.killpg(p.pid, signal.SIGTERM); time.sleep(10); os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(); rc = -999
            if rc is not None:
                txt = open(lf, errors="ignore").read()
                ok = "[p5] DONE" in txt or "[v5] DONE" in txt
                s = st[jid]; s["rc"] = rc; s["wall_s"] = round(now - t0)
                s["status"] = "done" if ok else ("pending" if s["tries"] < 2 else "failed")
                log("END", jid, rc, s["status"], s["wall_s"], "s")
                del procs[jid]
        for j in J:
            if st[j["id"]]["status"] != "pending" or j["id"] in procs or len(procs) >= a.max_mine:
                continue
            cnt = per_gpu(); g = min(GPUS, key=lambda x: (cnt[x], x))
            if cnt[g] >= MAX_PER_GPU or threads() + THREADS_PER_JOB >= THREAD_CAP:
                break
            s = st[j["id"]]; s["tries"] += 1
            lf = os.path.join(LOGD, f"{j['id']}_try{s['tries']}.log")
            if "argv" in j:
                cmd = ["timeout", "-k", "60", str(j.get("timeout_s", 14400)), os.path.join(ROOT, "scripts", "gen5", "run_isaac5.sh"), g] + list(j["argv"])
            else:
                cmd = ["timeout", "-k", "60", "14400", os.path.join(ROOT, "scripts", "gen5", "run_isaac5.sh"), g, "scripts/gen5/probe_transitions5.py",
                       "--base", j["base"], "--ctrls", ",".join(j["ctrls"])] + (["--heldout"] if j.get("heldout") else []) + (["--redo"] if j.get("redo") else [])
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=open(lf, "w"), stderr=subprocess.STDOUT, start_new_session=True)
            procs[j["id"]] = (p, lf, time.time()); s.update(status="running", gpu=g, pid=p.pid)
            log("START", j["id"], "gpu", g, "pid", p.pid, "per_gpu", cnt)
            time.sleep(45)
        json.dump(st, open(os.path.join(Q, "state.json"), "w"), indent=0)
        if not procs and all(v["status"] in ("done", "failed") for v in st.values()):
            log("all jobs finished")
            if not a.no_manifest:
                subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen5", "manifest5.py")], cwd=ROOT)
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
