#!/usr/bin/env python
"""
gen_v5b production queue (copy of scripts/gen5/queue5.py with data/gen_v5b, Scripts/data_collector/run_gen_v5b.py, audit5b, [v5b] lines, br/ + side/ progress). Original:
gen_v5 production queue (copy of scripts/gen4/queue_v4.py, which is not modified). Changes: data/gen_v5, the gen5 collector
(Scripts/data_collector/run_isaac5.sh + run_gen_v5.py), progress counted from '[v5b]' lines + episode npz + video frames, and an audit
FAIL QUARANTINES the job (status audit_fail, listed in data/gen_v5/queue/QUARANTINE.txt) -- the queue never pauses
(memory rule 'never pause long runs'). Original header:
gen_v4 production queue (plain python; launches Isaac jobs). GEN4_DATA_PLAN.md section 5.
  jobs   : data/gen_v4/queue/jobs.jsonl (re-read every cycle; append to add work):
           {id, env, base, suite, chunk, launcher ('base'|'isaaclab'), timeout_s, hang_s, prio}
  state  : data/gen_v4/queue/state.json {id: {status pending|running|done|failed|audit_fail, tries, pid, gpu, ...}}
  logs   : data/gen_v4/logs/<id>_try<k>.log, data/gen_v4/queue/queue.log
Rules:
  * GO file data/gen_v4/queue/GO must exist (written only after Phase 0/1 passed: camera check sheets PASSED in
    analysis/REBUILD_V4_PHASE01.md and data/probes_v4 manifest present); STOP file: no new launches (running jobs finish).
  * GPUs 0 and 1 (user approval 2026-10-19 for the gen_v4 rebuild): a job goes to the GPU with the fewest Isaac processes
    (ANY user, from nvidia-smi), only if that GPU has < MAX_PER_GPU (3); my running jobs <= max_mine (file, default 6).
  * thread gate: total threads of the current user + THREADS_PER_JOB < THREAD_CAP (1700) before every launch.
  * hang watchdog: a job is killed (its own process group) when the Isaac app is not up after STARTUP_S, or when its
    progress counter (log lines '[v4]' / '[probe]' + rollout / obs npz files + side-video frames written for the chunk's
    situations) has not grown for hang_s. Killed / crashed jobs are retried (MAX_TRIES) and RESUME.
  * audit gate: every finished job is audited (audit_gen_v4.py --part <id>); a FAIL marks the job audit_fail and writes
    STOP with the reason (the whole queue stops launching until a human looks).
  nohup $XFEAS_PY scripts/gen5/queue5.py > data/gen_v5/queue/queue.out 2>&1 &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import glob, json, os, re, signal, subprocess, sys, time

ROOT = xfeas_paths.ROOT
D4 = os.path.join(ROOT, "data", "gen_v5b")
Q = os.path.join(D4, "queue"); os.makedirs(Q, exist_ok=True)
LOGD = os.path.join(D4, "logs"); os.makedirs(LOGD, exist_ok=True)
JOBS, STATE = os.path.join(Q, "jobs.jsonl"), os.path.join(Q, "state.json")
MAX_PER_GPU, MAX_TRIES, STARTUP_S, THREAD_CAP, THREADS_PER_JOB = 3, 4, 600, 1600, 260
GPUS = ("0", "1")
PY = xfeas_paths.XFEAS_PY
ENV = dict(os.environ, XFEAS_FOOT_MATERIAL="1", XFEAS_PERC_MINPOOL_FIX="1", OMP_NUM_THREADS="4")
ENV.pop("XFEAS_LEGACY_PRIMITIVES", None); ENV.pop("XFEAS_CAM_MOUNT", None)


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a))
    print(s, flush=True); open(os.path.join(Q, "queue.log"), "a").write(s + "\n")


def isaac_pids():
    out = subprocess.run(["ps", "-eo", "pid,stat,args"], capture_output=True, text=True).stdout.splitlines()
    P = set()
    for l in out:
        f = l.split(None, 2)
        if len(f) == 3 and ("kit/python/bin/python3" in f[2] or "isaaclab_venv/bin/python" in f[2]) and not f[1].startswith("T"):
            P.add(int(f[0]))
    return P


def per_gpu():
    """Isaac processes per GPU index (any user), from nvidia-smi compute apps"""
    cnt = {g: 0 for g in GPUS}
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
        idx = {u.strip(): i.strip() for i, u in (l.split(",") for l in uu.strip().splitlines())}
        ap = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
        ip = isaac_pids()
        for l in ap.strip().splitlines():
            u, p = [x.strip() for x in l.split(",")]
            if int(p) in ip and idx.get(u) in cnt:
                cnt[idx[u]] += 1
    except Exception as e:
        log("per_gpu failed", repr(e))
    own = {g: 0 for g in GPUS}                 # my own Isaac processes by their CUDA_VISIBLE_DEVICES (also before CUDA init)
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
        return int(subprocess.run(["bash", "-c", "ps -u $(id -un) -L --no-headers | wc -l"], capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        return 99999


def load_jobs():
    J = []
    if os.path.exists(JOBS):
        for l in open(JOBS):
            l = l.strip()
            if l:
                try:
                    J.append(json.loads(l))
                except Exception:
                    pass
    return J


def max_mine():
    try:
        return int(open(os.path.join(Q, "max_mine")).read().strip())
    except Exception:
        return 6


_SIDS = {}


def progress(j, logf):
    """log lines + npz / frame files written for the chunk's situations"""
    n = 0
    try:
        n += sum(1 for l in open(logf, errors="ignore") if l.startswith(("[v5b",)))
    except Exception:
        pass
    if j["id"] not in _SIDS:
        S = json.load(open(os.path.join(ROOT, j["suite"])))
        i0, i1 = map(int, j["chunk"].split(":"))
        _SIDS[j["id"]] = [s["sid"] for s in S[i0:i1]]
    base = os.path.join(D4, j["env"], j["base"])
    for sid in _SIDS[j["id"]]:
        n += len(glob.glob(os.path.join(base, "br", f"{sid}__*.npz")))
        for v in glob.glob(os.path.join(base, "side", f"{sid}__*")):
            n += len(os.listdir(v))
    return n


def audit(jid):
    r = subprocess.run([PY, os.path.join(ROOT, "Scripts", "data_collector", "audit5b.py"), "--root", D4, "--part", jid, "--tag", f"job_{jid}", "--sheets"],
                       capture_output=True, text=True, cwd=ROOT)
    last = [l for l in r.stdout.splitlines() if l.startswith("[audit5b]")]
    return r.returncode, (last[-1] if last else r.stdout[-300:] + r.stderr[-300:])


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for k, v in st.items():
        if v["status"] == "running":
            alive = False
            try:
                os.kill(v["pid"], 0); alive = True
            except Exception:
                pass
            if alive:
                log("WARN orphan still alive (not adopted, left running):", k, v["pid"])
            v["status"] = "pending"; v["why"] = "queue_restart"
    procs = {}
    log("queue start pid", os.getpid())
    open(os.path.join(Q, "queue.pid"), "w").write(str(os.getpid()))
    while True:
        J = {j["id"]: j for j in load_jobs()}
        for jid in J:
            st.setdefault(jid, dict(status="pending", tries=0))
        if os.path.exists(os.path.join(Q, "RESUME_DEFERRED")):     # overnight continuation: deferred -> pending
            for v in st.values():
                if v["status"] == "deferred":
                    v["status"] = "pending"
            os.remove(os.path.join(Q, "RESUME_DEFERRED")); log("deferred jobs -> pending")
        now = time.time()
        for jid, (p, logf, t0, j, gpu) in list(procs.items()):
            rc = p.poll()
            txt = ""
            try:
                txt = open(logf, errors="ignore").read()
            except Exception:
                pass
            if rc is None:
                why = None
                if "[v5b] app up" not in txt and now - t0 > STARTUP_S:
                    why = "startup_hang"
                else:
                    pg = progress(j, logf)
                    if pg != st[jid].get("last_n"):
                        st[jid]["last_n"] = pg; st[jid]["t_last"] = now
                    elif now - st[jid].get("t_last", t0) > j.get("hang_s", 1800):
                        why = "hang"
                if why:
                    log("KILL", jid, why, "pgid", p.pid)
                    try:
                        os.killpg(p.pid, signal.SIGTERM); time.sleep(10); os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait(); rc = -999; st[jid]["why"] = why
            if rc is not None:
                ok = rc == 0 and "[v5b] DONE" in txt
                # floor-render burst (2026-10-25: cold first load of a scene -> every head-camera floor check failed, 0 episodes,
                # audit PASS): retry the job instead of marking it done
                n_floor = len(re.findall(r"^\[v5b\] REJECT \S+: floor_not_rendered", txt, re.M))
                n_eps = len(re.findall(r"^\[v5b\] (?!REJECT|situation|app |env |DONE)\S+__\S+\s", txt, re.M))
                if ok and n_floor >= 3 and n_floor >= n_eps:
                    ok = False; rc = rc if rc else -998; st[jid]["why"] = f"floor_reject_burst {n_floor} vs {n_eps} episodes"
                s = st[jid]; s["rc"] = rc; s["t_end"] = now; s["wall_s"] = round(now - t0)
                if ok:
                    arc, line = audit(jid)
                    s["audit"] = line
                    if arc == 0:
                        s["status"] = "done"
                    else:
                        s["status"] = "audit_fail"
                        open(os.path.join(Q, "QUARANTINE.txt"), "a").write(f"{time.strftime('%m-%d %H:%M')} audit FAIL {jid}: {line}\n")
                        log("AUDIT FAIL -> quarantined (queue continues)", jid, line)
                else:
                    # a try that finished >= 1 situation made progress (resume skips it; crash_repeat skips a situation after 2
                    # crashes): it does not count against MAX_TRIES (hard cap 3 x MAX_TRIES launches)
                    if re.search(r"^\[v5b\] situation \S+ done", txt, re.M) and s.get("launches", s["tries"]) < 3 * MAX_TRIES:
                        s["launches"] = s.get("launches", s["tries"]) + 1; s["tries"] -= 1
                    s["status"] = "pending" if s["tries"] < MAX_TRIES else "failed"
                    s.setdefault("why", f"rc={rc}")
                log("END", jid, "rc", rc, "->", s["status"], f"{s['wall_s']}s", s.get("why", ""), s.get("audit", ""))
                del procs[jid]
        dl = os.path.join(Q, "DEADLINE")            # gen5b: epoch seconds; no NEW launches after it (running jobs finish; resumable)
        past = False
        try:
            past = os.path.exists(dl) and time.time() > float(open(dl).read().strip())
        except Exception:
            pass
        if past and not procs:
            nd = [k for k, v in st.items() if v["status"] == "pending"]
            for k in nd:
                st[k]["status"] = "deferred"          # not started before the deadline; set back to pending to continue later
            if nd:
                log(f"DEADLINE passed, nothing running: {len(nd)} pending jobs -> deferred")
        if os.path.exists(os.path.join(Q, "GO")) and not os.path.exists(os.path.join(Q, "STOP")) and not past:
            pend = sorted([j for jid, j in J.items() if st[jid]["status"] == "pending" and jid not in procs], key=lambda j: (j.get("prio", 50), j["id"]))
            for j in pend:
                if len(procs) >= max_mine():
                    break
                cnt = per_gpu()
                gpu = min(GPUS, key=lambda g: (cnt[g], g))
                if cnt[gpu] >= MAX_PER_GPU:
                    break
                th = threads()
                if th + THREADS_PER_JOB >= THREAD_CAP:
                    log(f"thread gate: {th} threads, not launching"); break
                jid = j["id"]; s = st[jid]; s["tries"] += 1; s.pop("why", None)
                logf = os.path.join(LOGD, f"{jid}_try{s['tries']}.log")
                cmd = ["timeout", "-k", "60", str(j.get("timeout_s", 7200)), os.path.join(ROOT, "Scripts", "data_collector", "run_isaac5.sh"), gpu,
                       "Scripts/data_collector/run_gen_v5b.py", "--env", j["env"], "--base", j["base"], "--suite", j["suite"],
                       "--chunk", j["chunk"], "--part", jid] + j.get("extra", [])
                p = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=open(logf, "w"), stderr=subprocess.STDOUT, start_new_session=True)
                procs[jid] = (p, logf, time.time(), j, gpu)
                s.update(status="running", pid=p.pid, gpu=gpu, t_start=time.time(), t_last=time.time(), last_n=0)
                log("START", jid, "try", s["tries"], "pid", p.pid, "gpu", gpu, "per_gpu", cnt, "threads", th)
                time.sleep(30)
        json.dump(st, open(STATE + ".tmp", "w"), indent=0); os.replace(STATE + ".tmp", STATE)
        if not procs and st and all(v["status"] in ("done", "failed", "audit_fail", "deferred") for v in st.values()) and os.path.exists(os.path.join(Q, "EXIT_WHEN_EMPTY")):
            log("all jobs finished"); break
        time.sleep(20)


if __name__ == "__main__":
    main()
