# XFeas codebase guide

A tour of the repository for collaborators: what each folder is for, how the pieces connect, and what every file does.
Setup and commands are in the [README](../README.md).

---

## 1. The idea

A robot already has a locomotion controller (AMO, ReLIC, SONIC, ...), which XFeas never changes. Every 0.25 s XFeas looks at
the head-camera depth image and decides **which of the controller's own commands to send**: walk, slow down, crouch,
sidestep, stand up. A network scores every possible command for "what happens if I hold this for 1.5 s?" (fall / stall /
contact / progress / success) and the best one is sent. Commands are compared by what a short **self-test** measured them to
do on that controller, not by their names. Simple geometric rules (`skill_rules.py`) decide **when** to switch to a crouch
or sidestep and how to stand back up; the model decides **which** of the controller's commands to use.

## 2. Three pipelines

```
 DATA (Isaac Sim)                          TRAIN (PyTorch)                 CLOSED LOOP (Isaac Sim + PyTorch)
 make_suites      scenes to collect        branch_dataset   loads data     episode_runner   one evaluation episode
 job_queue        runs jobs on GPU 0       branch_labels    outcomes          └ xfeas_method   every 0.25 s asks ...
 branch_collector one job                  trainer          ─► model.pt       └ model_server   ... the network (separate process)
   branch_sim       save / replay a moment   feasibility_model                    └ controller adapter in xfeas/sim/controllers
   branch_schedule  which commands to try
 job_audit        checks each job
 controller_selftest  the self-test, used by all three pipelines
```

Entry points at the repo root: `train.py` (training), `evaluate.py` (closed loop + offline model check), `collect.py`
(one data job), `probe.py` (controller self-test).

## 3. Names you will see

**Controller IDs:**
- **G1 humanoid:** `g1c` AMO, `g1g` GR00T-WBC, `g1l` Isaac Lab policy, `g1u` URMA, `g1p` PHP, `g1s` GEAR-SONIC.
- **Spot:** `spotp` ReLIC, `spotf` Isaac Sim flat-terrain policy.
- **Other robots:** `jackal` (wheeled), `iris` (drone).
- **Variants:** `v_*`, `lk_*` and `mm_*` are look-alike variants of a real controller.

**Generations:** the method went through five versions, and the current one builds on the earlier ones (it extends their
classes and starts from the gen4 network weights). That older code lives in `xfeas/_support/` and keeps its original names:
`gen3`, `gen4`, `_v1`, `_v2`, `model_xe`. You only need it to follow an import. Some names still say "gen5" / "gen5b"
(data folders, run names, the arm `g5b`): that is the current method.

| Term | Meaning |
|---|---|
| spec | JSON describing a controller's command interface: names, ranges, nominal values, modes |
| self-test / probe | scripted commands on a controller, recording what each one does and what switching does |
| candidate | one command XFeas may choose: a point on the controller's command grid, plus KEEP and NOMINAL |
| descriptor | the vector describing a candidate by its measured effect |
| situation | one reproducible scene setup: environment, lane, obstacle, lighting, seed |
| moment / branch | a saved simulator state, and one command replayed from it |
| arm | a method being evaluated: `stock`, `g5sm` (current XFeas: rules time, model picks), `g5sr`, `g5sr0` (rules only), `g5s` (model only), `g5b` / `g5bho` (earlier model without SONIC) |
| cell | one evaluation job: robot × arm × scene × trials |
| success | within 0.5 m of the goal and no fall; no map; goal given relative to the robot |

---

## 4. Every file

### Root

| File | What it does |
|---|---|
| `train.py` | Train XFeas or evaluate a model offline (runs `xfeas/train/trainer.py`) |
| `evaluate.py` | Closed-loop evaluation (`school`, `gym`, `terrain`) or offline model check (`model`) |
| `collect.py` | One data-collection job in Isaac Sim |
| `probe.py` | Controller self-test in Isaac Sim |
| `env.sh` | Machine paths for the shell launchers; `source env.sh` in every shell |
| `pyproject.toml` | Package definition (`pip install -e .`) and formatter settings |
| `third_party/setup.sh` | Fetches the released controllers at pinned commits |
| `checkpoints/xfeas_all_g1s_v4.pt` | The current trained model (Git LFS); details in `checkpoints/README.md` |

### `xfeas/model/`: the network and its inputs

| File | What it does |
|---|---|
| `feasibility_model.py` | **The network.** Encodes the scene once per tick, scores every candidate (fall, stall, contact, progress, success) with a 5-member ensemble |
| `candidates.py` | Lists the candidate commands for a controller from its spec |
| `descriptors.py` | Describes each candidate by its self-test-measured effect |
| `controller_registry.py` | All controllers: IDs, robot, how to build each in simulation, spec files, held-out set |
| `joint_layout.py` | One joint layout across G1, Spot, Jackal and Iris |
| `path_follower.py` | The stock steering used by every method |

### `xfeas/data_collection/`: training data in Isaac Sim

| File | What it does |
|---|---|
| `branch_collector.py` | **One collection job**: walk a scene, save moments before the obstacle, replay each with different commands |
| `branch_schedule.py` | Which commands are tried at which moments |
| `branch_sim.py` | Simulator harness: save / restore a moment, run one held command |
| `controller_selftest.py` | **The self-test** of a controller |
| `make_suites.py`, `make_env_suites.py` | Write the scene lists (suites) and the job list |
| `make_controller_specs.py` | Write controller specs and the look-alike variants |
| `job_queue.py` | Run all jobs on GPU 0 with restarts; audit each and set failures aside |
| `job_audit.py` | Check one finished job and write contact sheets |
| `run_isaac.sh` | Isaac launcher used by every collection / self-test job |
| `watchdog.sh` | Restarts the queue if it dies |

### `xfeas/train/`: datasets and training

| File | What it does |
|---|---|
| `trainer.py` | **Training and offline evaluation** (`--holdout-base` leaves a controller out) |
| `branch_dataset.py` | Loads the side-by-side moments; `build` writes the manifest |
| `branch_labels.py` | Outcome labels of a tried command |
| `build_manifest_v4.py` | Builds the training manifest for the current model (with SONIC) and runs a data-quality check |
| `run_sonic_retrain.sh` | The current model's full chain: collect → manifest + check → train `all_g1s_v4` → closed-loop test |
| `trainer_gen4data.py`, `gen4_option_dataset.py` | The same model trained on the earlier gen4 side-by-side data, which `trainer.py` mixes in |
| `run_training.sh` | Full training sequence with restarts |

### `xfeas/closed_loop/`: running and scoring episodes

| File | What it does |
|---|---|
| `episode_runner.py` | **Runs one evaluation episode** (scene, robot, method, scoring, video); every method uses it |
| `xfeas_method.py` | **XFeas inside an episode**: every 0.25 s asks the model server and sends the chosen command |
| `model_server.py` | **The network as a separate PyTorch process**; `--selftest` checks a model offline |
| `skill_rules.py` | **When to switch skill**: detects overhangs / narrow gaps from the head-camera points; walk / crouch-crawl / sidestep timing and the staged stand-up. The model picks the move inside the skill |
| `sonic_primitives.py` | Optional SONIC execution helpers (arm `g5smP`): turn in place before a sidestep, hold heading and lateral position |
| `local_nav.py`, `avoid_threshold.json` | Map-free navigation shared by every method |
| `fall_judge.py` | The fall rule (crawl and kneel modes tilt on purpose) |
| `school_overhang_scenes.py` | Test scenes `SC2A_O1` / `O1L` / `O1N` (school course cut after its first doorway) |
| `school_episode.py`, `gym_episode.py` | Episode entry points for the school and gym courses |
| `terrain_method.py`, `terrain_harness_g1.py`, `terrain_harness_spot.py` | XFeas on terrain tiles and the tile harnesses |
| `run_school_cell.sh`, `run_gym_cell.sh`, `run_terrain_trial.sh` | Launch one cell / trial (`evaluate.py` calls these) |
| `run_school_matrix.sh` | All school-course cells, restart-safe |

### `xfeas/sim/`: robots and controllers in Isaac Sim

| File | What it does |
|---|---|
| `robots.py` | **Robot adapters**: one interface (`command(twist)`, `step(dt)`, `reset()`) for every robot |
| `camera_mount.py` | The shared head camera used everywhere |
| `gpu_wrap.py` | Starts Isaac Sim on one GPU, then runs the target script |
| `isaaclab_python.sh` | Isaac Sim with the Isaac Lab venv (terrain tiles) |
| `foot_material.py` | Consistent friction for feet and surfaces |
| `rec_cam.py`, `chase.py`, `episode_video.py`, `video.py` | Recording cameras and video (never seen by any method) |
| `arena.py` | Flat hazard arena from the first XFeas version |
| `controllers/g1_amo.py` | G1 + AMO. The AMO robot used in episodes is built by `_support/closed_loop/sidestep_g1.py`, which adds full-body colliders |
| `controllers/spot_posture.py` | Spot + ReLIC |
| `controllers/spot_flat5.py` | Spot + Isaac Sim flat-terrain policy |
| `controllers/xbody5.py` | Jackal and Iris |
| `controllers/virtual5.py` | Look-alike controller variants |
| `controllers/amo_mujoco_check.py` | Checks AMO in its own MuJoCo simulator against the Isaac port |
| `controllers/sonic/` | GEAR-SONIC: Isaac adapter, numpy re-implementation, policy server + client, MuJoCo reference |
| `controllers/php/` | PHP parkour: Isaac adapter, numpy/onnx re-implementation, server + client, harness adapter `php5.py` |
| `controllers/gr00t/`, `controllers/urma/`, `controllers/isaaclab/` | GR00T-WBC, URMA and the Isaac Lab G1 policy |

### `xfeas/scenes/`: scenes and courses

Everything is built **in memory** from the school's own assets; `assets/school.usd` is never written.

| File | What it does |
|---|---|
| `school_course.py`, `school_course_v2.py`, `school_course_v2a.py` | The school course (v2a is evaluated) |
| `gym_course*.py`, `course_flat.py` | Gym obstacle course versions and its 5 levels |
| `overhang_lib.py`, `overhang_scenes.py`, `g1_overhang.py`, `obstacle_lib.py` | Overhangs and ground obstacles |
| `climb_lib.py`, `climb_scenes.py` | Obstacles to climb |
| `stairs_scenes.py`, `stairs_gen2.py`, `hall_cam.py` | The school's real staircases |
| `sidestep_scene.py`, `jog_scene.py`, `jog5.py` | Narrow passages and fenced lanes with a jog |
| `real_assets.py`, `hazard_copy.py` | Hazards made by duplicating real scene objects |
| `scene_clean.py`, `scene_hazards.py`, `baked_scenes.py`, `school_*.json` | School clean-up and baked test scenes |
| `scene_lighting.py` | Day, dusk and night-with-headlight lighting |
| `course.py` | Early exploration course |
| `terrain/` | Terrain tiles: `catalog.py` (specs), `build.py` (builder), `terrain_classes.py` (physical classes) |

### `baselines/`, `tools/`, `assets/`

| File | What it does |
|---|---|
| `baselines/vamos_r1.py`, `vamos_gpu0.py`, `cmp_methods.py` | VAMOS (released) run through the same evaluation |
| `baselines/evora_plugin_r1.py`, `evora_mppi_server_r1.py` | EVORA (released) |
| `baselines/navloco_spot.py` | Navigation-locomotion planner on Spot |
| `baselines/gym_course_spot.py`, `gym_course_spot_jog.py` | Spot variants of the gym course |
| `tools/relocate_data_paths.py` | After downloading data: re-point stored absolute paths |
| `tools/relink_school_usd.py` | Point `assets/school.usd` at your asset pack |
| `tools/summarize_school_matrix.py`, `tools/terrain_table.py` | Result tables |
| `tools/decision_video.py` | Diagnostic video with each decision overlaid |
| `assets/school.usd`, `assets/courses/` | The school scene and the course definitions (start, path, goal, obstacles) |

### `xfeas/_support/`: earlier generations (read only when following an import)

| Folder | Contents |
|---|---|
| `model/` | gen1–gen4 networks (`model_xe`, `model_gen*`), their features (`gen3_common`, `gen4_common`, `sp_tokens`), the geometric feasibility check (`feas`, `feas_mem`), clearance map (`cmap`), spec loader (`ctrl_spec`) and `specs/` |
| `data_collection/` | The simulator harness chain the current one extends (`gensim` → `gensim_v1` → `gensim_v2` → `gensim3` → `gensim4` → `gensim5`), other environments (`envsim`), situation generators (`families`, `envfam*`), self-test plan (`probe`), look-alike controllers (`virtual`), gen3 command mapper (`driver`), gen5 random-switch collector parts |
| `train/` | Earlier datasets (`dataset_gen*`, `dataset_xe`, `dataset5`), labels (`labels5`), test-scene exclusion, robot descriptors (`*_desc.json`) |
| `closed_loop/` | Earlier closed-loop arms and model servers (`gen*_methods`, `select_gen*`, `select_xe`), course registration the current episodes import (`course4_ep`, `course_ctrl`, `school_ctrl`, `seq_course`), stock / oracle arms for single-skill scenes, and `sidestep_g1` (the AMO robot with full-body colliders) |
