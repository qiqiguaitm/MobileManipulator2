# Automatic exploration mapping for robot02

This package integrates **explore_lite + Nav2 + FAST-LIO / SC-PGO**. It needs no
model API. The previous custom frontier search, candidate retry queue,
return-trajectory validator, and SLAM Toolbox scan bridge have been removed.
Existing object-approach and saved-map navigation entry points remain separate.

## Run

From the MobileManipulator2 workspace:

```bash
# Read-only preflight; no sensors or motion started.
bash scripts/auto_explore.sh

# Sensors, mapping and Nav2 readiness check; CAN stays down, then exits.
bash scripts/auto_explore.sh --preview --bounds -5 -5 5 5

# Autonomous exploration; use bounds appropriate to the actual floor area.
bash scripts/auto_explore.sh --execute --bounds -25 -25 25 25
```

Bounds are metres relative to the fresh mapping origin (initial x forward,
y left), up to 50 x 50 m. The supervisor refuses concurrent mapping, saved-map
navigation, chassis or object-approach processes and requires CAN2 down. It does
not take over another stack. Execute only with an on-site operator and available
emergency stop, folded arm, and isolated stairs/drop-offs.

`Ctrl+C` stops motion and saves. `--max-seconds` and `--max-distance` default to
`0` (disabled); positive values enable explicit test limits. Normal completion
comes when explore_lite has no available targets and returns to the initial pose;
the supervisor then exports the point clouds and navigation map. Remaining unknown
boundaries are recorded as `remaining_boundary_cells`, not a partial-result gate.
"Complete" means available exploration targets are exhausted, not that every
unknown area has been observed. No map growth for 180 s or a 120 s return timeout
still produces a partial result. Return is attempted, not guaranteed.

## Modules

| Module | Responsibility |
|---|---|
| `supervisor.py` | Preflight, exclusive CAN/process ownership, shutdown, final export |
| `explorer.py` | Start/pause/resume explore_lite; readiness and completion handling |
| `runtime.py` | Sensor state, input/TF/CAN checks, velocity gate, diagnostic map saving |
| `motion_safety.py`, `geometry.py` | Live scan sweep and shared chassis geometry |
| `exploration_mapper.py`, `occupancy.py` | Accumulate observations using PGO corrections and project a 2D map |
| `archive.py` | Bounded background observation/archive writes |
| `map_odom_tf.py` | Publish the committed map-to-odom correction independently at 20 Hz |
| `startup_clearance.py` | Bounded, temporary rear unknown-space assumption |
| `rotation_debug.py` | Read-only goal/controller diagnostics and replay |
| `export_map.py` | Validate and convert final SC-PGO keyframes into navigation maps |

The one-command script starts `auto_explore.explorer`. The `ros2 run auto_explore
explore` executable resolves to that same explore_lite coordinator; there is no
second/custom exploration backend. Normally use the supervisor script, which
also generates the run configuration and manages the stack.

## Mapping and navigation

FAST-LIO keeps its local matching map and odometry. SC-PGO publishes alternating
raw/optimized keyframe poses on `/sc_pgo/pose_snapshot`. The mapper archives
body-frame observations with sensor origins and interpolates backend corrections
for history; the latest tail uses
`T_global = T_PGO_key * inverse(T_LIO_key) * T_LIO_now`.

Both mapping consumers use `/fastlio/odom_measurement` synchronized to point-cloud
measurement time. The 5 cm voxel map accumulates occupied endpoints/free rays and
projects occupied/free/unknown cells to a 5 cm 2D grid. Backend corrections trigger
archive reconstruction. The map and its correction are committed together;
motion is held while the mapper is not ready. Archive errors/backlog exhaustion
are reported, not silently dropped. A separate TF process retains the committed
correction and stops publishing after 1.5 s without a fresh heartbeat.

The raw grid is `/explore/full_map`; the navigation copy `/map` applies bounds,
current-body unknown-cell clearing and optional startup rear clearance.
explore_lite consumes the global costmap and owns all frontier selection,
blacklisting, goal retries and return goals. Its pinned source, licence and local
patches are recorded in `../m-explore-ros2/UPSTREAM.md`.

NavFn plans global paths. Both costmaps combine the accumulated map, live scan
obstacles and inflation. Nav2's RotationShimController aligns a new path heading
when error exceeds 0.35 rad, using a point 0.5 m along the path, then hands off to
DWB. The shim checks the padded footprint with a 2 s lookahead; blocked turns
fall back to DWB. Rotation is capped at 0.18 rad/s and acceleration at 0.4 rad/s².
It uses the previous command for acceleration ramping (`closed_loop=false`)
because FAST-LIO odometry leaves twist zero; heading feedback still uses measured
pose/TF. The pose progress checker accepts 0.10 m translation or 0.15 rad rotation
within 20 s. No blind backup/spin recovery or costmap-clearing recovery is enabled.

Geometry assumes the bare TRACER 2.0 chassis is 702 x 610 mm, with base_link at its
centre; accessories are not included. Nav2 and the scan sweep add 5 cm padding.
Linear speed is capped at 0.1 m/s with no reverse. The gate additionally checks
fresh sensors, TF, CAN faults and exclusive command ownership; the chassis driver
has a verified 0.4 s command watchdog. Stale input immediately zeros commands and
pauses exploration; continuous invalid input for 10 s terminates the run.

`--startup-rear-clearance 0.20` is the default (0 disables; maximum 0.25 m). It
changes only unknown cells in a bounded rear portion of the initial turning
sweep, never measured obstacles. It stays at the start pose and expires after
90 s, 0.60 m travel or 0.60 m displacement. This is an operator-authorized
blind-space assumption, not sensor evidence. Raw/archive/final maps are unchanged.

## RViz and diagnostics

Open `config/exploration.rviz`. It shows the live map/scans, Nav2 path, current
robot frames, and explore_lite frontiers: blue contours are available, red are
blacklisted, and green spheres mark frontier centroids. `/explore/current_goal`
shows the actual navigation target and state; it is distinct from the centroid.
Orange cells show startup rear clearance. Opening RViz does not start exploration.

`/explore/rotation_debug` displays DWB trajectory/footprint diagnostics. Blocker
cells are reconstructed from the nearest published costmap, not an exact internal
controller snapshot. During shim rotation no new DWB evaluations are emitted;
retained DWB markers describe the previous evaluation. Use `/explore/nav_cmd_vel`
and odometry for current motion. Recorded overlays are optional and off by default.
Saved `debug/rotation_debug.json`, `navigation_goal.json` and
`navigation_goals.jsonl` support read-only diagnosis/replay.

## Saved maps

Runs are saved under `maps/auto_explore/<timestamp>/`; existing maps are never
overwritten. After disabling CAN and gracefully stopping SC-PGO, the supervisor
validates keyframes/poses and runs the existing `merge_scans.py`,
`downsample_pcd.py`, and `pcd_to_2dmap.py` pipeline. It does not run the tool that
fills all unobserved space as free. The atomic `navigation/` output contains:

- `GlobalMap.pcd`, `GlobalMap_hdl.pcd`: optimized cloud and localization cloud.
- `map.yaml`, `map.pgm`: 5 cm occupancy map in the same coordinates.
- `map_preview.png`, `manifest.json`, `export.log`: preview and export evidence.

`result.json` reports exploration status and independent `navigation_export.ready`.
An interrupted/partial run can export a usable partial map without being complete.
Preview mode does not export navigation maps. `exploration_preview/` is diagnostic;
`occupancy/` contains the observation archive and `sc_pgo/` the backend keyframes.
Logs include `events.jsonl`, `explorer.log`, `explore_lite.log` and `stack.log`.

After exploration shuts down, the existing navigation entry point can use:

```bash
ros2 launch slam hdl_navigation_launch.py map_dir:=/absolute/run/path/navigation launch_chassis:=false use_odom_fusion:=false
```

Offline export of a stopped recording is available via
`python3 -m auto_explore.export_map --source /path/to/recording --destination
/new/path/navigation --project-root "$MM2_PROJECT_ROOT"`. The destination must not
exist. Final projection retains the existing obstacle height band z=0.05..1.5 m
relative to the SC-PGO origin, not an automatically measured floor.

## Validation

After sourcing ROS and the workspace, use an isolated domain:

```bash
ROS_DOMAIN_ID=166 ROS_LOCALHOST_ONLY=1 python3 -m pytest src/exploration/auto_explore/test -q
ROS_DOMAIN_ID=166 ROS_LOCALHOST_ONLY=1 python3 src/exploration/auto_explore/test/heading_controller_check.py
```

The second command launches the installed controller with fake pose feedback and
zero odometry twist; it checks rearward heading alignment followed by translation,
a straight path, blocked-footprint stopping and speed limits. It requires CAN2
down and starts no chassis driver. Mapper/archive, TF freshness, completion,
startup diagnostics, motion-gate and export tests cover the remaining integration.

`test/smoke_launch.py` exercises the no-motion coordinator with synthetic sensors;
`test/upstream_smoke.py` exercises explore_lite/Nav2 on synthetic inputs. Neither
starts a chassis driver. Run them only in isolated domains and stop with SIGINT.
`test/frontier_retry_check.py` and `test/goal_marker_check.py` use saved costmap
fixtures for upstream goal/marker regressions.

The 2026-10-13 pre-cleanup physical check travelled 5.24 m with the heading shim
and no `Failed to make progress` error before operator stop. This was a short
verification, not proof of complete-office exploration. Refactoring checks are
software/no-motion checks; they do not by themselves establish physical safety.
