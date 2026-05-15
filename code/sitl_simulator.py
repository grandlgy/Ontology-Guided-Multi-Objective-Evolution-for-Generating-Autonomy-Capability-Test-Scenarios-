"""
Lightweight 6-DoF UAV simulation testbed for scenario reproduction.

This is a custom Python-based simulator used as a substitute for full
AirSim+PX4 SITL when full Unreal Engine integration is unavailable.
It implements:
  - Simplified quadrotor / fixed-wing point-mass dynamics
  - Wind disturbance, GPS noise, battery depletion, comm jitter effects
  - PD waypoint-following controller
  - Failure detection (collision, runaway divergence, battery dry, mission timeout)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class FlightState:
    pos: np.ndarray              # 3D position (m)
    vel: np.ndarray              # 3D velocity (m/s)
    battery: float               # %
    t: float                     # simulated time (s)
    gps_bias: np.ndarray         # current GPS bias (m)
    alive: bool = True
    failure_reason: str = ""


@dataclass
class SimLog:
    t: List[float] = field(default_factory=list)
    pos: List[np.ndarray] = field(default_factory=list)
    vel: List[np.ndarray] = field(default_factory=list)
    gps_err: List[float] = field(default_factory=list)
    battery: List[float] = field(default_factory=list)
    failure_reason: str = ""
    success: bool = True
    total_time: float = 0.0
    min_obstacle_dist: float = 1e9


def _waypoints_for_mission(mis_type: int, init_pos: np.ndarray,
                           altitude: float) -> List[np.ndarray]:
    """Mission-dependent waypoint sequence."""
    if mis_type == 0:        # route patrol
        return [init_pos + np.array([0, 0, altitude]),
                init_pos + np.array([200, 0, altitude]),
                init_pos + np.array([200, 200, altitude]),
                init_pos + np.array([0, 200, altitude]),
                init_pos + np.array([0, 0, altitude])]
    elif mis_type == 1:      # hover (fixed-wing prohibited by ontology)
        return [init_pos + np.array([0, 0, altitude])]
    elif mis_type == 2:      # cooperative search (zig-zag)
        wps = [init_pos + np.array([0, 0, altitude])]
        for k in range(4):
            wps.append(init_pos + np.array([150*(k+1), 50*((-1)**k+1), altitude]))
        return wps
    else:                     # emergency avoidance: aggressive turn
        return [init_pos + np.array([0, 0, altitude]),
                init_pos + np.array([80, 0, altitude]),
                init_pos + np.array([80, 80, altitude]),
                init_pos + np.array([-30, 80, altitude])]


def _generate_obstacles(scenario: dict, rng: np.random.Generator) -> np.ndarray:
    """Sample obstacle positions based on obstacle_density and terrain."""
    density = scenario.get('tgt_obstacle_density', 0.5)
    terr = int(scenario.get('env_terrain', 0))
    # More obstacles in city (1) and forest (3)
    multiplier = {0: 0.4, 1: 1.4, 2: 0.8, 3: 1.2}[terr]
    n = int(round(40 * density * multiplier))
    n = max(0, min(n, 80))
    if n == 0:
        return np.zeros((0, 3))
    obs = rng.uniform([-50, -50, 5], [250, 250, 200], size=(n, 3))
    return obs


def simulate_scenario(scenario: dict, dt: float = 0.05,
                      max_t: float = 240.0,
                      seed: int = 0,
                      return_log: bool = True) -> SimLog:
    """
    Execute one mission under a concrete scenario and return the flight log.
    A binary 'success' label is produced based on physical failure conditions.
    """
    rng = np.random.default_rng(seed)
    log = SimLog()

    # Unpack scenario
    wind_speed = scenario['env_wind_speed']
    visibility = scenario['env_visibility']
    illum = int(scenario['env_illumination'])
    em = scenario['env_em_interference']
    precip = scenario['env_precipitation']
    plat = int(scenario['tgt_platform_type'])
    obs_density = scenario['tgt_obstacle_density']
    mis_type = int(scenario['mis_type'])
    altitude = scenario['mis_altitude']
    target_vel = max(scenario['mis_velocity'], 1.0)
    gps_loss = scenario['dist_gps_loss']
    comm_jitter = scenario['dist_comm_jitter']
    payload_change = scenario['dist_payload_change']
    bat0 = scenario['dist_battery_level']

    # Wind vector (random direction, magnitude = wind_speed plus gusts)
    wind_dir = rng.uniform(0, 2 * np.pi)
    wind_base = np.array([wind_speed * np.cos(wind_dir),
                          wind_speed * np.sin(wind_dir),
                          0.0])

    # Obstacles
    obstacles = _generate_obstacles(scenario, rng)

    # Waypoints
    init_pos = np.zeros(3)
    waypoints = _waypoints_for_mission(mis_type, init_pos, altitude)
    wp_idx = 0

    state = FlightState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.zeros(3),
        battery=bat0,
        t=0.0,
        gps_bias=np.zeros(3),
    )

    # GPS bias drift: stronger under high em or gps_loss
    gps_bias_sigma = 1.0 + 15.0 * em + 30.0 * gps_loss

    # Mass / aerodynamic surrogate
    mass = 1.5 * (1 + payload_change)
    drag = 0.25 + 0.10 * (precip / 20.0)

    # Controller gains
    kp = 0.9
    kd = 1.4

    # Battery drain rate: nominal + extra under heavy payload and high wind
    drain_per_s = 100.0 / (max(60.0 * (1 - 0.5*payload_change), 25.0) * 10.0)

    steps = int(max_t / dt)
    last_wp_reach_t = 0.0

    for step in range(steps):
        # ---- Update GPS bias as random walk ----
        state.gps_bias += rng.normal(0, gps_bias_sigma * np.sqrt(dt) * 0.3, 3)
        # Clip GPS bias to avoid arbitrarily large drift in short time
        bias_norm = np.linalg.norm(state.gps_bias)
        if bias_norm > 100:
            state.gps_bias *= 100 / bias_norm

        # Effective measured position (controller sees this)
        gps_drop = rng.random() < gps_loss * 0.05  # intermittent dropouts
        if gps_drop:
            measured_pos = state.pos + state.gps_bias + rng.normal(0, 5, 3)
        else:
            measured_pos = state.pos + state.gps_bias

        # ---- Mission progression ----
        if wp_idx >= len(waypoints):
            log.success = True
            break

        target = waypoints[wp_idx]
        err = target - measured_pos
        dist_to_wp = np.linalg.norm(err)
        if dist_to_wp < 4.0:
            wp_idx += 1
            last_wp_reach_t = state.t
            continue

        # ---- Controller (PD with comm jitter as control delay) ----
        # Comm jitter applied as effective gain reduction
        gain_factor = max(0.3, 1.0 - comm_jitter / 1500.0)
        desired_vel = err / max(dist_to_wp, 1e-3) * target_vel
        accel_cmd = (kp * (desired_vel - state.vel) - kd * 0.0) * gain_factor

        # Saturate
        accel_norm = np.linalg.norm(accel_cmd)
        if accel_norm > 6.0:
            accel_cmd = accel_cmd / accel_norm * 6.0

        # ---- Add wind gust ----
        gust = rng.normal(0, wind_speed * 0.15, 3)
        gust[2] *= 0.3   # less vertical gust
        wind_force = (wind_base + gust) * 0.3

        # ---- Dynamics ----
        net_accel = accel_cmd - drag * state.vel + wind_force / mass
        net_accel[2] -= 0.0  # gravity assumed compensated by altitude control loop
        state.vel = state.vel + net_accel * dt
        state.pos = state.pos + state.vel * dt
        state.t += dt
        state.battery -= drain_per_s * dt

        # ---- Failure checks ----
        # 1. Collision (only meaningful when illumination/visibility limit perception)
        # Visual perception capability degrades at night with low visibility
        perception_strength = 1.0
        if illum <= 1:
            perception_strength *= min(visibility, 2000) / 2000.0
        # Heavy precipitation also degrades perception
        perception_strength *= max(0.3, 1.0 - precip / 20.0)
        # Compute collision distance
        if obstacles.shape[0] > 0:
            dists = np.linalg.norm(obstacles - state.pos[None, :], axis=1)
            min_d = float(dists.min())
            if min_d < log.min_obstacle_dist:
                log.min_obstacle_dist = min_d
            # Safety radius increased when perception is poor
            safety_radius = 3.5 / max(perception_strength, 0.2)
            if min_d < safety_radius:
                state.alive = False
                state.failure_reason = "collision"
                log.success = False

        # 2. Battery dry
        if state.battery <= 0:
            state.alive = False
            state.failure_reason = "battery_exhausted"
            log.success = False

        # 3. Runaway (velocity blows up)
        if np.linalg.norm(state.vel) > 35:
            state.alive = False
            state.failure_reason = "control_divergence"
            log.success = False

        # 4. Mission timeout for hover task or general
        if state.t - last_wp_reach_t > 90 and mis_type != 1:
            state.alive = False
            state.failure_reason = "mission_timeout"
            log.success = False

        # 5. Hover task done after staying near target for 30 s
        if mis_type == 1 and state.t > 30 and dist_to_wp < 5.0:
            wp_idx += 1

        # 6. Out of bounds
        if abs(state.pos[0]) > 600 or abs(state.pos[1]) > 600 or state.pos[2] < -5:
            state.alive = False
            state.failure_reason = "out_of_bounds"
            log.success = False

        # Log
        if return_log and step % 4 == 0:
            log.t.append(state.t)
            log.pos.append(state.pos.copy())
            log.vel.append(state.vel.copy())
            log.gps_err.append(float(np.linalg.norm(state.gps_bias)))
            log.battery.append(state.battery)

        if not state.alive:
            break

    log.total_time = state.t
    log.failure_reason = state.failure_reason
    if log.success and wp_idx >= len(waypoints):
        log.failure_reason = "completed"
    elif not state.alive:
        log.success = False
    elif state.t >= max_t - dt:
        log.success = False
        log.failure_reason = "timeout"

    return log


def batch_simulate(scenarios: List[dict], n_runs: int = 10,
                   base_seed: int = 100) -> List[dict]:
    """Run each scenario n_runs times, return aggregated outcome statistics."""
    results = []
    for sc_idx, sc in enumerate(scenarios):
        failures = 0
        reasons = []
        times = []
        for r in range(n_runs):
            log = simulate_scenario(sc, seed=base_seed + sc_idx*1000 + r,
                                    return_log=False)
            if not log.success:
                failures += 1
            reasons.append(log.failure_reason)
            times.append(log.total_time)
        results.append({
            'scenario_idx': sc_idx,
            'failure_rate': failures / n_runs,
            'failure_reasons': reasons,
            'mean_time': float(np.mean(times)),
        })
    return results


if __name__ == "__main__":
    # Quick smoke test with a benign scenario
    benign = {
        'env_wind_speed': 2.0, 'env_visibility': 4000, 'env_illumination': 3,
        'env_terrain': 0, 'env_em_interference': 0.1, 'env_precipitation': 0.0,
        'tgt_platform_type': 0, 'tgt_rcs_dbsm': -10, 'tgt_obstacle_density': 0.1,
        'mis_type': 0, 'mis_altitude': 80, 'mis_velocity': 8.0,
        'dist_gps_loss': 0.05, 'dist_comm_jitter': 50, 'dist_payload_change': 0.0,
        'dist_battery_level': 85, 'scn_init_distance': 200,
    }
    log = simulate_scenario(benign, seed=42)
    print(f"Benign: success={log.success}, t={log.total_time:.1f}s, reason={log.failure_reason}")
