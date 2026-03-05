from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Callable, Optional, Tuple
import random
import statistics

"""
PEAS specification for the vacuum world simulator:

P (Performance measure): cumulative score where each time step the
    environment gives +1 per clean square and subtracts 1 per action.
    The simulator tracks `cumulative_score` in `PerformanceEvaluator`.

E (Environment): explicit `VacuumEnvironment` with a `VacuumState` that
    contains locations (graph adjacency), dirt distribution, agent location,
    and step counter. The environment evolves via its transition (action
    handlers) and does not depend on the agent implementation.

A (Actuators): modular `action_handlers` mapping action names (e.g.,
    Left/Right/Suck/NoOp) to handler callables. New actions can be added by
    adding entries to this mapping.

S (Sensors): modular `SensorModel` protocol. Example implementations include
    `LocalDirtSensor` and `NoisyLocalDirtSensor`. Percepts are produced by
    calling `env.percept()` which delegates to the sensor.
"""



# ============================================================
# Actuators (Actions) - extensible
# ============================================================

class Action:
    LEFT = "Left"
    RIGHT = "Right"
    SUCK = "Suck"
    NOOP = "NoOp"


# ============================================================
# Sensors (Percepts) - swappable / modular
# ============================================================

@dataclass(frozen=True)
class Percept:
    location: str
    is_dirty: bool


class SensorModel(Protocol):
    def sense(self, env: "VacuumEnvironment") -> Percept:
        ...


class LocalDirtSensor:
    """Percept = (current location, dirt status at current location)."""
    def sense(self, env: "VacuumEnvironment") -> Percept:
        loc = env.state.agent_location
        return Percept(location=loc, is_dirty=env.state.dirt.get(loc, False))


class NoisyLocalDirtSensor:
    """Example alternative sensor: flips dirt reading with probability flip_prob."""
    def __init__(self, flip_prob: float = 0.1):
        self.flip_prob = flip_prob

    def sense(self, env: "VacuumEnvironment") -> Percept:
        loc = env.state.agent_location
        true_dirty = env.state.dirt.get(loc, False)
        observed_dirty = (not true_dirty) if random.random() < self.flip_prob else true_dirty
        return Percept(location=loc, is_dirty=observed_dirty)


# ============================================================
# Environment State (size/shape extensible)
# ============================================================

@dataclass
class VacuumState:
    locations: List[str]                 # e.g., ["A","B"] or more
    adjacency: Dict[str, List[str]]      # shape as a graph
    agent_location: str
    dirt: Dict[str, bool]                # dirt distribution
    step: int = 0


# ============================================================
# Performance Measure (tracks cumulative performance)
# ============================================================

class PerformanceEvaluator:
    """
    Example performance function from the textbook-style description:
      +1 per clean square per time step
      -1 per action per time step
    Easily adjustable.
    """

    def __init__(self, reward_clean_per_square: int = 1, action_cost: int = 1):
        self.reward_clean_per_square = reward_clean_per_square
        self.action_cost = action_cost
        self.cumulative_score = 0

    def score_step(self, state: VacuumState, action: str) -> int:
        clean_squares = sum(1 for loc in state.locations if not state.dirt.get(loc, False))
        delta = (self.reward_clean_per_square * clean_squares) - self.action_cost
        self.cumulative_score += delta
        return delta


# ============================================================
# Environment Dynamics (transition function) - independent
# ============================================================

class VacuumEnvironment:
    """
    Modular, performance-measuring simulator for the vacuum world.

    Key property: agent program is separate.
    The environment provides percepts and applies actions via a transition function.
    """

    def __init__(
        self,
        initial_state: VacuumState,
        sensor: SensorModel,
        evaluator: PerformanceEvaluator,
        action_handlers: Optional[Dict[str, Callable[["VacuumEnvironment"], None]]] = None,
    ):
        self.state = initial_state
        self.sensor = sensor
        self.evaluator = evaluator

        # Modular actuators: action -> handler mapping (easy to extend)
        self.action_handlers = action_handlers or {
            Action.LEFT: self._handle_left,
            Action.RIGHT: self._handle_right,
            Action.SUCK: self._handle_suck,
            Action.NOOP: self._handle_noop,
        }

    def percept(self) -> Percept:
        """Generate a percept for the agent (modular sensor model)."""
        return self.sensor.sense(self)

    def step(self, action: str) -> Tuple[VacuumState, int]:
        """
        Apply one action using the environment transition function.
        Returns (new_state, performance_delta_for_this_step).
        """
        if action not in self.action_handlers:
            raise ValueError(f"Unknown action: {action}")

        # Apply action (environment changes itself)
        self.action_handlers[action](self)

        # Advance time
        self.state.step += 1

        # Update performance (environment tracks it)
        delta = self.evaluator.score_step(self.state, action)
        return self.state, delta

    def run(self, agent_program: "AgentProgram", steps: int) -> int:
        """
        Run a simulation episode for a fixed number of steps.
        Returns cumulative performance score.
        """
        for _ in range(steps):
            p = self.percept()
            a = agent_program(p)      # agent maps percept -> action
            self.step(a)              # environment applies action
        return self.evaluator.cumulative_score

    # ---------- Default action handlers ----------
    # Written to work for any adjacency graph.

    def _handle_left(self, _: "VacuumEnvironment") -> None:
        neighbors = self.state.adjacency.get(self.state.agent_location, [])
        if neighbors:
            self.state.agent_location = neighbors[0]

    def _handle_right(self, _: "VacuumEnvironment") -> None:
        neighbors = self.state.adjacency.get(self.state.agent_location, [])
        if neighbors:
            self.state.agent_location = neighbors[-1]

    def _handle_suck(self, _: "VacuumEnvironment") -> None:
        self.state.dirt[self.state.agent_location] = False

    def _handle_noop(self, _: "VacuumEnvironment") -> None:
        pass


# ============================================================
# Agent program interface (kept separate from environment)
# ============================================================

class AgentProgram(Protocol):
    def __call__(self, percept: Percept) -> str:
        ...


# Simple demo agent (optional): classic reflex
class SimpleReflexAgent:
    def __call__(self, percept: Percept) -> str:
        if percept.is_dirty:
            return Action.SUCK
        return Action.RIGHT


# Random agent: chooses uniformly among available primitive actions.
class RandomAgent:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def __call__(self, percept: Percept) -> str:
        return self.rng.choice([Action.SUCK, Action.LEFT, Action.RIGHT, Action.NOOP])


# ============================================================
# World builder utilities (size/shape/dirt placement configurable)
# ============================================================

def build_two_location_world(
    dirt_A: bool = True,
    dirt_B: bool = True,
    start: str = "A",
) -> VacuumState:
    locations = ["A", "B"]
    adjacency = {"A": ["B"], "B": ["A"]}  # A <-> B
    dirt = {"A": dirt_A, "B": dirt_B}
    return VacuumState(locations=locations, adjacency=adjacency, agent_location=start, dirt=dirt)


def build_line_world(n: int, dirt_prob: float = 0.5, start_index: int = 0, seed: int = 0) -> VacuumState:
    """
    Example extensibility: a line of n locations (L0-L{n-1}).
    'Left' moves to previous neighbor, 'Right' moves to next neighbor.
    Dirt placement is random with probability dirt_prob.
    """
    rng = random.Random(seed)
    locations = [f"L{i}" for i in range(n)]
    adjacency: Dict[str, List[str]] = {}
    for i, loc in enumerate(locations):
        neighbors = []
        if i - 1 >= 0:
            neighbors.append(locations[i - 1])
        if i + 1 < n:
            neighbors.append(locations[i + 1])
        adjacency[loc] = neighbors

    dirt = {loc: (rng.random() < dirt_prob) for loc in locations}
    start = locations[max(0, min(start_index, n - 1))]
    return VacuumState(locations=locations, adjacency=adjacency, agent_location=start, dirt=dirt)


# ============================================================
# Demo main (shows it runs + prints cumulative performance)
# ============================================================

def main() -> None:
    # Minimum required world: two locations
    state = build_two_location_world(dirt_A=True, dirt_B=False, start="A")

    # Modular components:
    sensor = LocalDirtSensor()  # swap to NoisyLocalDirtSensor(...) if you want
    evaluator = PerformanceEvaluator(reward_clean_per_square=1, action_cost=1)

    env = VacuumEnvironment(initial_state=state, sensor=sensor, evaluator=evaluator)

    # Demo run with a simple reflex agent
    agent = SimpleReflexAgent()
    total = env.run(agent, steps=10)

    print("=== Vacuum World Simulator (Exercise 2.11) ===")
    print("Final state:", env.state)
    print("Cumulative performance (simple reflex, single run):", total)

    # Experimental requirement: run at least two different agents (reflex vs random)
    def run_trials(agent_factory: Callable[[], AgentProgram], trials: int = 100, steps: int = 30):
        results = []
        for t in range(trials):
            # Randomize initial dirt distribution for each trial (two-location world)
            dirtA = random.choice([True, False])
            dirtB = random.choice([True, False])
            state_t = build_two_location_world(dirt_A=dirtA, dirt_B=dirtB, start=random.choice(["A", "B"]))
            sensor_t = LocalDirtSensor()
            evaluator_t = PerformanceEvaluator(reward_clean_per_square=1, action_cost=1)
            env_t = VacuumEnvironment(initial_state=state_t, sensor=sensor_t, evaluator=evaluator_t)
            agent_t = agent_factory()
            score = env_t.run(agent_t, steps=steps)
            results.append(score)
        return results

    trials = 100
    steps = 30
    reflex_results = run_trials(lambda: SimpleReflexAgent(), trials=trials, steps=steps)
    random_results = run_trials(lambda: RandomAgent(), trials=trials, steps=steps)

    print(f"\nExperimental comparison over {trials} trials, {steps} steps each:")
    print(f"SimpleReflexAgent: mean={statistics.mean(reflex_results):.2f}, stdev={statistics.pstdev(reflex_results):.2f}")
    print(f"RandomAgent:      mean={statistics.mean(random_results):.2f}, stdev={statistics.pstdev(random_results):.2f}")

if __name__ == "__main__":
    main()