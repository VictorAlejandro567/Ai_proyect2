from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional, Protocol, List, Callable
import random
import statistics


# ============================================================
# Actions
# ============================================================

class Action:
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    SUCK = "Suck"
    NOOP = "NoOp"


# ============================================================
# Percepts 
# ============================================================

@dataclass(frozen=True)
class Percept:
    location: Tuple[int, int]
    is_dirty: bool
    bump: bool  # True if last movement action failed (wall/obstacle)


class AgentProgram(Protocol):
    def __call__(self, percept: Percept) -> str:
        ...


# ============================================================
# Environment state
# ============================================================

@dataclass
class GridState:
    width: int
    height: int
    obstacles: Set[Tuple[int, int]]
    dirt: Dict[Tuple[int, int], bool]
    agent_location: Tuple[int, int]
    step: int = 0


# ============================================================
# Performance measure
# ============================================================

class PerformanceEvaluator:
    def __init__(self, reward_clean_per_square: int = 1, action_cost: int = 1):
        self.reward_clean_per_square = reward_clean_per_square
        self.action_cost = action_cost
        self.cumulative_score = 0

    def score_step(self, state: GridState, action: str) -> int:
        clean_squares = sum(
            1
            for x in range(state.width)
            for y in range(state.height)
            if (x, y) not in state.obstacles and not state.dirt.get((x, y), False)
        )
        delta = (self.reward_clean_per_square * clean_squares) - self.action_cost
        self.cumulative_score += delta
        return delta


# ============================================================
# Environment dynamics
# ============================================================

class GridEnvironment:
    """Grid environment with unknown geography to the agent."""

    def __init__(self, initial_state: GridState, evaluator: PerformanceEvaluator):
        self.state = initial_state
        self.evaluator = evaluator
        self.last_bump: bool = False

    def percept(self) -> Percept:
        loc = self.state.agent_location
        return Percept(
            location=loc,
            is_dirty=self.state.dirt.get(loc, False),
            bump=self.last_bump,
        )

    def step(self, action: str) -> Tuple[GridState, int]:
        x, y = self.state.agent_location
        new_loc = (x, y)
        self.last_bump = False  # reset each step

        if action == Action.UP:
            candidate = (x, y - 1)
            if self._is_free(candidate):
                new_loc = candidate
            else:
                self.last_bump = True
        elif action == Action.DOWN:
            candidate = (x, y + 1)
            if self._is_free(candidate):
                new_loc = candidate
            else:
                self.last_bump = True
        elif action == Action.LEFT:
            candidate = (x - 1, y)
            if self._is_free(candidate):
                new_loc = candidate
            else:
                self.last_bump = True
        elif action == Action.RIGHT:
            candidate = (x + 1, y)
            if self._is_free(candidate):
                new_loc = candidate
            else:
                self.last_bump = True
        elif action == Action.SUCK:
            self.state.dirt[self.state.agent_location] = False
        elif action == Action.NOOP:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        self.state.agent_location = new_loc
        self.state.step += 1
        delta = self.evaluator.score_step(self.state, action)
        return self.state, delta

    def _is_free(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        if x < 0 or y < 0 or x >= self.state.width or y >= self.state.height:
            return False
        if loc in self.state.obstacles:
            return False
        return True


# ============================================================
# World builder
# ============================================================

def build_random_grid(
    width: int,
    height: int,
    obstacle_prob: float,
    dirt_prob: float,
    seed: Optional[int] = None
) -> GridState:
    rng = random.Random(seed)
    obstacles: Set[Tuple[int, int]] = set()
    dirt: Dict[Tuple[int, int], bool] = {}

    for x in range(width):
        for y in range(height):
            if rng.random() < obstacle_prob:
                obstacles.add((x, y))
            else:
                dirt[(x, y)] = (rng.random() < dirt_prob)

    free_cells = [c for c in dirt.keys() if c not in obstacles]
    if not free_cells:
        raise ValueError("No free cells available")

    start = rng.choice(free_cells)
    return GridState(
        width=width,
        height=height,
        obstacles=obstacles,
        dirt=dirt,
        agent_location=start,
    )


# ============================================================
# Agents
# ============================================================

class SimpleReflexAgent:
    """Simple reflex: if dirty -> SUCK else always move RIGHT (blind)."""
    def __call__(self, percept: Percept) -> str:
        if percept.is_dirty:
            return Action.SUCK
        return Action.RIGHT


class RandomizedReflexAgent:
    """If dirty -> SUCK else pick a random move."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def __call__(self, percept: Percept) -> str:
        if percept.is_dirty:
            return Action.SUCK
        return self.rng.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.NOOP])


class StatefulReflexAgent:
    """
    Reflex agent with state:
    - Uses bump feedback to learn blocked directions from specific cells.
    - Tracks visited cells and prefers unvisited neighbors first.
    """

    DELTAS = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    def __init__(self):
        self.known_free: Set[Tuple[int, int]] = set()
        self.blocked: Dict[Tuple[int, int], Set[str]] = {}
        self.visited: Set[Tuple[int, int]] = set()

        self.last_location: Optional[Tuple[int, int]] = None
        self.last_action: Optional[str] = None

    def __call__(self, percept: Percept) -> str:
        loc = percept.location
        self.known_free.add(loc)
        self.visited.add(loc)

        # Learns blocked transitions from bump feedback.
        if percept.bump and self.last_location is not None and self.last_action in self.DELTAS:
            self.blocked.setdefault(self.last_location, set()).add(self.last_action)

        if percept.is_dirty:
            self.last_location = loc
            self.last_action = Action.SUCK
            return Action.SUCK

        # Prefers unvisited neighbors
        for a, (dx, dy) in self.DELTAS.items():
            if a in self.blocked.get(loc, set()):
                continue
            cand = (loc[0] + dx, loc[1] + dy)
            if cand not in self.visited:
                self.last_location = loc
                self.last_action = a
                return a

        # Otherwise, try any unblocked move
        for a in self.DELTAS:
            if a in self.blocked.get(loc, set()):
                continue
            self.last_location = loc
            self.last_action = a
            return a

        self.last_location = loc
        self.last_action = Action.NOOP
        return Action.NOOP


# ============================================================
# Experiments
# ============================================================

def run_episode(env: GridEnvironment, agent: AgentProgram, steps: int) -> int:
    for _ in range(steps):
        p = env.percept()
        a = agent(p)
        env.step(a)
    return env.evaluator.cumulative_score


def experiment_random_environments(
    num_envs: int = 50,
    width: int = 6,
    height: int = 6,
    obstacle_prob: float = 0.1,
    dirt_prob: float = 0.2,
    steps: int = 200,
):
    rng = random.Random(0)

    # use factories to create a fresh agent per environment/run
    agent_factories: List[Tuple[str, Callable[[], AgentProgram]]] = [
        ("SimpleReflex", lambda: SimpleReflexAgent()),
        ("RandomizedReflex", lambda: RandomizedReflexAgent()),
        ("StatefulReflex", lambda: StatefulReflexAgent()),
    ]

    results: Dict[str, List[float]] = {name: [] for name, _ in agent_factories}

    for _ in range(num_envs):
        state = build_random_grid(width, height, obstacle_prob, dirt_prob, seed=rng.randint(0, 10**9))

        for name, make_agent in agent_factories:
            evaluator = PerformanceEvaluator(reward_clean_per_square=1, action_cost=1)

            # Copy state so each agent faces the same environment
            state_copy = GridState(
                width=state.width,
                height=state.height,
                obstacles=set(state.obstacles),
                dirt=dict(state.dirt),
                agent_location=state.agent_location,
            )
            env = GridEnvironment(initial_state=state_copy, evaluator=evaluator)
            agent = make_agent()  # fresh instance each run

            score = run_episode(env, agent, steps)
            results[name].append(score)

    print(f"\nExperiment: {num_envs} random environments, {steps} steps each")
    for name, vals in results.items():
        print(f"{name}: mean={statistics.mean(vals):.2f}, stdev={statistics.pstdev(vals):.2f}")


def experiment_corridor_poor_for_random(length: int = 10, dirt_at_end: bool = True, steps: int = 200):
    # 1 x length corridor: start at (0,0) and dirt at far right
    width = length
    height = 1
    obstacles: Set[Tuple[int, int]] = set()
    dirt: Dict[Tuple[int, int], bool] = {(x, 0): False for x in range(width)}
    if dirt_at_end:
        dirt[(width - 1, 0)] = True

    state = GridState(width=width, height=height, obstacles=obstacles, dirt=dirt, agent_location=(0, 0))

    agents: List[Tuple[str, Callable[[], AgentProgram]]] = [
        ("RandomizedReflex", lambda: RandomizedReflexAgent(seed=1)),
        ("StatefulReflex", lambda: StatefulReflexAgent()),
        ("SimpleReflex", lambda: SimpleReflexAgent()),
    ]

    print(f"\nCorridor experiment length={length}, dirt_at_end={dirt_at_end}, steps={steps}")
    for name, make_agent in agents:
        evaluator = PerformanceEvaluator(reward_clean_per_square=1, action_cost=1)
        state_copy = GridState(
            width=state.width,
            height=state.height,
            obstacles=set(state.obstacles),
            dirt=dict(state.dirt),
            agent_location=state.agent_location,
        )
        env = GridEnvironment(initial_state=state_copy, evaluator=evaluator)
        agent = make_agent()
        score = run_episode(env, agent, steps)
        print(f"{name}: score={score}")


def main():
    print("=== Vacuum World (Exercise 2.14) experiments ===")
    experiment_random_environments(
        num_envs=40,
        width=6,
        height=6,
        obstacle_prob=0.08,
        dirt_prob=0.12,
        steps=200,
    )
    experiment_corridor_poor_for_random(length=12, dirt_at_end=True, steps=300)


if __name__ == "__main__":
    main()