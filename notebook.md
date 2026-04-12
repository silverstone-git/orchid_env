# Module 4: Build a Word Game Environment

Build a letter-guessing (Hangman-style) environment from scratch using the OpenEnv pattern.

**Time:** ~30 min · **Difficulty:** Intermediate · **GPU:** Not required


```python
!pip install -q openenv-core
!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true

import sys, os
repo = os.path.abspath('OpenEnv')
for p in [repo, os.path.join(repo, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)
print("Setup complete!")
```

## 1. Define the Types

Every OpenEnv environment starts with its data contracts: what actions can you take, what do you observe, what metadata exists?


```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# These would normally go in models.py

@dataclass
class WordGameAction:
    """Player guesses a single letter."""
    guess: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WordGameObservation:
    """What the player sees after each guess."""
    done: bool
    reward: Optional[float]
    masked_word: str            # e.g., "p_th_n"
    guessed_letters: List[str]  # All letters tried
    attempts_remaining: int
    message: str                # Feedback text
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WordGameState:
    """Episode metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    target_word: str = ""
    max_attempts: int = 6

print("Types defined: WordGameAction, WordGameObservation, WordGameState")
```

## 2. Implement the Environment

The environment implements three methods: `reset()`, `step()`, and `state`. This is where the game logic lives.


```python
import random
import uuid

WORDS = [
    "python", "neural", "tensor", "matrix", "vector",
    "kernel", "lambda", "signal", "binary", "cipher",
    "model", "layer", "epoch", "batch", "token",
]

class WordGameEnvironment:
    """A letter-guessing game environment following the OpenEnv pattern."""

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = 6

    def reset(self) -> WordGameObservation:
        """Start a new episode with a random word."""
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = 10
        self._state = WordGameState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=10,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction) -> WordGameObservation:
        """Process a letter guess."""
        letter = action.guess.lower().strip()
        self._state.step_count += 1

        # Already guessed?
        if letter in self._guessed:
            return WordGameObservation(
                done=False,
                reward=0.0,
                masked_word=self._mask(),
                guessed_letters=sorted(self._guessed),
                attempts_remaining=self._remaining,
                message=f"Already guessed '{letter}'. Try another.",
            )

        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        """Show guessed letters, hide the rest."""
        return "".join(c if c in self._guessed else "_" for c in self._target)

print("WordGameEnvironment defined.")
```

## 3. Test the Environment Directly

Before wiring up HTTP, test the pure game logic.


```python
env = WordGameEnvironment()
obs = env.reset()
print(f"Word: {obs.masked_word} ({len(obs.masked_word)} letters)")
print(f"Message: {obs.message}")
print(f"Attempts: {obs.attempts_remaining}")
print()

# Play with common letters
for letter in ["e", "a", "t", "n", "o", "r", "s", "i", "l"]:
    if obs.done:
        break
    obs = env.step(WordGameAction(guess=letter))
    print(f"  Guess '{letter}': {obs.masked_word}  ({obs.message})")

print(f"\nFinal: reward={obs.reward}, done={obs.done}")
print(f"State: episode={env.state.episode_id[:8]}..., steps={env.state.step_count}")
```

## 4. Write Policies

Let's write two policies and compare them.


```python
import string

class RandomLetterPolicy:
    """Guess random unused letters."""
    name = "Random"

    def select_action(self, obs: WordGameObservation) -> WordGameAction:
        available = [c for c in string.ascii_lowercase if c not in obs.guessed_letters]
        return WordGameAction(guess=random.choice(available))


class FrequencyPolicy:
    """Guess by English letter frequency."""
    name = "Frequency"
    FREQ_ORDER = "etaoinshrdlcumwfgypbvkjxqz"

    def select_action(self, obs: WordGameObservation) -> WordGameAction:
        for letter in self.FREQ_ORDER:
            if letter not in obs.guessed_letters:
                return WordGameAction(guess=letter)
        return WordGameAction(guess="a")  # fallback


def evaluate(env, policy, episodes=100):
    wins = 0
    total_steps = 0
    for _ in range(episodes):
        obs = env.reset()
        while not obs.done:
            action = policy.select_action(obs)
            obs = env.step(action)
        if obs.reward and obs.reward > 0:
            wins += 1
        total_steps += env.state.step_count
    return wins / episodes, total_steps / episodes


env = WordGameEnvironment()

for policy in [RandomLetterPolicy(), FrequencyPolicy()]:
    win_rate, avg_steps = evaluate(env, policy)
    print(f"{policy.name:15s} — Win rate: {win_rate*100:.1f}%, Avg steps: {avg_steps:.1f}")
```

Frequency should significantly outperform random. With technical vocabulary and individual letter guessing, both win rates are modest — but Frequency is typically 5–10× better than Random. Increase `max_attempts` in `WordGameEnvironment` (e.g. to 15) to see higher absolute win rates.

## 5. Wire Up FastAPI

In a real deployment, you'd create `server/app.py` with:

```python
from openenv.core.env_server import create_fastapi_app
from environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment)
```

That single call creates all endpoints: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/web`, `/docs`.

Let's simulate the server locally to demonstrate the full stack.


```python
# Write the environment files to disk for deployment
import os

os.makedirs('word_game/server', exist_ok=True)

# models.py — uses Pydantic (Action, Observation, State are Pydantic BaseModel subclasses)
models_code = '''
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class WordGameAction(Action):
    """Player guesses a single letter."""
    guess: str


class WordGameObservation(Observation):
    """What the player sees after each guess.

    Note: done and reward are inherited from Observation.
    """
    masked_word: str            # e.g. "p_th_n"
    guessed_letters: List[str]  # All letters tried
    attempts_remaining: int
    message: str                # Feedback text


class WordGameState(State):
    """Episode metadata.

    Note: episode_id and step_count are inherited from State.
    """
    target_word: str = ""
    max_attempts: int = 6
'''

with open('word_game/models.py', 'w') as f:
    f.write(models_code)

# client.py — uses EnvClient (WebSocket-based)
client_code = '''
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import WordGameAction, WordGameObservation, WordGameState


class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action: WordGameAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=WordGameObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                masked_word=obs_data.get("masked_word", ""),
                guessed_letters=obs_data.get("guessed_letters", []),
                attempts_remaining=obs_data.get("attempts_remaining", 0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordGameState:
        return WordGameState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target_word=payload.get("target_word", ""),
            max_attempts=payload.get("max_attempts", 6),
        )
'''

with open('word_game/client.py', 'w') as f:
    f.write(client_code)

# server/app.py
app_code = '''
from openenv.core.env_server import create_fastapi_app
from ..models import WordGameAction, WordGameObservation
from .environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment, WordGameAction, WordGameObservation)
'''

with open('word_game/server/app.py', 'w') as f:
    f.write(app_code)

print('Created word_game/models.py  (Pydantic models)')
print('Created word_game/client.py  (EnvClient subclass)')
print('Created word_game/server/app.py')
print()
print('Next steps:')
print('  1. Add server/environment.py with WordGameEnvironment class')
print('  2. Test locally: uvicorn word_game.server.app:app --reload')
print('  3. Deploy: openenv push --repo-id username/word-game')
```

## 6. The Client

The client translates between your typed models and JSON over the wire. Three methods:

```python
class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action):
        return {"guess": action.guess}

    def _parse_result(self, payload):
        return StepResult(
            observation=WordGameObservation(**payload),
            reward=payload.get("reward", 0),
            done=payload["done"],
        )

    def _parse_state(self, payload):
        return WordGameState(**payload)
```

Users of your environment would then write:

```python
from word_game import WordGameEnv, WordGameAction

with WordGameEnv(base_url="https://username-word-game.hf.space").sync() as env:
    result = env.reset()
    result = env.step(WordGameAction(guess="e"))
    print(result.observation.masked_word)
```

## 7. Scaffold with `openenv init`

Instead of writing everything by hand, use the CLI:

```bash
openenv init word_game
cd word_game
# Edit models.py, server/environment.py, client.py
uv run server           # Test locally
openenv push             # Deploy to HF Spaces
```

This creates the full directory structure. You fill in your types and game logic.

## Summary

You built a complete OpenEnv environment:

| File | What it does | Lines of code |
|------|-------------|---------------|
| `models.py` | Action, Observation, State types | ~30 |
| `server/environment.py` | Game logic (reset, step, state) | ~60 |
| `client.py` | HTTP client (3 parsing methods) | ~25 |
| `server/app.py` | FastAPI wiring | ~3 |

The pattern is always the same: **types → server logic → client → container**.

**Next:** [Module 5](../module-5/README.md) — Training a model to play games with GRPO.
