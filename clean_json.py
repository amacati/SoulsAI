import json
from pathlib import Path

p = Path(__file__).parent / "saves" / "results.json"
with open(p, "r") as f:
    x = json.load(f)

x["episodes_steps"] = x["episodes_steps"][5:]
x["episodes_rewards"] = x["episodes_rewards"][5:]

with open(p, "w") as f:
    json.dump(x, f)
