import logging
from pathlib import Path

from soulsgym.core.game_state import GameState

from soulsai.distributed.server.train_node.training_node import TrainingNode
from soulsai.utils import load_config
from soulsai.core.utils import gamestate2np

logger = logging.getLogger(__name__)


def decode_sample(sample):
    experience = sample.get("sample")
    experience[0] = gamestate2np(GameState.from_dict(experience[0]))
    experience[3] = gamestate2np(GameState.from_dict(experience[3]))
    return experience


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    training_node = TrainingNode(config, decode_sample)
    if config.fill_buffer:
        training_node.fill_buffer()
    training_node.run()
