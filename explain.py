"""
Explain
- gets the config file of a trained model
- Create an agent instance
- Run the agent's evaluation and explanation given a set of data
"""
from src.agents import *
from src.utils.utils import (
    updated_config,
    dict_print,
    create_save_loc,
    set_logger,
    set_seed,
)
import wandb

if __name__ == "__main__":
    # ############# handling the bash input arguments and yaml configuration file ###############
    config = updated_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    # create saving location and document config files
    create_save_loc(config)  # config['save_dir'] gets updated here!
    save_dir = config["save_dir"]

    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["train"]["seed"])
    set_logger(save_dir, config["log_level"], "explain_local", config["comment"])

    # printing the configuration
    dict_print(config)

    # ############# Wandb setup ###############
    wandb.init(
        project="ProtoASNet", # TODO setup project name
        config=config,
        name=None if config["run_name"] == "" else config["run_name"],
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled"
        notes=config["save_dir"],  # to know where the model is saved!
    )

    # ############# agent setup ###############
    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)

    if config["explain_locally"]:
        agent.explain_local(mode=config["eval_data_type"])
    elif config["explain_globally"]:
        agent.explain_global(mode=config["eval_data_type"])

    agent.finalize()