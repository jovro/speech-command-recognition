from argparse import ArgumentParser

from utils.config import process_config


def main():
    parser = ArgumentParser("Generic PyTorch project")
    parser.add_argument(
        "config",
        metavar="config_json_file",
        default="None",
        help="The configuration file in JSON"
    )
    args = parser.parse_args()
    config = process_config(args.config)

    agent_module, agent_name = config.agent.split(".")
    module = __import__("agents." + agent_module)
    agent_module = getattr(module, agent_module)
    agent_class = getattr(agent_module, agent_name)
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
