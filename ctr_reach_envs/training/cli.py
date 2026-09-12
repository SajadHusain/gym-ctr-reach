"""Two public training commands and an explicit boundary between plants."""
import argparse
import sys

from ctr_reach_envs.paper_config import paper_configuration


def mechanics_defaults(baseline=False):
    spec = paper_configuration()
    return dict(training_protocol="mechanics-comparison-v1", total_timesteps=10000,
        task_profile="generalized_reach", episode_steps=spec["max_steps_per_episode"],
        hidden_width=spec["hidden_layers"][0], layers=len(spec["hidden_layers"]),
        learning_rate=spec["actor_lr"], buffer_size=spec["buffer_size"], batch_size=spec["batch_size"],
        learning_starts=0, wait_for_completed_batch=True,
        train_freq=spec["legacy_defaults"]["rollout_steps"],
        gradient_steps=spec["legacy_defaults"]["gradient_steps"],
        exploration_profile="paper", physics_weight=0. if baseline else .1,
        physics_final_weight=0., physics_anneal_steps=None,
        physics_integration="rl_priority", checkpoint_freq=1000,
        output_dir="runs/ddpg_mechanics" if baseline else "runs/jacobian_mechanics")


def baseline_main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if any(x in argv for x in ("-h", "--help")) and not any(x.startswith("--profile") for x in argv):
        print("Profiles: --profile paper (default, preserved reproduction); --profile mechanics (matched equilibrium baseline).\n"
              "Use --profile mechanics --help for the comparison settings.\n")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=["paper", "mechanics"], default="paper")
    args, remaining = parser.parse_known_args(argv)
    if args.profile == "paper":
        from .paper import main
        main(remaining)
    else:
        from .mechanics import main
        main(remaining, defaults=mechanics_defaults(True), baseline=True)


def guided_main(argv=None):
    from .mechanics import main
    main(argv, defaults=mechanics_defaults(False))
