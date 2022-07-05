import subprocess
import logging
import hydra
from ekorpkit import eKonf, __hydra_version_base__


logger = eKonf.getLogger(__name__)


def run_job(**args):
    command = args.get("ekorpkit") or {}
    command = [f"{k}={v}" for k, v in command.items()]
    logger.info(f"Running ekorpkit with args: {command}")
    _run_cmd(command)


def _run_cmd(command):
    command = ["ekorpkit"] + command
    proc = subprocess.Popen(
        " ".join(command),
        shell=True,
    )
    proc.wait()


def run_workflow(**args):
    _workflows = args.get("workflow") or {}
    _default_args = args.get("ekorpkit") or {}
    _default_args = [f"{k}={v}" for k, v in _default_args.items()]
    for wf_name, workflow in _workflows.items():
        logger.info("Running workflow: {}".format(wf_name))
        _jobs = workflow.get("jobs") or []
        for job in _jobs:
            job_arg = _default_args + [f"+job/{wf_name}={job}"]
            logger.info(f"Running job [{job}] with args: {job_arg}")
            _run_cmd(job_arg)


@hydra.main(config_path="conf", config_name="run", version_base=__hydra_version_base__)
def main(cfg) -> None:
    verbose = cfg.verbose

    if verbose:
        logger.info("## eKorpkit-flow Command Line Interface##")
    eKonf.load_dotenv(verbose)

    if cfg.get("verbose"):
        logger.info("## hydra configuration resolved ##")
        eKonf.pprint(cfg)

    eKonf.instantiate(cfg)


if __name__ == "__main__":
    main()
