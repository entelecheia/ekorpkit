import os
from .logging import getLogger
from .env import set_osenv


logger = getLogger(__name__)


def mount_google_drive(
    workspace=None,
    project=None,
    mountpoint="/content/drive",
    force_remount=False,
    timeout_ms=120000,
):
    try:
        from google.colab import drive

        drive.mount(mountpoint, force_remount=force_remount, timeout_ms=timeout_ms)

        if isinstance(workspace, str):
            if not workspace.startswith(os.path.sep) and not workspace.startswith(".."):
                workspace = os.path.join(mountpoint, workspace)
            set_osenv("EKORPKIT_WORKSPACE_ROOT", workspace)
            logger.info(f"Setting EKORPKIT_WORKSPACE_ROOT to {workspace}")
        if isinstance(project, str):
            set_osenv("EKORPKIT_PROJECT_NAME", project)
            logger.info(f"Setting EKORPKIT_PROJECT_NAME to {project}")
    except ImportError:
        logger.warning("Google Colab not detected.")
