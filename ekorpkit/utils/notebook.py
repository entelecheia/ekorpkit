import logging
import sys


logger = logging.getLogger(__name__)


def _is_notebook():
    try:
        get_ipython
    except NameError:
        return False
    shell_type = get_ipython().__class__.__name__
    logger.info(f"shell type: {shell_type}")
    if shell_type == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole
    elif shell_type == "Shell":
        return True  # Google colab
    elif shell_type == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type


def _is_colab():
    is_colab = "google.colab" in sys.modules
    if is_colab:
        logger.info("Google Colab detected.")
    else:
        logger.info("Google Colab not detected.")
    return is_colab


def _get_display():
    from ipywidgets import Output

    if _is_notebook():
        return Output()
    else:
        return None


def _clear_output(wait=False):
    from IPython import display

    if _is_notebook():
        display.clear_output(wait=True)


def _display(
    *objs,
    include=None,
    exclude=None,
    metadata=None,
    transient=None,
    display_id=None,
    raw=False,
    clear=False,
    **kwargs,
):
    from IPython import display

    if _is_notebook() and objs is not None:
        return display.display(
            *objs,
            include=include,
            exclude=exclude,
            metadata=metadata,
            transient=transient,
            display_id=display_id,
            raw=raw,
            clear=clear,
            **kwargs,
        )


def _display_image(
    data=None,
    url=None,
    filename=None,
    format=None,
    embed=None,
    width=None,
    height=None,
    retina=False,
    unconfined=False,
    metadata=None,
    **kwargs,
):
    from IPython import display

    if _is_notebook():
        img = display.Image(
            data=data,
            url=url,
            filename=filename,
            format=format,
            embed=embed,
            width=width,
            height=height,
            retina=retina,
            unconfined=unconfined,
            metadata=metadata,
            **kwargs,
        )
        return display.display(img)
