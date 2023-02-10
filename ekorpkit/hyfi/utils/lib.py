import os
import subprocess
import sys
import importlib
from pathlib import Path
from .logging import getLogger
from ..io.file import is_dir, is_file


logger = getLogger()


def gitclone(url, targetdir=None, verbose=False):
    if targetdir:
        res = subprocess.run(
            ["git", "clone", url, targetdir], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
    else:
        res = subprocess.run(
            ["git", "clone", url], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
    if verbose:
        print(res)
    else:
        logger.info(res)


def pip(
    name,
    upgrade=False,
    prelease=False,
    editable=False,
    quiet=True,
    find_links=None,
    requirement=None,
    force_reinstall=False,
    verbose=False,
    **kwargs,
):
    _cmd = ["pip", "install"]
    if upgrade:
        _cmd.append("--upgrade")
    if prelease:
        _cmd.append("--pre")
    if editable:
        _cmd.append("--editable")
    if quiet:
        _cmd.append("--quiet")
    if find_links:
        _cmd += ["--find-links", find_links]
    if requirement:
        _cmd.append("--requirement")
    if force_reinstall:
        _cmd.append("--force-reinstall")
    for k in kwargs:
        k = k.replace("_", "-")
        _cmd.append(f"--{k}")
    _cmd.append(name)
    if verbose:
        logger.info(f"Installing: {' '.join(_cmd)}")
    res = subprocess.run(_cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
    if verbose:
        print(res)
    else:
        logger.info(res)


def pipi(name, verbose=False):
    res = subprocess.run(
        ["pip", "install", name], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    if verbose:
        print(res)
    else:
        logger.info(res)


def pipie(name, verbose=False):
    res = subprocess.run(
        ["git", "install", "-e", name], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    if verbose:
        print(res)
    else:
        logger.info(res)


def apti(name, verbose=False):
    res = subprocess.run(
        ["apt", "install", name], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    if verbose:
        print(res)
    else:
        logger.info(res)


def load_module_from_file(name, libpath, specname=None):
    module_path = os.path.join(libpath, name.replace(".", os.path.sep))
    if is_file(module_path + ".py"):
        module_path = module_path + ".py"
    else:
        if is_dir(module_path):
            module_path = os.path.join(module_path, "__init__.py")
        else:
            module_path = str(Path(module_path).parent / "__init__.py")

    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    if not specname:
        specname = spec.name
    sys.modules[specname] = module
    spec.loader.exec_module(module)


def ensure_import_module(name, libpath, liburi, specname=None, syspath=None):
    try:
        if specname:
            importlib.import_module(specname)
        else:
            importlib.import_module(name)
        logger.info(f"{name} imported")
    except ImportError:
        if not os.path.exists(libpath):
            logger.info(f"{libpath} not found, cloning from {liburi}")
            gitclone(liburi, libpath)
        if not syspath:
            syspath = libpath
        if syspath not in sys.path:
            sys.path.append(syspath)
        load_module_from_file(name, syspath, specname)
        specname = specname or name
        logger.info(f"{name} not imported, loading from {syspath} as {specname}")


def _dependencies(_key=None, _path=None):
    import re
    from collections import defaultdict

    if _path is None:
        _path = os.path.join(
            os.path.dirname(__file__), "resources", "requirements-extra.yaml"
        )

    with open(_path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k.strip())[0])
                for t in tags:
                    extra_deps[t].add(k.strip())

        # add tag `exhaustive` at the end
        extra_deps["exhaustive"] = set(vv for v in extra_deps.values() for vv in v)

    if _key is None or _key == "keys":
        tags = []
        for tag, deps in extra_deps.items():
            if len(deps) > 1:
                tags.append(tag)
        return tags
    else:
        return extra_deps[_key]


