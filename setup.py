import os
from setuptools import setup, find_packages
import versioneer


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_about():
    about = {}
    basedir = os.path.abspath(os.path.dirname(__file__))
    with open(
        os.path.join(basedir, "ekorpkit", "conf", "about", "app", "default.yaml")
    ) as f:
        for line in f:
            k, v = line.split(": ")
            about[k.strip()] = v.strip()
    return about


def requirements():
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8"
    ) as f:
        return f.read().splitlines()


def get_extra_requires(path, add_exhaustive=True):
    import re
    from collections import defaultdict

    with open(path) as fp:
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
        if add_exhaustive:
            extra_deps["exhaustive"] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


about = get_about()
setup(
    name="ekorpkit",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author=about["author"],
    url="https://github.com/entelecheia/ekorpkit",
    description=about["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements(),
    extras_require=get_extra_requires("ekorpkit/resources/requirements-extra.yaml"),
    keywords=[],
    packages=find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        "console_scripts": ["ekorpkit=ekorpkit.cli:hydra_main", "ekorpkit-run=ekorpkit.run:main"],
    },
)
