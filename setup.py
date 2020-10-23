# setup.py: install script for rl_nexus
import sys, os.path
'''
to install rl_nexus and its dependencies for development work,
run this cmd from dilbert directory:
    pip install -e .
'''
from setuptools import setup

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

# place holder for necessory package-level installs
install_requires=[]

def read_reqs(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), 'r').readlines()

extras = {
    # 'atari': read_reqs('rl_nexus/components/environments/Atari_Env/requirements.txt'),
    # 'procgen': read_reqs('rl_nexus/components/environments/Procgen_Env/requirements.txt'),
    # 'podworld': read_reqs('rl_nexus/components/environments/Pod_World_Env/requirements.txt'),
}
# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(
    name="Off-Policy",
    version="0.0.1",
    description="Off-Policy Evaluation and Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = "Hoang M. Le",
    author_email="hoang.le@microsoft.com",
    url = "https://github.com/hoangminhle/Off-Policy",
    packages=['rl_nexus'],
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
)
