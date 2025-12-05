from setuptools import setup, find_packages

setup(
    name="mtsgamma",
    version="0.1.0",
    description="MTS–Gamma TSP solver with curvature flow ordering and C4 refinement",
    author="Reconstructed",
    packages=find_packages(),
    entry_points={"console_scripts": ["mtsgamma=cli.mtsgamma_cli:main"]},
)
