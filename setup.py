from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = f.read().split("\n")

setup(
    version="1.0",
    name="noisepy",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=reqs,
    author="Chengxin Jiang & Marine Denolle",
    author_email="chengxin_jiang@fas.harvard.edu & mdenolle@fas.harvard.edu",
    description="A High-performance Computing Python Package for Ambient Noise Analysis",
    license="MIT license",
    url="https://github.com/mdenolle/NoisePy",
    keywords="ambient noise, cross-correlation, seismic monitoring, velocity change " " surface wave dispersion",
    platforms="OS Independent",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)
