import re

from setuptools import find_namespace_packages, setup

# Ensure we match the version set in optimum/intel/version.py
try:
    filepath = "optimum/ascend/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRE = [
    "torch>=1.11",
    "optimum>=1.14.0",
    "transformers>=4.26.0",
    "datasets>=1.4.0",
    "sentencepiece",
    "scipy",
    "accelerate",  # transformers 4.29 require accelerate for PyTorch
]


TESTS_REQUIRE = []

QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241"]

EXTRAS_REQUIRE = {}

setup(
    name="optimum-ascend",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, knowledge distillation, optimization, training",
    url="https://www.intel.com",
    author="HuggingFace Inc. Special Ops Team",
    author_email="765512740@qq.com",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
