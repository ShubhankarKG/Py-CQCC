import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Py-CQCC",
    version="0.1.0",
    author="Shubhankar Gupta",
    author_email="shubhankar.gupto.11@gmail.com",
    description="Python implementation of CQCC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShubhankarKG/Py-CQCC",
    project_urls={
        "Bug Tracker": "https://github.com/ShubhankarKG/Py-CQCC/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)