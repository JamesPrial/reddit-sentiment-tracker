"""Setup script for Reddit Sentiment Tracker"""

from setuptools import setup, find_packages

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reddit-sentiment-tracker",
    version="0.1.0",
    author="James Prial",
    author_email="prialjames17@gmail.com",
    description="A comprehensive Reddit sentiment tracking and analysis system",
    license="AGPL-3.0-or-later",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/reddit-sentiment-tracker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "reddit-tracker=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.sql", "*.md"],
    },
)