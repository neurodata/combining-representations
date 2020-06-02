from setuptools import setup, find_packages

requirements = [
    "numpy>=1.8.1",
]

with open("README.md", "r") as readme_file:
    readme = readme_file.read()



setup(
    name="combining-representations",
    version="0.0.1",
    author="Hayden S. Helm",
    author_email="haydenshelm@gmail.com",
    description="A package to implement and extend the method desribed in 'Learning to rank via combining representations'",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/combining-representations/",
    packages=find_packages(),
    install_requires=requirements,
)