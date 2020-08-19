import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "fastai<2",
    "seaborn",
    "opencv-python~=4.4"
]

setuptools.setup(
    name="rpsalweaklydet",
    version="0.0.1",
    author="Renato Hermoza",
    author_email="renato.hermozaaragones@adelaide.edu.au",
    description="Region Proposals for Saliency Map Refinement for Weakly-supervised Disease Localisation and Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renato145/rpsalweaklydet",
    install_requires=requirements,
    python_requires='>=3.6',
    packages=['rpsalweaklydet'],
)
