from setuptools import setup, find_packages

setup(
    name="diffusion_lm",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    description="A python project for implementing diffusion lm",
    url="https://github.com/mathematiguy/diffusion-nlp",
    author="Caleb Moses",
    author_email="calebjdmoses@gmail.com",
    include_package_data=True,
)
