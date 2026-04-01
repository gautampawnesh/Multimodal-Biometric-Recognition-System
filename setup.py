from setuptools import find_packages, setup

requirements = [
    "opencv-python-headless==4.8.1.78",
    "Pillow==10.0.1",
    "scikit-learn==1.3.1",
    "tqdm==4.66.1"
]

setup(
    name="biometric-recognition",
    version="0.1",
    description="Iris fingerprint dataset based biometric recognition system.",
    packages=find_packages(exclude=["dependencies", "docs"]),
    package_data={"": ["*.py", "*.yml", ".env.*", "Dockerfile_cpu", "Dockerfile_gpu"]},
    include_package_data=True,
    install_requires=requirements,
)
