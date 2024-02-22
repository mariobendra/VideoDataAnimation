from setuptools import setup, find_packages

setup(
    name="VideoDataAnimation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
        "opencv-python",  # cv2
        "tqdm",
    ],
    author="Mario Bendra & Patrick Bendra",
    author_email="m.bendra22@gmail.com",
    description="A library for creating side-by-side video and data visualizations.",
    keywords="video data animation matplotlib opencv",
    url="https://github.com/mariobendra/VideoDataAnimation.git"
)