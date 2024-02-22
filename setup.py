from setuptools import setup, find_packages
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


setup(
    name="VideoDataAnimation",
    version="0.3.1",
    long_description=long_description,
    long_description_content_type='text/markdown',
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
