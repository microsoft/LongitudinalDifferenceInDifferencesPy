from setuptools import setup


def contentsOfReadMe():
    with open('README.md') as f:
        return f.read()


setup(
    name='LongitudinalDifferenceInDifferencesPy',
    version='1.0.1',
    packages=["LongitudinalDifferenceInDifferencesPy"],
    description='Runs a difference in differences analysis in Python.',
    long_description=contentsOfReadMe(),
    long_description_content_type='text/markdown',
    author='Spencer Buja',
    author_email='csbuja@umich.edu',
    url='https://github.com/microsoft/LongitudinalDifferenceInDifferencesPy',
    license='MIT',
    platforms='ALL'
)