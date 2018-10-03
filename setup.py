from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy==1.15.2',
    'pandas==0.20.3',
    'Keras==2.1.3',
    'scikit-learn==0.19.1',
    'tensorflow==1.11.0',
    'tensorflow_hub'
]

setup(
    name='trainer',
    version='1.0',
    author='Jake Cheng',
    author_email='jake.ct.cheng@gmail.com',
    license='JCTC Solutions Inc.',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training code to build TrashNet classifer for sorting household trash'
)
