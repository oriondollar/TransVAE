import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
	name='transvae',
    version='0.4',
	description='A package for training and analyzing attention VAEs for molecular design.',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/oriondollar/TransVAE',
    author='Orion Dollar',
    author_email='orion.dollar@gmail.com',
	license='MIT License',
	packages=find_packages(),
	install_requires=[
		'numpy',
        'torch',
		'pandas',
		'seaborn',
		'matplotlib'],
	classifiers=[
    	'Intended Audience :: Science/Research',
   		'License :: OSI Approved :: MIT License',
    	'Operating System :: POSIX :: Linux',
    	'Programming Language :: Python :: 3.7',
		]
)
