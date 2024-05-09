from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Package metadata
setup(
    name='Topsis-Maninder-102153010',  # Name of your package
    version='0.0.1',  # Version of your package
    description='A Python module for TOPSIS analysis',  # Brief description of your package
    long_description=long_description,  # Detailed description (can include Markdown)
    long_description_content_type="text/markdown",  # Specify that the long description is in Markdown format
    author='Maninder Maan',  # Author name
    packages=find_packages(where='src'),  # Find all packages in the 'src' directory
    keywords=['topsis', 'rank', 'performance score', 'topsis analysis'],  # Keywords associated with your package
    classifiers=[
        "Programming Language :: Python :: 3",  # Specify the supported Python version
        "License :: OSI Approved :: MIT License",  # License information
        "Operating System :: OS Independent",  # Specify that the package is OS independent
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
    package_dir={'': 'src'},  # Specify the package directory
    install_requires=[
        'pandas',  # Add any dependencies required by your package
        'scikit-learn',  # Add scikit-learn for normalization
    ],
    entry_points={
        'console_scripts': [
            'topsis-maninder-102153010=topsis_module:main',  # Replace topsis_module with your actual module name
        ],
    },
)
