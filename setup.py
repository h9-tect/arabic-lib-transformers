from setuptools import setup, find_packages
with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='Sarsor',
    version='0.1',
    author='Hesham Haroon',
    author_email='heshamharoon19@gmail.com',
    description='A library for processing Arabic text',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/h9-tect/arabic-lib-transformers',
    packages=['Sarsor'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['transformers', 'nltk', 'pyarabic', 'torch', 'csv', 'sentencepiece']
)

