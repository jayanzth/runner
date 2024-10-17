from setuptools import setup

setup(
    name='app',
    version='1.0',
    install_requires=[
        'Flask==2.2.2',
        'Werkzeug==2.2.2',
        'pandas==1.5.3',
        'nltk==3.7',
        'scikit-learn==1.2.0',
    ],
    python_requires='>=3.11',
)
