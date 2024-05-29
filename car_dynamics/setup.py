from setuptools import setup

setup(
    name='car_dynamics',
    version='0.0.0',
    author='Wenli Xiao',
    author_email='wxiao2@andrew.cmu.edu',
    description='A package for simplify RC Car dynamics application.',
    packages=['car_dynamics','bayes_race'],
    install_requires=[
        # 'termcolor',
        'rich',
        'scipy',
        'pandas',
        # 'gym',
		# 'cvxpy',
		# 'casadi',
		# 'botorch==0.1.4',
		# 'gpytorch==0.3.6',
		# 'matplotlib==3.1.2',
		# 'scikit-learn==0.22.2.post1',
		# 'tikzplotlib',
    ],
)
