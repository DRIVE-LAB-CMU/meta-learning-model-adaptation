""" Params for Carla BMW 
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


def CarlaParams(control='pwm'):
	"""	choose "pwm" for Dynamic and "acc" for Kinematic model
		see bayes_race.models for details
	"""
	lf = 0.14
	lr = 0.22 
	# lf = 1.34
	# lr = 1.3
	mass = 1265
	Iz = 2093
	Bf = 5.579
	Cf = 1.2
	Df = 16000
	Br = 7.579
	Cr = 1.2
	Dr = 16000
	Cm1 = 6.
	# Cm1 = 550*(3.45*0.919)/(0.34)
	Cm2 = 2800/0.34
	Cr0 = 50.
	Cr2 = 0.5

	max_acc = 5. 			# max acceleration [m/s^2]
	min_acc = -50. 			# max deceleration [m/s^2]
	max_pwm = 1. 			# max PWM duty cycle
	min_pwm = -1. 			# min PWM duty cycle
	max_steer = 0.35 		# max steering angle [rad]
	min_steer = -0.35 		# min steering angle [rad]
	max_steer_vel = 5. 		# max steering velocity [rad/s]

	if control is 'pwm':
		max_inputs = [max_pwm, max_steer]
		min_inputs = [min_pwm, min_steer]

		max_rates = [None, max_steer_vel]
		min_rates = [None, -max_steer_vel]

	elif control is 'acc':
		max_inputs = [max_acc, max_steer]
		min_inputs = [min_acc, min_steer]

		max_rates = [None, max_steer_vel]
		min_rates = [None, -max_steer_vel]

	else:
		raise NotImplementedError(
			'choose control as "pwm" for Dynamic model \
			and "acc" for Kinematic model'
			)

	params = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Bf': Bf,
		'Br': Br,
		'Cf': Cf,
		'Cr': Cr,
		'Df': Df,
		'Dr': Dr,
		'Cm1': Cm1,
		'Cm2': Cm2,
		'Cr0': Cr0,
		'Cr2': Cr2,
		'max_acc': max_acc,
		'min_acc': min_acc,		
		'max_pwm': max_pwm,
		'min_pwm': min_pwm,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_steer_vel': max_steer_vel,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_rates': max_rates,
		'min_rates': min_rates,
		}
	return params