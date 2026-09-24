import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "fva_car" / "speed_controller.py"
SPEC = importlib.util.spec_from_file_location("real_speed_controller", MODULE_PATH)
speed_controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speed_controller)
SpeedPI = speed_controller.SpeedPI


class SpeedControllerTests(unittest.TestCase):
	def test_integral_uses_real_elapsed_time(self):
		controller = SpeedPI(kp=2.0, ki=3.0, output_filter_time=0.0)
		controller.update(0.1, 0.02, -1.0, 1.0)
		output = controller.update(0.1, 0.07, -1.0, 1.0)
		self.assertAlmostEqual(output, 0.206)
		self.assertAlmostEqual(controller.integral, 0.009)

	def test_saturation_prevents_windup(self):
		for direction in (-1.0, 1.0):
			controller = SpeedPI(output_filter_time=0.0)
			for _ in range(1000):
				output = controller.update(direction * 10.0, 0.02, -1.0, 1.0)
				self.assertEqual(output, direction)
			self.assertEqual(controller.integral, 0.0)

	def test_integral_unwinds_from_saturation(self):
		controller = SpeedPI(kp=1.0, ki=1.0)
		controller.integral = 2.0
		for _ in range(20):
			controller.update(-0.2, 0.1, -1.0, 1.0)
		self.assertAlmostEqual(controller.integral, 1.6)

	def test_invalid_configuration_is_rejected(self):
		with self.assertRaises(ValueError):
			SpeedPI(kp=-1.0)
		controller = SpeedPI()
		with self.assertRaises(ValueError):
			controller.update(0.0, -0.01, -1.0, 1.0)
		with self.assertRaises(ValueError):
			controller.update(0.0, 0.01, 1.0, -1.0)
		with self.assertRaises(ValueError):
			SpeedPI(output_filter_time=-0.01)

	def test_output_filter_smooths_control_and_reset_clears_it(self):
		controller = SpeedPI(kp=1.0, ki=0.0, output_filter_time=0.05)
		output = controller.update(1.0, 0.05, -1.0, 1.0)
		self.assertAlmostEqual(output, 1.0 - 1.0 / 2.718281828459045)
		self.assertLess(output, 1.0)
		controller.reset()
		self.assertEqual(controller.output, 0.0)


if __name__ == "__main__":
	unittest.main()
