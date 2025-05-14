import unittest
import numpy as np
from scipy.integrate import odeint
from main import system_of_odes


class TestSystemOfOdes(unittest.TestCase):

    def setUp(self):
        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28
        self.vector = [1.0, 2.0, 3.0]
        self.t = 0

    def test_system_of_odes_output(self):
        result = system_of_odes(self.vector, self.t, self.sigma, self.beta, self.rho)
        self.assertEqual(len(result), 3)

    def test_system_of_odes_sample_calculation(self):
        result = system_of_odes(self.vector, self.t, self.sigma, self.beta, self.rho)
        expected_result = [
            self.sigma * (self.vector[1] - self.vector[0]),
            self.vector[0] * (self.rho - self.vector[2]) - self.vector[1],
            self.vector[0] * self.vector[1] - self.beta * self.vector[2]
        ]
        np.testing.assert_almost_equal(result, expected_result, decimal=5)

    def test_ode_integration(self):
        time_points = np.linspace(0, 40, 100)
        initial_position = [0.0, 1.0, 1.05]

        positions = odeint(system_of_odes, initial_position, time_points, args=(self.sigma, self.beta, self.rho))

        self.assertEqual(positions.shape, (100, 3))

        np.testing.assert_almost_equal(positions[0], initial_position, decimal=5)


if __name__ == "__main__":
    unittest.main()