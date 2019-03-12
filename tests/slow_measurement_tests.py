import unittest
import copy
import numpy as np
import random
from tqdm import tqdm

import sys
sys.path.append('..')
import qcircuits as qc
from tests import random_state


epsilon = 1e-10


class SlowMeasurementTests(unittest.TestCase):

    def test_measurement(self):
        num_tests = 3
        samples = 300000
        prob_epsilon = 0.05

        for test_i in range(num_tests):
            d = np.random.randint(1, 4)
            x = random_state(d=d)
            ps = x.probabilities
            counts = np.zeros_like(ps)

            for sample in tqdm(range(samples)):
                measurements = copy.deepcopy(x).measure()
                counts[measurements] += 1

            proportional_abs_diff = np.max(np.abs(ps - counts/samples) / ps)
            self.assertLess(proportional_abs_diff, prob_epsilon)

    def test_out_of_order_measurement_without_reduce(self):
        num_tests = 3
        samples = 300000
        prob_epsilon = 0.05

        for test_i in range(num_tests):
            d = np.random.randint(1, 4)
            x = random_state(d=d)
            ps = x.probabilities
            counts = np.zeros_like(ps)

            for sample in tqdm(range(samples)):
                state = copy.deepcopy(x)
                measurements = [0] * d
                indices = list(range(d))
                random.shuffle(indices)
                for measure_i in indices:
                    measurements[measure_i] = state.measure(qubit_indices=measure_i, remove=False)

                counts[tuple(measurements)] += 1

            proportional_abs_diff = np.max(np.abs(ps - counts/samples) / ps)
            self.assertLess(proportional_abs_diff, prob_epsilon)

    def test_out_of_order_measurement_with_reduce(self):
        num_tests = 3
        samples = 300000
        prob_epsilon = 0.05

        for test_i in range(num_tests):
            d = np.random.randint(1, 4)
            x = random_state(d=d)
            ps = x.probabilities
            counts = np.zeros_like(ps)

            for sample in tqdm(range(samples)):
                state = copy.deepcopy(x)
                measurements = [0] * d
                indices = list(range(d))
                random.shuffle(indices)
                lookup_indices = list(range(d))
                for measure_i in indices:
                    qubit_index = lookup_indices.index(measure_i)
                    lookup_indices.remove(measure_i)
                    measurements[measure_i] = state.measure(qubit_indices=qubit_index, remove=True)

                counts[tuple(measurements)] += 1

            proportional_abs_diff = np.max(np.abs(ps - counts/samples) / ps)
            self.assertLess(proportional_abs_diff, prob_epsilon)


if __name__ == '__main__':
    unittest.main()
