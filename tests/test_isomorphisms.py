import unittest

import torch

from src.identifiability_checks import build_network, extract_parameters
from src.network_isomorphisms import (
    check_extensional_isomorphism,
    check_faithful_isomorphism,
    create_permuted_network,
)


class IsomorphismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.architecture = [3, 4, 3, 1]
        self.model = build_network(self.architecture, "tanh")

    def test_permuted_tanh_network_is_faithfully_isomorphic(self):
        equivalent = create_permuted_network(self.model, self.architecture, "tanh")

        report = check_faithful_isomorphism(
            extract_parameters(self.model), extract_parameters(equivalent)
        )

        self.assertTrue(report["is_faithfully_isomorphic"])
        self.assertLess(report["total_alignment_cost"], 1e-5)

    def test_unrelated_network_is_not_extensionally_isomorphic(self):
        torch.manual_seed(7)
        unrelated = build_network(self.architecture, "tanh")

        report = check_extensional_isomorphism(
            self.model, unrelated, self.architecture, n_test=200
        )

        self.assertFalse(report["is_extensionally_isomorphic"])


if __name__ == "__main__":
    unittest.main()
