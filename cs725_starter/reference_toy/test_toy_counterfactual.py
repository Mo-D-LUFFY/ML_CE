import unittest
from toy_counterfactual import (
    Profile, model_score, model_label, constraint_reasons,
    generate_feasible_counterfactuals,
)

class ToyRecourseTests(unittest.TestCase):
    def setUp(self):
        self.original = Profile(2, 0, 3)

    def test_original(self):
        self.assertEqual(model_score(self.original), 10)
        self.assertEqual(model_label(self.original), 0)

    def test_past_grade_flip_is_invalid(self):
        p = Profile(2, 0, 5)
        self.assertEqual(model_label(p), 1)
        self.assertTrue(constraint_reasons(self.original, p))

    def test_hours_change_is_valid(self):
        p = Profile(4, 0, 3)
        self.assertEqual(model_label(p), 1)
        self.assertEqual(constraint_reasons(self.original, p), [])

    def test_support_combination_valid_when_available(self):
        p = Profile(3, 1, 3)
        self.assertEqual(model_label(p), 1)
        self.assertEqual(constraint_reasons(self.original, p), [])
        self.assertTrue(constraint_reasons(self.original, p, support_accessible=False))

    def test_max_hours(self):
        self.assertTrue(constraint_reasons(self.original, Profile(9, 0, 3)))

    def test_optimal_feasible_candidate(self):
        candidates = generate_feasible_counterfactuals(self.original)
        self.assertEqual(candidates[0]["candidate"], {"study_hours": 4, "extra_support": 0, "earlier_grade": 3})
        self.assertEqual(candidates[0]["illustrative_cost_units"], 2)
        self.assertEqual(len(candidates), 3)

if __name__ == "__main__":
    unittest.main()
