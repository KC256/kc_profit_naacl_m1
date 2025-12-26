
import unittest
import torch
import sys
import os

# Define mock environment variables and paths to import model_12
sys.path.append("/home/fukuda/M1_reserch/kc_profit_naacl_m1/profit_naacl_v3")

# Mocking config constants that might be missing or need specific values
# We need to ensure these match what model_12 expects or safe defaults
class MockConfig:
    tsec_lstm1_outshape = 128
    lstm2_outshape = 128
    # Add other necessary constants if import fails

# Attempt import. If it fails due to missing configs, we might need to mock configs_stock
try:
    from model_12 import Actor, Critic
except ImportError:
    # Quick fix to mock configs_stock if it's not easily importable or has side effects
    sys.modules['configs_stock'] = MockConfig()
    from model_12 import Actor, Critic

class TestModel12Init(unittest.TestCase):
    def test_actor_linear_c_dim(self):
        # Case 1: num_stocks = 22
        num_stocks = 22
        actor = Actor(num_stocks=num_stocks)
        expected_dim = 32 + num_stocks * 144 # 32 + 22 * 144 = 3200
        self.assertEqual(actor.linear_c.in_features, expected_dim, 
                         f"Actor linear_c in_features should be {expected_dim} for num_stocks={num_stocks}")

        # Case 2: num_stocks = 20
        num_stocks = 20
        actor = Actor(num_stocks=num_stocks)
        expected_dim = 32 + num_stocks * 144 # 32 + 20 * 144 = 2912
        self.assertEqual(actor.linear_c.in_features, expected_dim,
                         f"Actor linear_c in_features should be {expected_dim} for num_stocks={num_stocks}")

    def test_critic_linear_c_dim(self):
        # Case 1: num_stocks = 22
        num_stocks = 22
        critic = Critic(num_stocks=num_stocks)
        expected_dim = 32 + num_stocks * 144 # 32 + 22 * 144 = 3200
        self.assertEqual(critic.linear_c.in_features, expected_dim,
                         f"Critic linear_c in_features should be {expected_dim} for num_stocks={num_stocks}")

        # Case 2: num_stocks = 20
        num_stocks = 20
        critic = Critic(num_stocks=num_stocks)
        expected_dim = 32 + num_stocks * 144 # 32 + 20 * 144 = 2912
        self.assertEqual(critic.linear_c.in_features, expected_dim,
                         f"Critic linear_c in_features should be {expected_dim} for num_stocks={num_stocks}")

if __name__ == '__main__':
    unittest.main()
