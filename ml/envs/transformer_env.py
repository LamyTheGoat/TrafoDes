"""
Gymnasium Environment for Transformer Design Optimization

Reinforcement learning environment for optimizing transformer designs.
Uses either physics-based simulation or a trained surrogate model.

Usage:
    import gymnasium as gym
    from ml.envs.transformer_env import TransformerDesignEnv

    env = TransformerDesignEnv(specifications)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
"""

import sys
import os
import numpy as np
from typing import Dict, Optional, Tuple, Any

# Try to import gymnasium, fall back to gym if not available
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = 'gymnasium'
except ImportError:
    import gym
    from gym import spaces
    GYM_VERSION = 'gym'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mainRect import (
    # Constants
    CORE_MINIMUM, CORE_MAXIMUM, CORELENGTH_MINIMUM,
    FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM,
    FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM,
    FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM,
    HVTHICK_MINIMUM, HVTHICK_MAXIMUM,
    HV_LEN_MINIMUM, HV_LEN_MAXIMUM,
    LVRATE, HVRATE, POWERRATING, FREQUENCY,
    GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, UCC_TOLERANCE,
    PENALTY_NLL_FACTOR, PENALTY_LL_FACTOR, PENALTY_UCC_FACTOR, MAX_GRADIENT,
    materialToBeUsedWire_Resistivity,
    # Functions
    CalculateFinalizedPriceIntolerant_Optimized,
    CalculateLoadLosses,
    CalculateNoLoadLosses,
    CalculateUx, CalculateUr, CalculateImpedance,
    CalculatePrice,
    CalculateVoltsPerTurns,
    CalculateInduction,
    CalculateStrayDiameter,
    CalculateNumberOfCoolingDucts_WithLosses,
)


class TransformerDesignEnv(gym.Env):
    """
    Transformer Design Optimization Environment.

    Observation Space:
        - 7 design parameters (normalized to [-1, 1])
        - Current cost
        - Constraint violations (NLL, LL, Ucc)
        - Feasibility flag

    Action Space:
        - 7 continuous values in [-1, 1] representing design parameters
        (OR deltas to current design, depending on action_mode)

    Reward:
        - Negative cost (minimization)
        - Penalty for constraint violations
        - Bonus for finding feasible solutions
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self,
                 specifications: Optional[Dict] = None,
                 use_surrogate: bool = False,
                 surrogate_model: Optional[Any] = None,
                 action_mode: str = 'absolute',  # 'absolute' or 'delta'
                 max_steps: int = 50,
                 reward_scale: float = 1e-4,
                 constraint_penalty: float = 100.0,
                 feasibility_bonus: float = 10.0,
                 put_cooling_ducts: bool = True):
        """
        Initialize environment.

        Args:
            specifications: Dict with power_rating, hv_rating, lv_rating, etc.
            use_surrogate: Use surrogate model instead of physics
            surrogate_model: Trained surrogate model (if use_surrogate=True)
            action_mode: 'absolute' = action is the design, 'delta' = action is change
            max_steps: Maximum steps per episode
            reward_scale: Scale factor for reward
            constraint_penalty: Penalty multiplier for constraint violations
            feasibility_bonus: Bonus for finding feasible design
            put_cooling_ducts: Whether to include cooling ducts
        """
        super().__init__()

        # Default specifications
        self.specifications = specifications or {
            'power_rating': POWERRATING,
            'hv_rating': HVRATE,
            'lv_rating': LVRATE,
            'frequency': FREQUENCY,
            'guaranteed_nll': GUARANTEED_NO_LOAD_LOSS,
            'guaranteed_ll': GUARANTEED_LOAD_LOSS,
            'guaranteed_ucc': GUARANTEED_UCC,
            'ucc_tolerance': UCC_TOLERANCE,
        }

        self.use_surrogate = use_surrogate
        self.surrogate_model = surrogate_model
        self.action_mode = action_mode
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.constraint_penalty = constraint_penalty
        self.feasibility_bonus = feasibility_bonus
        self.put_cooling_ducts = put_cooling_ducts

        # Parameter bounds
        self.bounds = np.array([
            [CORE_MINIMUM, CORE_MAXIMUM],           # core_diameter
            [CORELENGTH_MINIMUM, CORE_MAXIMUM],     # core_length (max = core_dia)
            [FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM], # lv_turns
            [FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM],  # foil_height
            [FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM],  # foil_thickness
            [HVTHICK_MINIMUM, HVTHICK_MAXIMUM],     # hv_thickness
            [HV_LEN_MINIMUM, HV_LEN_MAXIMUM],       # hv_length
        ], dtype=np.float32)

        # Action space: normalized parameters [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Observation space: [params, cost, violations, feasible]
        # params: 7 values
        # cost: 1 value (normalized)
        # violations: 3 values (nll, ll, ucc)
        # feasible: 1 value (0 or 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.current_design = None
        self.current_results = None
        self.best_feasible_cost = np.inf
        self.best_feasible_design = None

        # Price normalization (rough estimate)
        self.price_norm = 10000.0

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to design parameters."""
        # Scale from [-1, 1] to [0, 1]
        scaled = (action + 1) / 2

        # Scale to parameter ranges
        params = scaled * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

        # Apply constraints
        # core_length <= core_diameter
        params[1] = min(params[1], params[0])
        # hv_length >= hv_thickness
        params[6] = max(params[6], params[5])
        # Round lv_turns to integer
        params[2] = round(params[2])

        return params

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize design parameters to [-1, 1]."""
        scaled = (params - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return scaled * 2 - 1

    def _evaluate_design(self, params: np.ndarray) -> Dict:
        """Evaluate design using physics or surrogate model."""
        if self.use_surrogate and self.surrogate_model is not None:
            return self._evaluate_surrogate(params)
        return self._evaluate_physics(params)

    def _evaluate_physics(self, params: np.ndarray) -> Dict:
        """Evaluate design using physics-based calculations."""
        core_dia, core_len, lv_turns, foil_h, foil_t, hv_t, hv_l = params
        lv_turns = int(lv_turns)

        specs = self.specifications

        try:
            # Calculate cooling ducts
            if self.put_cooling_ducts:
                lv_cd, hv_cd, ll_zero, ll_lv, ll_hv = CalculateNumberOfCoolingDucts_WithLosses(
                    lv_turns, foil_h, foil_t, hv_t, hv_l,
                    core_dia, core_len, materialToBeUsedWire_Resistivity,
                    specs['power_rating'], specs['hv_rating'], specs['lv_rating'],
                    False, False
                )
                if lv_cd == 0 and hv_cd == 0:
                    ll = ll_zero
                else:
                    ll, ll_hv, ll_lv = CalculateLoadLosses(
                        lv_turns, foil_t, core_dia, foil_h, hv_t, hv_l,
                        materialToBeUsedWire_Resistivity, specs['power_rating'],
                        specs['hv_rating'], specs['lv_rating'], core_len,
                        lv_cd, hv_cd, False
                    )
            else:
                lv_cd, hv_cd = 0, 0
                ll, ll_hv, ll_lv = CalculateLoadLosses(
                    lv_turns, foil_t, core_dia, foil_h, hv_t, hv_l,
                    materialToBeUsedWire_Resistivity, specs['power_rating'],
                    specs['hv_rating'], specs['lv_rating'], core_len, 0, 0, False
                )

            # Check for invalid calculation
            if ll >= 1e17 or ll < 0:
                return self._invalid_result()

            # NLL
            nll = CalculateNoLoadLosses(
                lv_turns, foil_h, foil_t, hv_t, hv_l, core_dia,
                specs['lv_rating'], core_len, lv_cd, hv_cd, False
            )
            if nll >= 1e17 or nll < 0:
                return self._invalid_result()

            # Impedance
            stray_dia = CalculateStrayDiameter(
                lv_turns, foil_t, foil_h, hv_t, hv_l,
                core_dia, core_len, lv_cd, hv_cd, False
            )
            ux = CalculateUx(
                specs['power_rating'], stray_dia, lv_turns, foil_t, foil_h,
                hv_t, hv_l, specs['frequency'], specs['lv_rating'],
                lv_cd, hv_cd, False
            )
            ur = CalculateUr(ll, specs['power_rating'])
            ucc = CalculateImpedance(ux, ur)

            if ucc >= 1e17 or ucc < 0:
                return self._invalid_result()

            # Price
            price = CalculatePrice(
                lv_turns, foil_h, foil_t, hv_t, hv_l,
                core_dia, core_len, lv_cd, hv_cd, False, False
            )
            if price >= 1e17 or price < 0:
                return self._invalid_result()

            # Constraint violations
            nll_violation = max(0, nll - specs['guaranteed_nll'])
            ll_violation = max(0, ll - specs['guaranteed_ll'])
            ucc_violation = max(0, abs(ucc - specs['guaranteed_ucc']) - specs['ucc_tolerance'])

            is_feasible = (nll_violation == 0 and ll_violation == 0 and ucc_violation == 0)

            return {
                'nll': nll,
                'll': ll,
                'ucc': ucc,
                'price': price,
                'nll_violation': nll_violation,
                'll_violation': ll_violation,
                'ucc_violation': ucc_violation,
                'is_feasible': is_feasible,
                'is_valid': True,
            }

        except Exception:
            return self._invalid_result()

    def _evaluate_surrogate(self, params: np.ndarray) -> Dict:
        """Evaluate design using surrogate model."""
        import torch

        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            preds = self.surrogate_model.predict_denormalized(x, normalize_input=True)

            nll = preds['nll'].item()
            ll = preds['ll'].item()
            ucc = preds['ucc'].item()
            price = preds['price'].item()
            valid_prob = preds['valid'].item()

        specs = self.specifications
        nll_violation = max(0, nll - specs['guaranteed_nll'])
        ll_violation = max(0, ll - specs['guaranteed_ll'])
        ucc_violation = max(0, abs(ucc - specs['guaranteed_ucc']) - specs['ucc_tolerance'])

        is_feasible = (nll_violation == 0 and ll_violation == 0 and ucc_violation == 0)

        return {
            'nll': nll,
            'll': ll,
            'ucc': ucc,
            'price': price,
            'nll_violation': nll_violation,
            'll_violation': ll_violation,
            'ucc_violation': ucc_violation,
            'is_feasible': is_feasible,
            'is_valid': valid_prob > 0.5,
        }

    def _invalid_result(self) -> Dict:
        """Return invalid design result."""
        return {
            'nll': 1e18,
            'll': 1e18,
            'ucc': 1e18,
            'price': 1e18,
            'nll_violation': 1e18,
            'll_violation': 1e18,
            'ucc_violation': 1e18,
            'is_feasible': False,
            'is_valid': False,
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_design is None or self.current_results is None:
            return np.zeros(12, dtype=np.float32)

        params_norm = self._normalize_params(self.current_design)
        cost_norm = self.current_results['price'] / self.price_norm

        obs = np.concatenate([
            params_norm,
            [cost_norm],
            [self.current_results['nll_violation'] / self.specifications['guaranteed_nll']],
            [self.current_results['ll_violation'] / self.specifications['guaranteed_ll']],
            [self.current_results['ucc_violation'] / self.specifications['guaranteed_ucc']],
            [float(self.current_results['is_feasible'])],
        ]).astype(np.float32)

        return obs

    def _calculate_reward(self, results: Dict, prev_results: Optional[Dict] = None) -> float:
        """Calculate reward signal."""
        if not results['is_valid']:
            return -1000.0 * self.reward_scale

        # Base reward: negative cost
        cost = results['price']
        reward = -cost * self.reward_scale

        # Constraint penalties
        penalty = (
            results['nll_violation'] * PENALTY_NLL_FACTOR +
            results['ll_violation'] * PENALTY_LL_FACTOR +
            results['ucc_violation'] * PENALTY_UCC_FACTOR
        )
        reward -= penalty * self.constraint_penalty * self.reward_scale

        # Feasibility bonus
        if results['is_feasible']:
            reward += self.feasibility_bonus * self.reward_scale

            # Extra bonus for improvement
            if cost < self.best_feasible_cost:
                improvement = (self.best_feasible_cost - cost) / self.price_norm
                reward += improvement * 10.0 * self.reward_scale

        return reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.best_feasible_cost = np.inf
        self.best_feasible_design = None

        # Initialize with random or provided starting point
        if options and 'initial_design' in options:
            self.current_design = np.array(options['initial_design'], dtype=np.float32)
        else:
            # Random start
            if self.np_random is not None:
                random_action = self.np_random.uniform(-1, 1, size=7).astype(np.float32)
            else:
                random_action = np.random.uniform(-1, 1, size=7).astype(np.float32)
            self.current_design = self._denormalize_action(random_action)

        self.current_results = self._evaluate_design(self.current_design)

        if self.current_results['is_feasible']:
            self.best_feasible_cost = self.current_results['price']
            self.best_feasible_design = self.current_design.copy()

        obs = self._get_observation()
        info = {
            'design': self.current_design.copy(),
            'results': self.current_results.copy(),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.current_step += 1

        # Apply action
        if self.action_mode == 'absolute':
            new_design = self._denormalize_action(action)
        else:  # delta mode
            delta = action * 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
            new_design = np.clip(
                self.current_design + delta,
                self.bounds[:, 0],
                self.bounds[:, 1]
            )
            # Apply constraints
            new_design[1] = min(new_design[1], new_design[0])
            new_design[6] = max(new_design[6], new_design[5])
            new_design[2] = round(new_design[2])

        # Evaluate new design
        prev_results = self.current_results
        self.current_design = new_design
        self.current_results = self._evaluate_design(new_design)

        # Calculate reward
        reward = self._calculate_reward(self.current_results, prev_results)

        # Update best
        if self.current_results['is_feasible']:
            if self.current_results['price'] < self.best_feasible_cost:
                self.best_feasible_cost = self.current_results['price']
                self.best_feasible_design = self.current_design.copy()

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            'design': self.current_design.copy(),
            'results': self.current_results.copy(),
            'best_cost': self.best_feasible_cost,
            'best_design': self.best_feasible_design.copy() if self.best_feasible_design is not None else None,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = 'human'):
        """Render current state."""
        if self.current_design is None:
            print("Environment not initialized")
            return

        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print(f"Design: CoreDia={self.current_design[0]:.1f}, CoreLen={self.current_design[1]:.1f}, "
              f"LVTurns={int(self.current_design[2])}, FoilH={self.current_design[3]:.1f}")
        print(f"Results: NLL={self.current_results['nll']:.1f}, LL={self.current_results['ll']:.1f}, "
              f"Ucc={self.current_results['ucc']:.2f}%, Price=${self.current_results['price']:.2f}")
        print(f"Feasible: {self.current_results['is_feasible']}")
        if self.best_feasible_cost < np.inf:
            print(f"Best Cost: ${self.best_feasible_cost:.2f}")

    def close(self):
        """Clean up resources."""
        pass


# Register environment with gymnasium if available
try:
    from gymnasium.envs.registration import register
    register(
        id='TransformerDesign-v0',
        entry_point='ml.envs.transformer_env:TransformerDesignEnv',
    )
except ImportError:
    pass
