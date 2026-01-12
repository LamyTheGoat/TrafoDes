"""
Neural Optimizer Integration Module

Provides a unified API for using trained ML models for transformer optimization.
Can be used as a drop-in replacement or complement to the physics-based optimizer.

Usage:
    from ml.inference.neural_optimizer import NeuralOptimizer

    optimizer = NeuralOptimizer()
    optimizer.load_models('ml/checkpoints')

    # Fast prediction
    outputs = optimizer.predict_outputs(design_params)

    # Fast optimization
    best_design = optimizer.optimize_fast(specifications)

    # Hybrid optimization (neural + physics verification)
    best_design = optimizer.optimize_hybrid(specifications)

    # Inverse design
    design = optimizer.inverse_design(target_specs)
"""

import sys
import os
import time
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Union
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class NeuralOptimizer:
    """
    Neural optimizer for transformer design.

    Combines surrogate model, inverse design, and optional RL policy
    for fast, high-quality optimization.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize neural optimizer.

        Args:
            device: Device to use ('cpu', 'mps', 'cuda'). Auto-detects if None.
        """
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.surrogate = None
        self.inverse_model = None
        self.rl_policy = None

        # Parameter bounds
        from mainRect import (
            CORE_MINIMUM, CORE_MAXIMUM,
            FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM,
            FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM,
            FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM,
            HVTHICK_MINIMUM, HVTHICK_MAXIMUM,
            HV_LEN_MAXIMUM,
        )

        self.bounds = np.array([
            [CORE_MINIMUM, CORE_MAXIMUM],
            [0, CORE_MAXIMUM],
            [FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM],
            [FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM],
            [FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM],
            [HVTHICK_MINIMUM, HVTHICK_MAXIMUM],
            [HVTHICK_MINIMUM, HV_LEN_MAXIMUM],
        ], dtype=np.float32)

    def load_models(self, checkpoint_dir: str):
        """
        Load trained models from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing model checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load surrogate model
        surrogate_path = checkpoint_dir / 'surrogate' / 'best_model.pt'
        if surrogate_path.exists():
            from ml.models.surrogate import load_surrogate_model
            self.surrogate = load_surrogate_model(str(surrogate_path), str(self.device))
            print(f"Loaded surrogate model from {surrogate_path}")

        # Load inverse model
        inverse_path = checkpoint_dir / 'inverse' / 'best_model.pt'
        if inverse_path.exists():
            from ml.models.inverse import load_inverse_model
            self.inverse_model = load_inverse_model(str(inverse_path), str(self.device))
            print(f"Loaded inverse model from {inverse_path}")

        # Load RL policy (if trained)
        rl_path = checkpoint_dir / 'rl' / 'policy.pt'
        if rl_path.exists():
            # Would load stable-baselines3 model here
            print(f"RL policy path exists but loading not implemented: {rl_path}")

    def load_surrogate(self, path: str):
        """Load only the surrogate model."""
        from ml.models.surrogate import load_surrogate_model
        self.surrogate = load_surrogate_model(path, str(self.device))

    def load_inverse(self, path: str):
        """Load only the inverse design model."""
        from ml.models.inverse import load_inverse_model
        self.inverse_model = load_inverse_model(path, str(self.device))

    def predict_outputs(self, params: Union[np.ndarray, Dict]) -> Dict:
        """
        Predict transformer outputs from design parameters.

        Args:
            params: Design parameters as array [core_dia, core_len, lv_turns,
                    foil_h, foil_t, hv_t, hv_len] or dict

        Returns:
            Dictionary with predicted NLL, LL, Ucc, Price, Valid
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate model not loaded. Call load_models() first.")

        # Convert dict to array if needed
        if isinstance(params, dict):
            params = np.array([
                params['core_diameter'],
                params['core_length'],
                params['lv_turns'],
                params['foil_height'],
                params['foil_thickness'],
                params['hv_thickness'],
                params['hv_length'],
            ], dtype=np.float32)

        # Ensure array
        params = np.atleast_2d(params).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(params, device=self.device)
            preds = self.surrogate.predict_denormalized(x, normalize_input=True)

        return {
            'nll': preds['nll'].cpu().numpy().squeeze(),
            'll': preds['ll'].cpu().numpy().squeeze(),
            'ucc': preds['ucc'].cpu().numpy().squeeze(),
            'price': preds['price'].cpu().numpy().squeeze(),
            'valid': preds['valid'].cpu().numpy().squeeze() > 0.5,
        }

    def optimize_fast(self,
                      specifications: Optional[Dict] = None,
                      n_iterations: int = 100,
                      n_candidates: int = 1000,
                      use_inverse_start: bool = True) -> Tuple[np.ndarray, float]:
        """
        Fast optimization using surrogate model with gradient descent.

        Args:
            specifications: Target specifications (NLL, LL, Ucc targets)
            n_iterations: Number of gradient descent iterations
            n_candidates: Number of random candidates to try
            use_inverse_start: Use inverse model for initial guess

        Returns:
            Tuple of (best_params, best_cost)
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate model not loaded.")

        # Generate initial candidates
        if use_inverse_start and self.inverse_model is not None and specifications is not None:
            # Use inverse model for some candidates
            target_specs = torch.tensor([[
                specifications.get('guaranteed_nll', 800),
                specifications.get('guaranteed_ll', 7000),
                specifications.get('guaranteed_ucc', 6.0),
                10000,  # Price estimate
            ]], dtype=torch.float32, device=self.device)

            inverse_candidates = self.inverse_model.predict(
                target_specs, n_samples=n_candidates // 2, normalize_input=True
            ).cpu().numpy()
        else:
            inverse_candidates = np.array([]).reshape(0, 7)

        # Random candidates
        n_random = n_candidates - len(inverse_candidates)
        random_candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_random, 7)
        ).astype(np.float32)

        # Apply constraints
        random_candidates[:, 1] = np.minimum(random_candidates[:, 1], random_candidates[:, 0])
        random_candidates[:, 6] = np.maximum(random_candidates[:, 6], random_candidates[:, 5])
        random_candidates[:, 2] = np.round(random_candidates[:, 2])

        # Combine candidates
        if len(inverse_candidates) > 0:
            candidates = np.vstack([inverse_candidates, random_candidates])
        else:
            candidates = random_candidates

        # Evaluate all candidates
        with torch.no_grad():
            x = torch.tensor(candidates, dtype=torch.float32, device=self.device)
            preds = self.surrogate.predict_denormalized(x, normalize_input=True)
            costs = preds['price'].cpu().numpy()
            valid = preds['valid'].cpu().numpy() > 0.5

        # Add penalty for invalid designs
        costs[~valid] += 1e6

        # Get best candidate
        best_idx = np.argmin(costs)
        best_params = candidates[best_idx]
        best_cost = costs[best_idx]

        # Optional: Gradient-based refinement
        if n_iterations > 0:
            refined_params, refined_cost = self._gradient_refinement(
                best_params, n_iterations, specifications
            )
            if refined_cost < best_cost:
                best_params = refined_params
                best_cost = refined_cost

        return best_params, float(best_cost)

    def _gradient_refinement(self,
                             initial_params: np.ndarray,
                             n_iterations: int,
                             specifications: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
        """Gradient-based refinement using surrogate."""
        from mainRect import (
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            PENALTY_NLL_FACTOR, PENALTY_LL_FACTOR, PENALTY_UCC_FACTOR,
            UCC_TOLERANCE,
        )

        specs = specifications or {}
        nll_target = specs.get('guaranteed_nll', GUARANTEED_NO_LOAD_LOSS)
        ll_target = specs.get('guaranteed_ll', GUARANTEED_LOAD_LOSS)
        ucc_target = specs.get('guaranteed_ucc', GUARANTEED_UCC)
        ucc_tol = specs.get('ucc_tolerance', UCC_TOLERANCE)

        # Normalize initial params
        norm_params = (initial_params - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        norm_params = np.clip(norm_params, 0.01, 0.99)  # Stay away from bounds

        # Convert to tensor with gradients
        x = torch.tensor(norm_params, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([x], lr=0.01)

        best_cost = float('inf')
        best_x = norm_params.copy()

        for _ in range(n_iterations):
            optimizer.zero_grad()

            # Denormalize
            params = x * torch.tensor(self.bounds[:, 1] - self.bounds[:, 0], device=self.device) + \
                    torch.tensor(self.bounds[:, 0], device=self.device)

            # Get predictions
            preds = self.surrogate.predict_denormalized(params.unsqueeze(0), normalize_input=True)

            # Cost function
            cost = preds['price']

            # Constraint penalties
            nll_violation = torch.relu(preds['nll'] - nll_target)
            ll_violation = torch.relu(preds['ll'] - ll_target)
            ucc_violation = torch.relu(torch.abs(preds['ucc'] - ucc_target) - ucc_tol)

            penalty = (
                nll_violation * PENALTY_NLL_FACTOR +
                ll_violation * PENALTY_LL_FACTOR +
                ucc_violation * PENALTY_UCC_FACTOR
            )

            loss = cost + penalty

            loss.backward()
            optimizer.step()

            # Clamp to valid range
            with torch.no_grad():
                x.clamp_(0.01, 0.99)

            # Track best
            current_cost = loss.item()
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.detach().cpu().numpy().copy()

        # Denormalize best result
        best_params = best_x * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

        # Apply constraints
        best_params[1] = min(best_params[1], best_params[0])
        best_params[6] = max(best_params[6], best_params[5])
        best_params[2] = round(best_params[2])

        return best_params, best_cost

    def optimize_hybrid(self,
                        specifications: Optional[Dict] = None,
                        n_candidates: int = 100,
                        n_verify: int = 10) -> Tuple[np.ndarray, float, Dict]:
        """
        Hybrid optimization: neural candidates + physics verification.

        Uses surrogate model to generate candidates, then verifies
        top candidates with actual physics calculations.

        Args:
            specifications: Target specifications
            n_candidates: Number of neural candidates to generate
            n_verify: Number of top candidates to verify with physics

        Returns:
            Tuple of (best_params, best_cost, full_results)
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate model not loaded.")

        # Generate candidates using surrogate
        candidates, neural_costs = [], []
        for _ in range(n_candidates):
            params, cost = self.optimize_fast(
                specifications=specifications,
                n_iterations=50,
                n_candidates=100,
                use_inverse_start=True
            )
            candidates.append(params)
            neural_costs.append(cost)

        # Sort by neural cost
        sorted_indices = np.argsort(neural_costs)

        # Verify top candidates with physics
        from mainRect import CalculateFinalizedPriceIntolerant_Optimized, LVRATE, HVRATE, POWERRATING, FREQUENCY

        specs = specifications or {}
        best_physics_cost = float('inf')
        best_params = None
        best_results = None

        for idx in sorted_indices[:n_verify]:
            params = candidates[idx]

            try:
                physics_cost = CalculateFinalizedPriceIntolerant_Optimized(
                    int(params[2]),  # lv_turns
                    params[3],  # foil_height
                    params[4],  # foil_thickness
                    params[5],  # hv_thickness
                    params[6],  # hv_length
                    params[0],  # core_diameter
                    params[1],  # core_length
                    LVRATE_VAL=specs.get('lv_rating', LVRATE),
                    HVRATE_VAL=specs.get('hv_rating', HVRATE),
                    POWER=specs.get('power_rating', POWERRATING),
                    FREQ=specs.get('frequency', FREQUENCY),
                    tolerance=1,
                    PutCoolingDucts=True,
                )

                if physics_cost < best_physics_cost and physics_cost < 1e17:
                    best_physics_cost = physics_cost
                    best_params = params
                    best_results = {
                        'neural_cost': neural_costs[idx],
                        'physics_cost': physics_cost,
                        'params': params.copy(),
                    }

            except Exception as e:
                continue

        if best_params is None:
            # Fallback to best neural result
            best_idx = sorted_indices[0]
            best_params = candidates[best_idx]
            best_physics_cost = neural_costs[best_idx]
            best_results = {'neural_cost': neural_costs[best_idx], 'fallback': True}

        return best_params, best_physics_cost, best_results

    def inverse_design(self,
                       target_nll: float,
                       target_ll: float,
                       target_ucc: float,
                       n_samples: int = 10) -> List[np.ndarray]:
        """
        Generate design parameters from target specifications.

        Args:
            target_nll: Target No-Load Loss
            target_ll: Target Load Loss
            target_ucc: Target Impedance
            n_samples: Number of design alternatives to generate

        Returns:
            List of design parameter arrays
        """
        if self.inverse_model is None:
            raise RuntimeError("Inverse model not loaded.")

        target_specs = torch.tensor([[
            target_nll,
            target_ll,
            target_ucc,
            10000,  # Rough price estimate
        ]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            params = self.inverse_model.predict(
                target_specs, n_samples=n_samples, normalize_input=True
            )

        return [p.cpu().numpy() for p in params]

    def benchmark(self,
                  n_tests: int = 100,
                  specifications: Optional[Dict] = None) -> Dict:
        """
        Benchmark neural vs physics optimization.

        Args:
            n_tests: Number of test cases
            specifications: Specifications to use

        Returns:
            Benchmark results including speedup and accuracy
        """
        from mainRect import CalculateFinalizedPriceIntolerant_Optimized, LVRATE, HVRATE, POWERRATING, FREQUENCY

        specs = specifications or {}

        results = {
            'neural_times': [],
            'physics_times': [],
            'neural_costs': [],
            'physics_costs': [],
            'cost_ratios': [],
        }

        for _ in range(n_tests):
            # Neural optimization
            t0 = time.time()
            params, neural_cost = self.optimize_fast(specifications=specs)
            neural_time = time.time() - t0

            # Physics verification
            t0 = time.time()
            try:
                physics_cost = CalculateFinalizedPriceIntolerant_Optimized(
                    int(params[2]), params[3], params[4], params[5], params[6],
                    params[0], params[1],
                    LVRATE_VAL=specs.get('lv_rating', LVRATE),
                    HVRATE_VAL=specs.get('hv_rating', HVRATE),
                    POWER=specs.get('power_rating', POWERRATING),
                    tolerance=1, PutCoolingDucts=True,
                )
            except:
                physics_cost = 1e18
            physics_time = time.time() - t0

            results['neural_times'].append(neural_time)
            results['physics_times'].append(physics_time)
            results['neural_costs'].append(neural_cost)
            results['physics_costs'].append(physics_cost if physics_cost < 1e17 else np.nan)
            if physics_cost < 1e17:
                results['cost_ratios'].append(neural_cost / physics_cost)

        # Summary statistics
        return {
            'avg_neural_time': np.mean(results['neural_times']),
            'avg_physics_time': np.mean(results['physics_times']),
            'speedup': np.mean(results['physics_times']) / np.mean(results['neural_times']),
            'mean_cost_ratio': np.nanmean(results['cost_ratios']),
            'median_cost_ratio': np.nanmedian(results['cost_ratios']),
            'valid_rate': np.sum(~np.isnan(results['physics_costs'])) / n_tests,
        }


# Convenience function to create optimizer with default models
def create_optimizer(checkpoint_dir: str = 'ml/checkpoints') -> NeuralOptimizer:
    """Create neural optimizer and load models."""
    optimizer = NeuralOptimizer()
    optimizer.load_models(checkpoint_dir)
    return optimizer
