"""
Optimization Engine
Optimizes greenhouse operations for resource efficiency and plant health
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy.optimize import minimize


class OptimizationEngine:
    """Optimizes greenhouse settings for multiple objectives"""

    def __init__(self):
        self.current_conditions = {}
        self.optimization_history = []

    def optimize_for_growth(self,
                           current_conditions: Dict[str, float],
                           actuator_constraints: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize settings for maximum plant growth
        actuator_constraints: Dict of parameter -> (min, max) tuples
        """
        optimal_targets = {
            'temperature': 22,  # °C
            'humidity': 60,  # %
            'soil_moisture': 50,  # %
            'light': 25000,  # lux
            'co2': 1000  # ppm
        }

        recommendations = {}
        priority_actions = []

        for param, target in optimal_targets.items():
            if param in current_conditions:
                current = current_conditions[param]
                difference = target - current

                if param in actuator_constraints:
                    min_val, max_val = actuator_constraints[param]
                    achievable_target = max(min_val, min(max_val, target))

                    recommendations[param] = {
                        'current': round(current, 2),
                        'target': round(achievable_target, 2),
                        'adjustment_needed': round(achievable_target - current, 2),
                        'feasible': min_val <= target <= max_val
                    }

                    if abs(achievable_target - current) > 0.1:
                        priority = self._calculate_adjustment_priority(param, abs(difference))
                        priority_actions.append({
                            'parameter': param,
                            'action': 'increase' if difference > 0 else 'decrease',
                            'magnitude': abs(difference),
                            'priority': priority
                        })

        # Sort actions by priority
        priority_actions.sort(key=lambda x: (x['priority'], x['magnitude']), reverse=True)

        return {
            'objective': 'maximize_growth',
            'recommendations': recommendations,
            'priority_actions': priority_actions,
            'timestamp': datetime.now().isoformat()
        }

    def optimize_for_efficiency(self,
                                current_conditions: Dict[str, float],
                                energy_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize for resource efficiency while maintaining acceptable growth
        energy_costs: Cost per unit for each resource (e.g., kwh for lighting)
        """
        # Acceptable ranges (wider than optimal for cost savings)
        acceptable_ranges = {
            'temperature': (18, 26),
            'humidity': (45, 75),
            'soil_moisture': (35, 65),
            'light': (12000, 35000),
            'co2': (600, 1400)
        }

        recommendations = {}
        potential_savings = []

        for param, (min_acceptable, max_acceptable) in acceptable_ranges.items():
            if param in current_conditions:
                current = current_conditions[param]

                # If current value is in acceptable range, no change needed
                if min_acceptable <= current <= max_acceptable:
                    recommendations[param] = {
                        'current': current,
                        'status': 'acceptable',
                        'action': 'maintain'
                    }
                else:
                    # Recommend moving to nearest acceptable boundary
                    if current < min_acceptable:
                        target = min_acceptable
                        action = 'increase'
                    else:
                        target = max_acceptable
                        action = 'decrease'

                    recommendations[param] = {
                        'current': round(current, 2),
                        'target': round(target, 2),
                        'status': 'out_of_range',
                        'action': action
                    }

                # Calculate potential savings
                if param in energy_costs and current > max_acceptable:
                    excess = current - max_acceptable
                    cost_per_unit = energy_costs[param]
                    estimated_savings = excess * cost_per_unit

                    potential_savings.append({
                        'parameter': param,
                        'excess_value': round(excess, 2),
                        'estimated_daily_savings': round(estimated_savings, 2)
                    })

        total_potential_savings = sum(s['estimated_daily_savings'] for s in potential_savings)

        return {
            'objective': 'maximize_efficiency',
            'recommendations': recommendations,
            'potential_savings': potential_savings,
            'total_daily_savings': round(total_potential_savings, 2),
            'timestamp': datetime.now().isoformat()
        }

    def optimize_irrigation_schedule(self,
                                    soil_moisture_history: List[float],
                                    evapotranspiration_rate: float = 0.5) -> Dict[str, Any]:
        """
        Optimize irrigation schedule based on moisture patterns
        evapotranspiration_rate: Water loss rate in %/hour
        """
        if not soil_moisture_history:
            return {'error': 'No moisture history available'}

        current_moisture = soil_moisture_history[-1]
        target_min = 40
        target_max = 60

        # Calculate moisture depletion rate
        if len(soil_moisture_history) >= 2:
            recent_readings = soil_moisture_history[-12:]  # Last 12 readings
            avg_depletion = (recent_readings[0] - recent_readings[-1]) / len(recent_readings)
        else:
            avg_depletion = evapotranspiration_rate

        # Predict when irrigation will be needed
        if avg_depletion > 0:
            hours_until_min = (current_moisture - target_min) / avg_depletion
        else:
            hours_until_min = 999  # Not depleting

        # Recommend irrigation schedule
        if current_moisture < target_min:
            recommendation = {
                'action': 'irrigate_now',
                'duration_seconds': 60,
                'reason': f'Moisture below minimum ({current_moisture:.1f}% < {target_min}%)'
            }
        elif current_moisture < target_max and hours_until_min < 2:
            recommendation = {
                'action': 'irrigate_soon',
                'duration_seconds': 45,
                'hours_until_needed': round(hours_until_min, 1),
                'reason': 'Moisture will drop below minimum soon'
            }
        else:
            recommendation = {
                'action': 'wait',
                'hours_until_needed': round(hours_until_min, 1),
                'reason': f'Moisture adequate ({current_moisture:.1f}%)'
            }

        return {
            'current_moisture': round(current_moisture, 1),
            'target_range': f"{target_min}-{target_max}%",
            'depletion_rate': round(avg_depletion, 3),
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }

    def optimize_lighting_schedule(self,
                                   plant_type: str = "general",
                                   target_dli: float = 15) -> Dict[str, Any]:
        """
        Optimize lighting schedule for target Daily Light Integral
        target_dli: Target DLI in mol/m²/day (12-17 for most crops)
        """
        # Assume grow lights provide ~300 μmol/m²/s at 100% intensity
        light_output_per_second = 300  # μmol/m²/s at 100%

        # Calculate hours needed to reach target DLI
        # DLI = (intensity * hours * 3600) / 1,000,000
        hours_needed = (target_dli * 1_000_000) / (light_output_per_second * 3600)

        # Round to reasonable photoperiod
        hours_needed = round(hours_needed)
        hours_needed = max(12, min(18, hours_needed))  # Clamp to 12-18 hours

        # Create schedule
        schedule = {
            'photoperiod_hours': hours_needed,
            'recommended_start_time': "06:00",
            'recommended_end_time': f"{(6 + hours_needed) % 24:02d}:00",
            'intensity_percent': 100,
            'target_dli': target_dli,
            'estimated_dli': round((light_output_per_second * hours_needed * 3600) / 1_000_000, 1)
        }

        # Energy optimization variant
        # Use natural light during day, supplement as needed
        energy_optimized = {
            'use_natural_light': True,
            'supplement_hours': '06:00-09:00, 17:00-20:00',
            'intensity_percent': 70,
            'estimated_energy_savings': '40%'
        }

        return {
            'standard_schedule': schedule,
            'energy_optimized_schedule': energy_optimized,
            'plant_type': plant_type,
            'timestamp': datetime.now().isoformat()
        }

    def multi_objective_optimization(self,
                                    current_conditions: Dict[str, float],
                                    weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimize for multiple objectives simultaneously
        weights: Relative importance of each objective (growth, efficiency, sustainability)
        """
        if weights is None:
            weights = {
                'growth': 0.5,
                'efficiency': 0.3,
                'sustainability': 0.2
            }

        # Growth score (0-100)
        growth_score = self._calculate_growth_score(current_conditions)

        # Efficiency score (inverse of resource usage)
        efficiency_score = self._calculate_efficiency_score(current_conditions)

        # Sustainability score (renewable energy usage, water conservation)
        sustainability_score = self._calculate_sustainability_score(current_conditions)

        # Combined score
        combined_score = (
            weights['growth'] * growth_score +
            weights['efficiency'] * efficiency_score +
            weights['sustainability'] * sustainability_score
        )

        # Generate balanced recommendations
        recommendations = self._generate_balanced_recommendations(
            current_conditions,
            weights
        )

        return {
            'scores': {
                'growth': round(growth_score, 1),
                'efficiency': round(efficiency_score, 1),
                'sustainability': round(sustainability_score, 1),
                'combined': round(combined_score, 1)
            },
            'weights': weights,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_adjustment_priority(self, parameter: str, magnitude: float) -> int:
        """Calculate priority for parameter adjustment"""
        # Critical parameters get higher priority
        critical_params = ['temperature', 'soil_moisture']

        base_priority = 10 if parameter in critical_params else 5
        magnitude_factor = min(magnitude / 10, 5)

        return int(base_priority + magnitude_factor)

    def _calculate_growth_score(self, conditions: Dict[str, float]) -> float:
        """Calculate growth favorability score"""
        optimal = {
            'temperature': 22,
            'humidity': 60,
            'soil_moisture': 50,
            'light': 25000
        }

        score = 0
        count = 0

        for param, optimal_val in optimal.items():
            if param in conditions:
                deviation = abs(conditions[param] - optimal_val) / optimal_val
                param_score = max(0, 100 * (1 - deviation))
                score += param_score
                count += 1

        return score / count if count > 0 else 50

    def _calculate_efficiency_score(self, conditions: Dict[str, float]) -> float:
        """Calculate resource efficiency score"""
        # Lower resource usage = higher efficiency
        resource_usage = conditions.get('light', 25000) / 50000  # Normalize to 0-1
        resource_usage += conditions.get('temperature', 22) / 30

        efficiency = 100 * (1 - resource_usage / 2)
        return max(0, min(100, efficiency))

    def _calculate_sustainability_score(self, conditions: Dict[str, float]) -> float:
        """Calculate sustainability score"""
        # Placeholder - would integrate renewable energy %, water recycling, etc.
        return 75.0

    def _generate_balanced_recommendations(self,
                                          conditions: Dict[str, float],
                                          weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate recommendations balancing multiple objectives"""
        recommendations = []

        # If growth is prioritized, recommend optimal conditions
        if weights['growth'] > 0.4:
            recommendations.append({
                'objective': 'growth',
                'suggestion': 'Maintain conditions close to optimal ranges',
                'parameters': {'temperature': 22, 'humidity': 60, 'soil_moisture': 50}
            })

        # If efficiency is prioritized, recommend resource conservation
        if weights['efficiency'] > 0.4:
            recommendations.append({
                'objective': 'efficiency',
                'suggestion': 'Use wider acceptable ranges to reduce resource consumption',
                'parameters': {'temperature': '18-26', 'lighting_hours': '12-14'}
            })

        return recommendations
