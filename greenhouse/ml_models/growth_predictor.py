"""
Plant Growth and Yield Predictor
Predicts plant growth metrics and yield estimation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class GrowthPredictor:
    """Predicts plant growth and yield based on environmental conditions"""

    def __init__(self):
        self.growth_model = None
        self.yield_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Optimal ranges for different parameters (based on common greenhouse crops)
        self.optimal_ranges = {
            'temperature': (18, 25),  # Celsius
            'humidity': (50, 70),  # Percent
            'soil_moisture': (40, 60),  # Percent
            'light': (15000, 30000),  # Lux
            'co2': (800, 1200)  # ppm
        }

    def calculate_growth_score(self, conditions: Dict[str, float]) -> float:
        """
        Calculate growth favorability score (0-100)
        Based on how close conditions are to optimal ranges
        """
        score = 0
        weight_sum = 0

        weights = {
            'temperature': 0.25,
            'humidity': 0.15,
            'soil_moisture': 0.25,
            'light': 0.25,
            'co2': 0.10
        }

        for param, (min_val, max_val) in self.optimal_ranges.items():
            if param in conditions:
                value = conditions[param]
                optimal_mid = (min_val + max_val) / 2
                optimal_range = max_val - min_val

                # Calculate how far from optimal (0 = perfect, 1 = far)
                deviation = abs(value - optimal_mid) / (optimal_range / 2)
                deviation = min(deviation, 2.0)  # Cap at 2x range

                # Convert to score (100 = perfect, 0 = poor)
                param_score = max(0, 100 * (1 - deviation / 2))

                weight = weights.get(param, 0.1)
                score += param_score * weight
                weight_sum += weight

        if weight_sum > 0:
            score = score / weight_sum

        return min(100, max(0, score))

    def estimate_growth_rate(self, conditions: Dict[str, float], plant_type: str = "general") -> Dict[str, Any]:
        """
        Estimate daily growth rate based on current conditions
        Returns growth metrics and recommendations
        """
        growth_score = self.calculate_growth_score(conditions)

        # Base growth rate (cm/day for general plants)
        base_growth = 0.5

        # Adjust based on conditions
        growth_multiplier = growth_score / 100
        estimated_growth = base_growth * growth_multiplier

        # Identify limiting factors
        limiting_factors = []
        for param, (min_val, max_val) in self.optimal_ranges.items():
            if param in conditions:
                value = conditions[param]
                if value < min_val:
                    limiting_factors.append({
                        'parameter': param,
                        'current': value,
                        'optimal_min': min_val,
                        'status': 'too_low'
                    })
                elif value > max_val:
                    limiting_factors.append({
                        'parameter': param,
                        'current': value,
                        'optimal_max': max_val,
                        'status': 'too_high'
                    })

        return {
            'growth_score': round(growth_score, 1),
            'estimated_daily_growth_cm': round(estimated_growth, 2),
            'growth_rate_category': self._categorize_growth_rate(growth_score),
            'limiting_factors': limiting_factors,
            'timestamp': datetime.now().isoformat()
        }

    def _categorize_growth_rate(self, score: float) -> str:
        """Categorize growth rate"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"

    def predict_yield(self,
                     current_conditions: Dict[str, float],
                     plant_age_days: int,
                     plant_type: str = "general",
                     target_harvest_days: int = 90) -> Dict[str, Any]:
        """
        Predict yield based on current and projected conditions
        """
        # Current growth score
        current_score = self.calculate_growth_score(current_conditions)

        # Days until harvest
        days_remaining = max(0, target_harvest_days - plant_age_days)

        # Assume conditions remain similar
        projected_score = current_score

        # Base yield (kg per plant for general crops)
        base_yield = 2.0

        # Yield multiplier based on growth score
        # Excellent conditions (95+) = 1.2x, Poor (<50) = 0.5x
        yield_multiplier = 0.3 + (projected_score / 100) * 0.9

        estimated_yield = base_yield * yield_multiplier

        # Calculate confidence based on plant age
        confidence = min(100, (plant_age_days / target_harvest_days) * 100)

        return {
            'estimated_yield_kg': round(estimated_yield, 2),
            'yield_category': self._categorize_yield(yield_multiplier),
            'days_to_harvest': days_remaining,
            'plant_age_days': plant_age_days,
            'current_growth_score': round(current_score, 1),
            'prediction_confidence': round(confidence, 1),
            'timestamp': datetime.now().isoformat()
        }

    def _categorize_yield(self, multiplier: float) -> str:
        """Categorize expected yield"""
        if multiplier >= 1.1:
            return "Above Average"
        elif multiplier >= 0.9:
            return "Average"
        elif multiplier >= 0.7:
            return "Below Average"
        else:
            return "Low"

    def recommend_improvements(self, conditions: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Provide recommendations to improve growth conditions
        """
        recommendations = []

        for param, (min_val, max_val) in self.optimal_ranges.items():
            if param in conditions:
                value = conditions[param]

                if value < min_val:
                    recommendations.append({
                        'parameter': param,
                        'current_value': round(value, 1),
                        'target_range': f"{min_val}-{max_val}",
                        'action': f"Increase {param}",
                        'priority': self._calculate_priority(value, min_val, max_val),
                        'suggestion': self._get_suggestion(param, 'increase')
                    })
                elif value > max_val:
                    recommendations.append({
                        'parameter': param,
                        'current_value': round(value, 1),
                        'target_range': f"{min_val}-{max_val}",
                        'action': f"Decrease {param}",
                        'priority': self._calculate_priority(value, min_val, max_val),
                        'suggestion': self._get_suggestion(param, 'decrease')
                    })

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return recommendations

    def _calculate_priority(self, value: float, min_val: float, max_val: float) -> str:
        """Calculate recommendation priority"""
        optimal_mid = (min_val + max_val) / 2
        optimal_range = max_val - min_val
        deviation = abs(value - optimal_mid) / optimal_range

        if deviation > 1.0:
            return "High"
        elif deviation > 0.5:
            return "Medium"
        else:
            return "Low"

    def _get_suggestion(self, parameter: str, action: str) -> str:
        """Get specific suggestions for parameter adjustment"""
        suggestions = {
            'temperature': {
                'increase': "Activate heating system or close vents",
                'decrease': "Open vents, activate cooling, or shade the greenhouse"
            },
            'humidity': {
                'increase': "Activate misting system or reduce ventilation",
                'decrease': "Increase ventilation or activate dehumidifier"
            },
            'soil_moisture': {
                'increase': "Activate irrigation system",
                'decrease': "Reduce watering frequency or improve drainage"
            },
            'light': {
                'increase': "Turn on grow lights or remove shading",
                'decrease': "Add shading or reduce artificial lighting"
            },
            'co2': {
                'increase': "Activate CO2 enrichment system",
                'decrease': "Increase ventilation"
            }
        }

        return suggestions.get(parameter, {}).get(action, f"{action} {parameter}")

    def calculate_dli(self, hourly_light_readings: List[float]) -> float:
        """
        Calculate Daily Light Integral (DLI)
        DLI is the amount of photosynthetically active radiation received per day
        """
        # Convert lux to PAR (rough approximation: PAR ≈ lux * 0.0185)
        par_readings = [lux * 0.0185 for lux in hourly_light_readings]

        # Calculate DLI in mol/m²/day
        # DLI = (average PAR in μmol/m²/s * seconds in photoperiod) / 1,000,000
        if par_readings:
            avg_par = sum(par_readings) / len(par_readings)
            hours = len(par_readings)
            dli = (avg_par * hours * 3600) / 1_000_000
            return round(dli, 2)

        return 0.0
