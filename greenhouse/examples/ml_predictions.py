"""
Example: Machine Learning Predictions
Demonstrates ML-based condition prediction and growth analysis
"""
import time
import random
from datetime import datetime, timedelta
from greenhouse.ml_models.growth_predictor import GrowthPredictor
from greenhouse.ml_models.optimization_engine import OptimizationEngine


def main():
    print("="*60)
    print("Machine Learning Predictions Example")
    print("="*60 + "\n")

    # Simulate current greenhouse conditions
    current_conditions = {
        'temperature': 21.5,
        'humidity': 62.0,
        'soil_moisture': 48.0,
        'light': 22000,
        'co2': 950
    }

    print("Current Greenhouse Conditions:")
    for param, value in current_conditions.items():
        print(f"  {param:15s}: {value:.1f}")

    # 1. Growth Analysis
    print("\n" + "="*60)
    print("1. GROWTH ANALYSIS")
    print("="*60)

    growth_predictor = GrowthPredictor()

    # Estimate growth rate
    growth_metrics = growth_predictor.estimate_growth_rate(current_conditions)

    print(f"\nGrowth Score: {growth_metrics['growth_score']}/100")
    print(f"Category: {growth_metrics['growth_rate_category']}")
    print(f"Estimated Daily Growth: {growth_metrics['estimated_daily_growth_cm']} cm/day")

    if growth_metrics['limiting_factors']:
        print("\nLimiting Factors:")
        for factor in growth_metrics['limiting_factors']:
            print(f"  • {factor['parameter']}: {factor['current']:.1f} ({factor['status']})")
    else:
        print("\n✓ All conditions within optimal range!")

    # 2. Yield Prediction
    print("\n" + "="*60)
    print("2. YIELD PREDICTION")
    print("="*60)

    plant_age_days = 45
    target_harvest_days = 90

    yield_prediction = growth_predictor.predict_yield(
        current_conditions=current_conditions,
        plant_age_days=plant_age_days,
        target_harvest_days=target_harvest_days
    )

    print(f"\nPlant Age: {yield_prediction['plant_age_days']} days")
    print(f"Days to Harvest: {yield_prediction['days_to_harvest']} days")
    print(f"Estimated Yield: {yield_prediction['estimated_yield_kg']} kg per plant")
    print(f"Yield Category: {yield_prediction['yield_category']}")
    print(f"Prediction Confidence: {yield_prediction['prediction_confidence']}%")

    # 3. Improvement Recommendations
    print("\n" + "="*60)
    print("3. IMPROVEMENT RECOMMENDATIONS")
    print("="*60)

    recommendations = growth_predictor.recommend_improvements(current_conditions)

    if recommendations:
        print("\nRecommended Actions:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['action']}")
            print(f"   Current: {rec['current_value']}")
            print(f"   Target: {rec['target_range']}")
            print(f"   Suggestion: {rec['suggestion']}")
    else:
        print("\n✓ No improvements needed - conditions are optimal!")

    # 4. Optimization for Growth
    print("\n" + "="*60)
    print("4. OPTIMIZATION FOR MAXIMUM GROWTH")
    print("="*60)

    optimizer = OptimizationEngine()

    actuator_constraints = {
        'temperature': (15, 30),
        'humidity': (40, 80),
        'soil_moisture': (30, 70),
        'light': (10000, 40000)
    }

    optimization = optimizer.optimize_for_growth(current_conditions, actuator_constraints)

    print("\nOptimization Recommendations:")
    for param, rec in optimization['recommendations'].items():
        if rec.get('adjustment_needed', 0) != 0:
            print(f"  {param:15s}: {rec['current']:.1f} → {rec['target']:.1f} "
                  f"(adjust by {rec['adjustment_needed']:+.1f})")

    print("\nPriority Actions:")
    for i, action in enumerate(optimization['priority_actions'][:3], 1):
        print(f"  {i}. {action['action'].upper()} {action['parameter']} "
              f"by {action['magnitude']:.1f} (priority: {action['priority']})")

    # 5. Multi-Objective Optimization
    print("\n" + "="*60)
    print("5. MULTI-OBJECTIVE OPTIMIZATION")
    print("="*60)

    # Balance growth, efficiency, and sustainability
    weights = {
        'growth': 0.5,
        'efficiency': 0.3,
        'sustainability': 0.2
    }

    multi_opt = optimizer.multi_objective_optimization(current_conditions, weights)

    print("\nObjective Scores:")
    for objective, score in multi_opt['scores'].items():
        print(f"  {objective:15s}: {score:.1f}/100")

    # 6. Irrigation Optimization
    print("\n" + "="*60)
    print("6. IRRIGATION OPTIMIZATION")
    print("="*60)

    # Simulate moisture history
    soil_moisture_history = [52, 51, 49, 48, 47, 45, 44, 42, 41, 40, 38]

    irrigation_opt = optimizer.optimize_irrigation_schedule(
        soil_moisture_history=soil_moisture_history,
        evapotranspiration_rate=0.5
    )

    print(f"\nCurrent Moisture: {irrigation_opt['current_moisture']}%")
    print(f"Target Range: {irrigation_opt['target_range']}")
    print(f"Depletion Rate: {irrigation_opt['depletion_rate']:.3f} %/hour")
    print(f"\nRecommendation: {irrigation_opt['recommendation']['action'].upper()}")
    print(f"Reason: {irrigation_opt['recommendation']['reason']}")
    if 'duration_seconds' in irrigation_opt['recommendation']:
        print(f"Duration: {irrigation_opt['recommendation']['duration_seconds']} seconds")

    # 7. Lighting Optimization
    print("\n" + "="*60)
    print("7. LIGHTING OPTIMIZATION")
    print("="*60)

    lighting_opt = optimizer.optimize_lighting_schedule(
        plant_type="tomatoes",
        target_dli=15
    )

    print("\nStandard Schedule:")
    schedule = lighting_opt['standard_schedule']
    print(f"  Photoperiod: {schedule['photoperiod_hours']} hours")
    print(f"  Start Time: {schedule['recommended_start_time']}")
    print(f"  End Time: {schedule['recommended_end_time']}")
    print(f"  Intensity: {schedule['intensity_percent']}%")
    print(f"  Estimated DLI: {schedule['estimated_dli']} mol/m²/day")

    print("\nEnergy-Optimized Schedule:")
    energy_opt = lighting_opt['energy_optimized_schedule']
    print(f"  Use Natural Light: {energy_opt['use_natural_light']}")
    print(f"  Supplement Hours: {energy_opt['supplement_hours']}")
    print(f"  Intensity: {energy_opt['intensity_percent']}%")
    print(f"  Energy Savings: {energy_opt['estimated_energy_savings']}")

    print("\n" + "="*60)
    print("ML Predictions example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
