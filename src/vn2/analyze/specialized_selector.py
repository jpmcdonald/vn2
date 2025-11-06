"""
Specialized ensemble model selector for cost optimization.

Instead of selecting one "best" model per SKU, this implements a multi-objective
approach using different models for different purposes:
1. Stockout Risk Classifier - High precision for predicting zero inventory
2. Overstock Risk Detector - High precision for excess inventory scenarios  
3. Demand Density Specialist - Most accurate PMF shape for variable demand

This aligns with the research insight that chasing forecast accuracy doesn't
necessarily improve financial results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .sequential_planner import Costs


@dataclass
class SpecializedModels:
    """Container for specialized model assignments per SKU."""
    stockout_model: str
    overstock_model: str
    density_model: str
    confidence_scores: Dict[str, float]


def analyze_model_specializations(
    eval_results_path: Path,
    week4_results_path: Path
) -> pd.DataFrame:
    """
    Analyze which models are best at different tasks.
    
    Args:
        eval_results_path: Path to evaluation results
        week4_results_path: Path to Week 4 analysis results
    
    Returns:
        DataFrame with model specialization scores
    """
    # Load results
    eval_df = pd.read_parquet(eval_results_path) if eval_results_path.exists() else None
    week4_df = pd.read_csv(week4_results_path) if week4_results_path.exists() else None
    
    if week4_df is None:
        raise ValueError(f"Week 4 results not found: {week4_results_path}")
    
    specializations = []
    
    for model in week4_df['model'].unique():
        model_df = week4_df[week4_df['model'] == model]
        
        # Stockout prediction capability
        # Good at predicting when stockouts will occur (high shortage cost scenarios)
        shortage_cases = model_df[model_df['shortage_cost'] > 0]
        if len(shortage_cases) > 0:
            # Precision: When model predicts high cost, how often is there actually a shortage?
            high_pred_shortage = shortage_cases[shortage_cases['expected_cost'] > shortage_cases['expected_cost'].median()]
            stockout_precision = len(high_pred_shortage[high_pred_shortage['shortage_cost'] > 0]) / max(len(high_pred_shortage), 1)
            # Recall: When there are actual shortages, how often does model predict high cost?
            actual_shortages = shortage_cases[shortage_cases['shortage_cost'] > 10]
            stockout_recall = len(actual_shortages[actual_shortages['expected_cost'] > actual_shortages['expected_cost'].median()]) / max(len(actual_shortages), 1)
            stockout_f1 = 2 * stockout_precision * stockout_recall / max(stockout_precision + stockout_recall, 0.01)
        else:
            stockout_precision = stockout_recall = stockout_f1 = 0.0
        
        # Overstock prediction capability  
        # Good at predicting when holding costs will be high
        holding_cases = model_df[model_df['holding_cost'] > 0]
        if len(holding_cases) > 0:
            high_pred_holding = holding_cases[holding_cases['expected_cost'] > holding_cases['expected_cost'].median()]
            overstock_precision = len(high_pred_holding[high_pred_holding['holding_cost'] > 0]) / max(len(high_pred_holding), 1)
            actual_overstock = holding_cases[holding_cases['holding_cost'] > 2]
            overstock_recall = len(actual_overstock[actual_overstock['expected_cost'] > actual_overstock['expected_cost'].median()]) / max(len(actual_overstock), 1)
            overstock_f1 = 2 * overstock_precision * overstock_recall / max(overstock_precision + overstock_recall, 0.01)
        else:
            overstock_precision = overstock_recall = overstock_f1 = 0.0
        
        # Density accuracy (calibration quality)
        coverage = model_df['within_ci'].mean()
        calibration_score = 1.0 - abs(coverage - 0.9)  # How close to 90% coverage
        
        # Cost prediction accuracy
        cost_mae = np.mean(np.abs(model_df['cost_difference']))
        cost_accuracy = 1.0 / (1.0 + cost_mae)  # Higher is better
        
        # Overall stability (low variance in performance)
        cost_std = np.std(model_df['cost_difference'])
        stability_score = 1.0 / (1.0 + cost_std)
        
        specializations.append({
            'model': model,
            'n_skus': len(model_df),
            'stockout_precision': stockout_precision,
            'stockout_recall': stockout_recall,
            'stockout_f1': stockout_f1,
            'overstock_precision': overstock_precision,
            'overstock_recall': overstock_recall,
            'overstock_f1': overstock_f1,
            'calibration_score': calibration_score,
            'cost_accuracy': cost_accuracy,
            'stability_score': stability_score,
            'total_cost_error': model_df['cost_difference'].sum(),
            # Composite scores for each specialization
            'stockout_specialist_score': stockout_f1 * stability_score,
            'overstock_specialist_score': overstock_f1 * stability_score, 
            'density_specialist_score': calibration_score * cost_accuracy * stability_score
        })
    
    return pd.DataFrame(specializations).sort_values('total_cost_error')


def select_specialized_models_per_sku(
    week4_results_path: Path,
    model_specializations: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Select specialized models for each SKU based on their characteristics.
    
    Args:
        week4_results_path: Path to Week 4 analysis results
        model_specializations: DataFrame with model specialization scores
        output_path: Optional path to save results
    
    Returns:
        DataFrame with specialized model assignments per SKU
    """
    week4_df = pd.read_csv(week4_results_path)
    
    # Create model rankings for each specialization
    stockout_models = model_specializations.nlargest(5, 'stockout_specialist_score')['model'].tolist()
    overstock_models = model_specializations.nlargest(5, 'overstock_specialist_score')['model'].tolist()
    density_models = model_specializations.nlargest(5, 'density_specialist_score')['model'].tolist()
    
    print("Top specialists:")
    print(f"Stockout: {stockout_models[:3]}")
    print(f"Overstock: {overstock_models[:3]}")
    print(f"Density: {density_models[:3]}")
    
    specialized_assignments = []
    
    for (store, product), sku_df in week4_df.groupby(['store', 'product']):
        # Analyze SKU characteristics
        sku_models = sku_df['model'].tolist()
        
        # Historical performance indicators
        total_shortage = sku_df['shortage_cost'].sum()
        total_holding = sku_df['holding_cost'].sum()
        avg_demand = sku_df['actual_demand'].mean()
        demand_variability = sku_df['actual_demand'].std() / max(avg_demand, 0.1)
        
        # Classify SKU type
        if total_shortage > total_holding * 2:
            sku_type = 'shortage_prone'
        elif total_holding > total_shortage * 2:
            sku_type = 'overstock_prone'
        elif demand_variability > 1.0:
            sku_type = 'high_variability'
        else:
            sku_type = 'stable'
        
        # Select models based on SKU type and availability
        def select_best_available(preferred_models: List[str], available_models: List[str]) -> str:
            for model in preferred_models:
                if model in available_models:
                    return model
            return available_models[0] if available_models else 'slurp_bootstrap'  # fallback
        
        # Specialized model selection
        if sku_type == 'shortage_prone':
            stockout_model = select_best_available(stockout_models, sku_models)
            overstock_model = select_best_available(overstock_models + density_models, sku_models)
            density_model = select_best_available(density_models, sku_models)
        elif sku_type == 'overstock_prone':
            stockout_model = select_best_available(density_models, sku_models)
            overstock_model = select_best_available(overstock_models, sku_models)
            density_model = select_best_available(density_models, sku_models)
        elif sku_type == 'high_variability':
            # Use best density model for all purposes
            best_density = select_best_available(density_models, sku_models)
            stockout_model = best_density
            overstock_model = best_density
            density_model = best_density
        else:  # stable
            # Use balanced approach
            stockout_model = select_best_available(stockout_models + density_models, sku_models)
            overstock_model = select_best_available(overstock_models + density_models, sku_models)
            density_model = select_best_available(density_models, sku_models)
        
        # Calculate confidence scores based on model performance on this SKU
        confidence_scores = {}
        for model_name in [stockout_model, overstock_model, density_model]:
            model_perf = sku_df[sku_df['model'] == model_name]
            if len(model_perf) > 0:
                # Confidence based on how well model performed vs others for this SKU
                relative_error = model_perf['cost_difference'].iloc[0] / max(sku_df['cost_difference'].mean(), 0.1)
                confidence_scores[model_name] = max(0.1, 1.0 - abs(relative_error))
            else:
                confidence_scores[model_name] = 0.5  # default
        
        specialized_assignments.append({
            'store': store,
            'product': product,
            'sku_type': sku_type,
            'stockout_model': stockout_model,
            'overstock_model': overstock_model,
            'density_model': density_model,
            'stockout_confidence': confidence_scores.get(stockout_model, 0.5),
            'overstock_confidence': confidence_scores.get(overstock_model, 0.5),
            'density_confidence': confidence_scores.get(density_model, 0.5),
            'total_shortage': total_shortage,
            'total_holding': total_holding,
            'demand_variability': demand_variability
        })
    
    result_df = pd.DataFrame(specialized_assignments)
    
    if output_path:
        result_df.to_parquet(output_path)
        print(f"Saved specialized model assignments to: {output_path}")
    
    return result_df


def create_ensemble_forecasts(
    specialized_assignments: pd.DataFrame,
    checkpoints_dir: Path,
    output_dir: Path,
    target_week: int = 5
) -> None:
    """
    Create ensemble forecasts using specialized model assignments.
    
    This combines forecasts from different models based on their specializations
    to create more robust cost estimates.
    
    Args:
        specialized_assignments: DataFrame with model assignments per SKU
        checkpoints_dir: Path to forecast checkpoints
        output_dir: Path to save ensemble forecasts
        target_week: Week to create forecasts for
    """
    output_dir.mkdir(exist_ok=True)
    ensemble_forecasts = []
    
    for _, assignment in specialized_assignments.iterrows():
        store = int(assignment['store'])
        product = int(assignment['product'])
        
        # Load forecasts from each specialized model
        forecasts = {}
        for role in ['stockout', 'overstock', 'density']:
            model_name = assignment[f'{role}_model']
            confidence = assignment[f'{role}_confidence']
            
            checkpoint_path = checkpoints_dir / model_name / f'{store}_{product}' / f'fold_{target_week-1}.pkl'
            
            if checkpoint_path.exists():
                try:
                    import pickle
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    if 'quantiles' in checkpoint:
                        quantiles_df = checkpoint['quantiles']
                        # Use h1 forecast (next period demand)
                        if 1 in quantiles_df.index:
                            forecasts[role] = {
                                'quantiles': quantiles_df.loc[1].values,
                                'confidence': confidence,
                                'model': model_name
                            }
                except Exception as e:
                    continue
        
        if len(forecasts) > 0:
            # Create ensemble forecast
            # Weight by confidence and specialization purpose
            ensemble_quantiles = None
            total_weight = 0
            
            for role, forecast_data in forecasts.items():
                weight = forecast_data['confidence']
                if role == 'density':
                    weight *= 1.5  # Give density models higher weight for overall forecast
                
                if ensemble_quantiles is None:
                    ensemble_quantiles = forecast_data['quantiles'] * weight
                else:
                    ensemble_quantiles += forecast_data['quantiles'] * weight
                
                total_weight += weight
            
            if total_weight > 0:
                ensemble_quantiles /= total_weight
                
                ensemble_forecasts.append({
                    'store': store,
                    'product': product,
                    'ensemble_quantiles': ensemble_quantiles.tolist(),
                    'models_used': {role: data['model'] for role, data in forecasts.items()},
                    'confidences': {role: data['confidence'] for role, data in forecasts.items()},
                    'sku_type': assignment['sku_type']
                })
    
    # Save ensemble forecasts
    ensemble_df = pd.DataFrame(ensemble_forecasts)
    ensemble_output_path = output_dir / f'ensemble_forecasts_week{target_week}.parquet'
    ensemble_df.to_parquet(ensemble_output_path)
    
    print(f"Created ensemble forecasts for {len(ensemble_forecasts)} SKUs")
    print(f"Saved to: {ensemble_output_path}")


def main():
    """Main entry point for testing."""
    week4_results_path = Path('models/results/week4_expected_vs_realized.csv')
    eval_results_path = Path('models/results/eval_agg_v3.parquet')
    
    if not week4_results_path.exists():
        print(f"Error: {week4_results_path} not found")
        return
    
    print('='*80)
    print('SPECIALIZED ENSEMBLE MODEL SELECTOR')
    print('='*80)
    print()
    
    # Analyze model specializations
    print("Analyzing model specializations...")
    specializations = analyze_model_specializations(eval_results_path, week4_results_path)
    
    print("MODEL SPECIALIZATION SCORES:")
    print('-'*80)
    print(specializations[['model', 'stockout_f1', 'overstock_f1', 'calibration_score', 
                          'stockout_specialist_score', 'overstock_specialist_score', 
                          'density_specialist_score']].round(3).to_string(index=False))
    print()
    
    # Select specialized models per SKU
    print("Selecting specialized models per SKU...")
    assignments = select_specialized_models_per_sku(
        week4_results_path,
        specializations,
        Path('models/results/specialized_model_assignments.parquet')
    )
    
    print()
    print("SKU TYPE DISTRIBUTION:")
    print(assignments['sku_type'].value_counts())
    print()
    
    print("MODEL USAGE BY SPECIALIZATION:")
    print("Stockout models:", assignments['stockout_model'].value_counts().head())
    print("Overstock models:", assignments['overstock_model'].value_counts().head())
    print("Density models:", assignments['density_model'].value_counts().head())


if __name__ == '__main__':
    main()
