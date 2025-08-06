#!/usr/bin/env python3
"""
02-01-HEIM-CORE: Core HEIM Calculator Implementation (FULLY FIXED VERSION)

SAVE AS: PYTHON/02-01-heim-core.py
RUN AS: python PYTHON/02-01-heim-core.py (for testing)

Core mathematical implementation of HEIM (Health Equity Informative Marker)
Following directory hierarchy - run from root, output to ANALYSIS/02-00-HEIM-ANALYSIS/

FIXES APPLIED:
- Harmonic mean now handles zeros properly
- Geographic coverage uses better normalization
- Aggregation methods don't collapse to zero
- DataFrame truth value check fixed
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import RobustScaler
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class HEIMConfig:
    """Configuration for HEIM calculation"""
    # Component weights (context-dependent)
    weight_representation: float = 0.4
    weight_structure: float = 0.3
    weight_function: float = 0.3
    
    # Normalization parameters
    use_log_scale_burden: bool = True  # Log scale for 8 orders of magnitude
    aggregation_method: str = 'arithmetic'  # CHANGED: 'arithmetic' is more stable than 'harmonic'
    
    # Sparsity handling
    sparsity_penalty_threshold: float = 0.01  # Papers/expected ratio below this gets penalty
    missing_data_imputation: str = 'burden_weighted'  # 'zero', 'mean', 'burden_weighted'
    
    # Temporal parameters
    temporal_window_years: int = 5  # For trend analysis
    temporal_bonus_weight: float = 0.1  # Bonus for positive trends
    
    # Output directory
    output_dir: str = "ANALYSIS/02-00-HEIM-ANALYSIS"

class HEIMCalculator:
    """
    Core HEIM calculator with mathematical formalization
    Designed to work with the project's directory structure
    FIXED: Proper handling of zeros, normalization, and DataFrame checks
    """
    
    def __init__(self, config: Optional[HEIMConfig] = None):
        self.config = config or HEIMConfig()
        self.scaler = RobustScaler()  # Robust to outliers
        self.burden_data = None
        self.normalization_params = {}
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def set_burden_data(self, burden_df: pd.DataFrame):
        """Set global burden data for weighting"""
        self.burden_data = burden_df
        
        # Precompute burden normalization
        if self.config.use_log_scale_burden:
            # Log transform for diseases spanning 8 orders of magnitude
            burden_values = burden_df['dalys'].values if 'dalys' in burden_df.columns else burden_df['total_burden'].values
            burden_values = np.where(burden_values > 0, burden_values, 1)  # Avoid log(0)
            self.burden_weights = np.log10(burden_values + 1)
            self.burden_weights = self.burden_weights / self.burden_weights.sum()
        else:
            burden_values = burden_df['dalys'].values if 'dalys' in burden_df.columns else burden_df['total_burden'].values
            self.burden_weights = burden_values / burden_values.sum()
    
    def calculate_h_r(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Calculate H-R (Representation) component
        Diversity of ancestry, variants, exposures, social determinants
        """
        components = {}
        
        # 1. Ancestry diversity (Simpson's index with burden weighting)
        if 'ancestry_counts' in data:
            ancestry_diversity = self._calculate_weighted_diversity(
                data['ancestry_counts'],
                apply_burden_weight=True
            )
            components['ancestry_diversity'] = ancestry_diversity
        else:
            components['ancestry_diversity'] = 0.0
        
        # 2. Geographic representation (normalized by disease burden)
        if 'geographic_coverage' in data:
            geo_score = self._normalize_geographic_coverage(
                data['geographic_coverage'],
                data.get('disease_burden_by_region', {})
            )
            components['geographic_representation'] = geo_score
        else:
            components['geographic_representation'] = 0.0
        
        # 3. Disease coverage (weighted by global burden)
        if 'disease_coverage' in data:
            disease_score = self._calculate_disease_coverage_score(
                data['disease_coverage']
            )
            components['disease_coverage'] = disease_score
        else:
            components['disease_coverage'] = 0.0
        
        # 4. Social determinants representation
        if 'social_determinants' in data:
            sdoh_score = len(data['social_determinants']) / 10.0  # Normalize to 10 key SDOH
            components['social_determinants'] = min(1.0, sdoh_score)
        else:
            components['social_determinants'] = 0.0
        
        # Aggregate H-R score
        h_r = self._aggregate_scores(list(components.values()))
        
        # Apply sparsity penalty if needed
        if 'total_samples' in data and 'expected_samples' in data:
            sparsity_ratio = data['total_samples'] / max(1, data['expected_samples'])
            if sparsity_ratio < self.config.sparsity_penalty_threshold:
                penalty = np.log10(sparsity_ratio * 100) / 2  # Log penalty
                h_r *= max(0.5, 1 + penalty)  # Apply penalty but keep >= 0.5
        
        return h_r, components
    
    def calculate_h_s(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Calculate H-S (Structure) component
        Governance, access, consent models, community participation
        """
        components = {}
        
        # 1. Governance score
        governance_factors = {
            'community_board': 0.25,
            'indigenous_leadership': 0.25,
            'transparent_policies': 0.2,
            'benefit_sharing': 0.3
        }
        governance_score = sum(
            weight for factor, weight in governance_factors.items()
            if data.get('governance', {}).get(factor, False)
        )
        components['governance'] = governance_score
        
        # 2. Access equity
        if 'access_model' in data:
            access_scores = {
                'open': 1.0,
                'registered': 0.8,
                'controlled': 0.6,
                'restricted': 0.3,
                'closed': 0.0
            }
            components['access'] = access_scores.get(data['access_model'], 0.5)
        else:
            components['access'] = 0.5
        
        # 3. Consent model appropriateness
        if 'consent_model' in data:
            consent_scores = {
                'dynamic': 1.0,
                'tiered': 0.9,
                'broad': 0.7,
                'specific': 0.5,
                'presumed': 0.2
            }
            components['consent'] = consent_scores.get(data['consent_model'], 0.5)
        else:
            components['consent'] = 0.5
        
        # 4. Community participation
        if 'community_engagement_score' in data:
            components['community'] = min(1.0, data['community_engagement_score'])
        else:
            components['community'] = 0.0
        
        # Aggregate H-S score
        h_s = self._aggregate_scores(list(components.values()))
        
        return h_s, components
    
    def calculate_h_f(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Calculate H-F (Function) component
        Equity of predictions, clinical performance, deployment outcomes
        """
        components = {}
        
        # 1. Performance equity across populations
        if 'performance_by_population' in data:
            # Calculate coefficient of variation (lower is better)
            performances = list(data['performance_by_population'].values())
            if len(performances) > 1:
                cv = stats.variation(performances)
                equity_score = 1.0 / (1.0 + cv)  # Convert to 0-1 where 1 is perfect equity
            else:
                equity_score = 0.5
            components['performance_equity'] = equity_score
        else:
            components['performance_equity'] = 0.5
        
        # 2. Clinical deployment coverage
        if 'deployment_regions' in data and 'total_regions' in data:
            deployment_score = len(data['deployment_regions']) / data['total_regions']
            components['deployment_coverage'] = min(1.0, deployment_score)
        else:
            components['deployment_coverage'] = 0.0
        
        # 3. Health outcome improvements (burden-weighted)
        if 'outcome_improvements' in data:
            outcome_score = self._calculate_outcome_improvement_score(
                data['outcome_improvements']
            )
            components['outcome_impact'] = outcome_score
        else:
            components['outcome_impact'] = 0.0
        
        # 4. Bias mitigation effectiveness
        if 'bias_metrics' in data:
            bias_score = 1.0 - np.mean([
                abs(metric) for metric in data['bias_metrics'].values()
            ])
            components['bias_mitigation'] = max(0.0, bias_score)
        else:
            components['bias_mitigation'] = 0.5
        
        # Aggregate H-F score
        h_f = self._aggregate_scores(list(components.values()))
        
        return h_f, components
    
    def calculate_heim(self, data: Dict[str, Any], 
                      context: str = 'default') -> Dict[str, Any]:
        """
        Calculate complete HEIM score with all components
        
        Args:
            data: Input data dictionary
            context: Context for weight selection ('genomics', 'clinical', 'public_health')
        
        Returns:
            Dictionary with HEIM score and detailed breakdown
        """
        # Adjust weights based on context
        weights = self._get_context_weights(context)
        
        # Calculate components
        h_r, h_r_details = self.calculate_h_r(data)
        h_s, h_s_details = self.calculate_h_s(data)
        h_f, h_f_details = self.calculate_h_f(data)
        
        # Apply temporal bonus if improving
        temporal_bonus = 0.0
        if 'temporal_trend' in data:
            trend = data['temporal_trend']
            if trend > 0:  # Positive trend
                temporal_bonus = min(0.1, trend * self.config.temporal_bonus_weight)
        
        # Calculate weighted HEIM score
        heim_raw = (
            weights['w_r'] * h_r +
            weights['w_s'] * h_s +
            weights['w_f'] * h_f
        )
        
        # Apply temporal bonus
        heim_final = min(1.0, heim_raw * (1 + temporal_bonus))
        
        # Determine interpretation
        interpretation = self._interpret_heim_score(heim_final)
        
        return {
            'heim_score': heim_final,
            'components': {
                'H-R': h_r,
                'H-S': h_s,
                'H-F': h_f
            },
            'weights': weights,
            'details': {
                'representation': h_r_details,
                'structure': h_s_details,
                'function': h_f_details
            },
            'temporal_bonus': temporal_bonus,
            'interpretation': interpretation,
            'context': context
        }
    
    def _calculate_weighted_diversity(self, counts: Dict, 
                                     apply_burden_weight: bool = False) -> float:
        """Calculate diversity index with optional burden weighting"""
        if not counts:
            return 0.0
        
        values = np.array(list(counts.values()))
        total = values.sum()
        
        if total == 0:
            return 0.0
        
        # Simpson's diversity index
        proportions = values / total
        simpson = 1 - np.sum(proportions ** 2)
        
        # Apply burden weighting if needed
        if apply_burden_weight and self.burden_data is not None:
            # Weight by disease burden representation
            burden_factor = self._get_burden_representation_factor(counts)
            simpson *= burden_factor
        
        return simpson
    
    def _normalize_geographic_coverage(self, coverage: Dict, 
                                      burden_by_region: Dict) -> float:
        """Normalize geographic coverage by regional disease burden"""
        if not coverage:
            return 0.0
        
        if not burden_by_region:
            # Better normalization: consider both diversity and sample size
            num_countries = len(coverage)
            total_samples = sum(coverage.values())
            
            # Diversity score (more countries is better)
            # Use sigmoid: good coverage at 20+ countries
            diversity_score = 1 - np.exp(-num_countries / 20)
            
            # Sample size score (more samples is better)
            # Use sigmoid: good coverage at 10000+ samples
            sample_score = 1 - np.exp(-total_samples / 10000)
            
            # Combine both aspects
            return (diversity_score + sample_score) / 2
        
        weighted_coverage = 0.0
        total_burden = sum(burden_by_region.values())
        
        for region, samples in coverage.items():
            region_burden = burden_by_region.get(region, 0)
            if total_burden > 0:
                weight = region_burden / total_burden
                # Sigmoid normalization for sample counts
                normalized_samples = 1 - np.exp(-samples / 1000)  # Asymptotic at 1000 samples
                weighted_coverage += weight * normalized_samples
        
        return min(1.0, weighted_coverage)
    
    def _calculate_disease_coverage_score(self, disease_coverage: Dict) -> float:
        """Calculate disease coverage weighted by global burden"""
        # FIXED: Proper DataFrame existence check
        if self.burden_data is None or (isinstance(self.burden_data, pd.DataFrame) and self.burden_data.empty):
            # Simple proportion without burden weighting
            if not disease_coverage:
                return 0.0
            covered = sum(1 for v in disease_coverage.values() if v)
            return covered / max(1, len(disease_coverage))
        
        covered_burden = 0.0
        total_burden = self.burden_data['dalys'].sum() if 'dalys' in self.burden_data.columns else self.burden_data['total_burden'].sum()
        
        for disease, has_data in disease_coverage.items():
            if has_data:
                disease_data = self.burden_data[self.burden_data['disease'] == disease]
                if len(disease_data) > 0:
                    disease_burden = disease_data.iloc[0]['dalys'] if 'dalys' in disease_data.columns else disease_data.iloc[0]['total_burden']
                    covered_burden += disease_burden
        
        return covered_burden / total_burden if total_burden > 0 else 0.0
    
    def _calculate_outcome_improvement_score(self, improvements: Dict) -> float:
        """Calculate outcome improvement score weighted by burden"""
        if not improvements:
            return 0.0
        
        weighted_improvement = 0.0
        
        for disease, improvement in improvements.items():
            weight = 1.0  # Default weight
            if self.burden_data is not None and not self.burden_data.empty:
                disease_data = self.burden_data[self.burden_data['disease'] == disease]
                if len(disease_data) > 0:
                    disease_burden = disease_data.iloc[0]['dalys'] if 'dalys' in disease_data.columns else disease_data.iloc[0]['total_burden']
                    # Log scale for burden weight
                    weight = np.log10(disease_burden + 1) / 10
            
            # Sigmoid transformation of improvement
            normalized_improvement = 1 - np.exp(-improvement)
            weighted_improvement += weight * normalized_improvement
        
        # Normalize by number of diseases
        return weighted_improvement / max(1, len(improvements))
    
    def _aggregate_scores(self, scores: List[float]) -> float:
        """
        Aggregate multiple scores based on configuration
        FIXED: Properly handles zeros without collapsing the score
        """
        scores = [s for s in scores if s is not None and not np.isnan(s)]
        
        if not scores:
            return 0.0
        
        if self.config.aggregation_method == 'arithmetic':
            return np.mean(scores)
        elif self.config.aggregation_method == 'geometric':
            # For geometric mean, handle zeros specially
            non_zero_scores = [s for s in scores if s > 0]
            if not non_zero_scores:
                return 0.0
            # Include zero penalty: reduce by proportion of zeros
            zero_penalty = len(non_zero_scores) / len(scores)
            geometric_mean = np.exp(np.mean(np.log(non_zero_scores)))
            return geometric_mean * zero_penalty
        elif self.config.aggregation_method == 'harmonic':
            # FIXED: Handle zeros properly in harmonic mean
            # Use a floor value instead of epsilon to prevent collapse
            floor_value = 0.1
            scores_adj = [max(s, floor_value) for s in scores]
            harmonic = len(scores_adj) / np.sum(1.0 / np.array(scores_adj))
            
            # Apply penalty for zeros but don't collapse to zero
            zero_count = sum(1 for s in scores if s == 0)
            if zero_count > 0:
                # Reduce score proportionally to number of zeros
                penalty_factor = (len(scores) - zero_count) / len(scores)
                harmonic = harmonic * (0.5 + 0.5 * penalty_factor)
            
            return harmonic
        else:
            return np.mean(scores)
    
    def _get_context_weights(self, context: str) -> Dict[str, float]:
        """Get context-specific weights for HEIM components"""
        context_weights = {
            'default': {
                'w_r': self.config.weight_representation,
                'w_s': self.config.weight_structure,
                'w_f': self.config.weight_function
            },
            'genomics': {
                'w_r': 0.5,  # Ancestry diversity critical
                'w_s': 0.2,  # Governance important
                'w_f': 0.3   # Function matters
            },
            'clinical': {
                'w_r': 0.3,  # Representation important
                'w_s': 0.2,  # Structure less critical
                'w_f': 0.5   # Function most critical
            },
            'public_health': {
                'w_r': 0.4,  # Population coverage key
                'w_s': 0.4,  # Access critical
                'w_f': 0.2   # Function less immediate
            }
        }
        
        return context_weights.get(context, context_weights['default'])
    
    def _get_burden_representation_factor(self, counts: Dict) -> float:
        """Calculate how well representation matches disease burden"""
        # This would map representation to burden distribution
        # Simplified version - returns 0.5-1.0 based on alignment
        return 0.75  # Placeholder - implement based on your burden data
    
    def _interpret_heim_score(self, score: float) -> Dict[str, str]:
        """Interpret HEIM score with actionable insights"""
        if score >= 0.8:
            level = "Excellent"
            action = "Maintain current equity practices and share as best practice"
        elif score >= 0.6:
            level = "Good"
            action = "Focus on lowest-scoring components for improvement"
        elif score >= 0.4:
            level = "Moderate"
            action = "Significant equity improvements needed across multiple dimensions"
        elif score >= 0.2:
            level = "Poor"
            action = "Major equity interventions required urgently"
        else:
            level = "Critical"
            action = "Fundamental redesign needed with equity as primary goal"
        
        return {
            'level': level,
            'score_range': f"{score:.2f}",
            'action': action,
            'target': "Aim for HEIM ≥ 0.7 for ethical deployment"
        }
    
    def save_results(self, results: Dict, output_name: str):
        """Save HEIM results following directory structure"""
        output_dir = Path(self.config.output_dir) / "BASELINE-SCORES"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_file = output_dir / f"{output_name}_heim_results.json"
        with open(json_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_safe = self._convert_for_json(results)
            json.dump(json_safe, f, indent=2)
        
        print(f"✓ Results saved to: {json_file}")
        return json_file
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def test_heim_calculator():
    """Test HEIM calculator with sample data"""
    print("="*70)
    print("TESTING HEIM CALCULATOR (FULLY FIXED VERSION)")
    print("="*70)
    
    calculator = HEIMCalculator()
    
    # Test case 1: Good equity
    good_data = {
        'ancestry_counts': {'EUR': 2500, 'AFR': 2500, 'EAS': 2500, 'SAS': 2500},
        'geographic_coverage': {'USA': 1000, 'UK': 1000, 'China': 1000, 'Kenya': 1000},
        'disease_coverage': {f'disease_{i}': True for i in range(100)},
        'social_determinants': ['education', 'income', 'occupation'],
        'governance': {
            'community_board': True, 
            'transparent_policies': True,
            'benefit_sharing': True
        },
        'access_model': 'open',
        'consent_model': 'broad',
        'performance_by_population': {'EUR': 0.95, 'AFR': 0.94, 'EAS': 0.95, 'SAS': 0.94}
    }
    
    result = calculator.calculate_heim(good_data, context='public_health')
    
    print(f"\nTest Case: Good Equity")
    print(f"HEIM Score: {result['heim_score']:.3f}")
    print(f"Interpretation: {result['interpretation']['level']}")
    print(f"Components:")
    print(f"  H-R: {result['components']['H-R']:.3f}")
    print(f"  H-S: {result['components']['H-S']:.3f}")
    print(f"  H-F: {result['components']['H-F']:.3f}")
    
    # Test case 2: Poor equity
    poor_data = {
        'ancestry_counts': {'EUR': 9500, 'AFR': 100, 'EAS': 200, 'SAS': 200},
        'geographic_coverage': {'USA': 5000, 'UK': 4000},
        'disease_coverage': {f'disease_{i}': True for i in range(20)},
        'governance': {},
        'access_model': 'restricted',
        'consent_model': 'presumed',
        'performance_by_population': {'EUR': 0.95, 'AFR': 0.65, 'EAS': 0.70, 'SAS': 0.68}
    }
    
    result_poor = calculator.calculate_heim(poor_data, context='public_health')
    
    print(f"\nTest Case: Poor Equity")
    print(f"HEIM Score: {result_poor['heim_score']:.3f}")
    print(f"Interpretation: {result_poor['interpretation']['level']}")
    print(f"Components:")
    print(f"  H-R: {result_poor['components']['H-R']:.3f}")
    print(f"  H-S: {result_poor['components']['H-S']:.3f}")
    print(f"  H-F: {result_poor['components']['H-F']:.3f}")
    
    # Verify scores make sense
    assert result['heim_score'] > result_poor['heim_score'], "Good equity should score higher than poor"
    assert result['heim_score'] > 0.5, "Good equity should score > 0.5"
    assert result_poor['heim_score'] < 0.4, "Poor equity should score < 0.4"
    
    print("\n✓ All tests passed!")
    
    # Save test results
    calculator.save_results(result, "test_good_equity")
    calculator.save_results(result_poor, "test_poor_equity")
    
    print(f"\n✓ HEIM Calculator test complete!")
    print(f"Output saved to: {calculator.config.output_dir}/BASELINE-SCORES/")

if __name__ == "__main__":
    # Test the calculator
    test_heim_calculator()