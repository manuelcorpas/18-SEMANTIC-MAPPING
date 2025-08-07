#!/usr/bin/env python3
"""
HEIM (Health Equity Informative Marker) - Mathematical Framework v2.0
A universal metric for biomedical data equity

Mathematical formalization addressing:
1. 8-order magnitude normalization using log-transform
2. Weighted geometric mean for equity (penalizes zeros)
3. Burden-weighted representation
4. Temporal bonus system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class HEIMConfig:
    """Configuration for HEIM calculation with mathematical rigor"""
    
    # Component weights (context-dependent)
    weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'default': {'H-R': 0.4, 'H-S': 0.3, 'H-F': 0.3},
        'genomics': {'H-R': 0.5, 'H-S': 0.2, 'H-F': 0.3},
        'clinical': {'H-R': 0.3, 'H-S': 0.2, 'H-F': 0.5},
        'public_health': {'H-R': 0.4, 'H-S': 0.4, 'H-F': 0.2}
    })
    
    # Normalization parameters
    burden_log_base: float = 10  # Log base for burden normalization
    burden_floor: float = 1000  # Minimum burden to avoid log(0)
    
    # Aggregation method
    aggregation: str = 'weighted_geometric'  # Best for equity metrics
    
    # Sparsity handling
    sparsity_penalty_type: str = 'sigmoid'  # Smooth penalty function
    sparsity_threshold: float = 0.1  # Below 10% of expected = penalty
    
    # Temporal parameters
    temporal_window: int = 5  # Years for trend calculation
    temporal_bonus_max: float = 0.15  # Max 15% bonus for improvement
    temporal_decay: float = 0.9  # Decay factor for older improvements

class HEIMCalculator:
    """
    Mathematically rigorous HEIM calculator
    
    Key innovations:
    1. Log-transform for burden spanning 8 orders of magnitude
    2. Weighted geometric mean preserving equity properties
    3. Sigmoid functions for smooth transitions
    4. Bayesian priors for missing data
    """
    
    def __init__(self, config: Optional[HEIMConfig] = None):
        self.config = config or HEIMConfig()
        self.burden_normalizer = None
        self.component_cache = {}
        
    def set_global_burden_data(self, burden_df: pd.DataFrame):
        """
        Initialize burden normalizer with log-transform
        Handles 8 orders of magnitude gracefully
        """
        burden_col = 'dalys' if 'dalys' in burden_df.columns else 'total_burden'
        
        # Log-transform with floor to handle zeros
        burden_values = burden_df[burden_col].values
        burden_values = np.maximum(burden_values, self.config.burden_floor)
        
        # Create log-normalized weights
        log_burdens = np.log10(burden_values)
        
        # Store normalizer for consistent scaling
        self.burden_normalizer = {
            'min': log_burdens.min(),
            'max': log_burdens.max(),
            'mean': log_burdens.mean(),
            'std': log_burdens.std(),
            'weights': log_burdens / log_burdens.sum(),
            'disease_map': dict(zip(burden_df['disease_name'], log_burdens))
        }
    
    def calculate_h_r(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        H-R: Representation Component
        Mathematical formulation using entropy and diversity indices
        """
        components = {}
        
        # 1. Ancestry Diversity (Shannon Entropy normalized)
        if 'ancestry_counts' in data and data['ancestry_counts']:
            counts = np.array(list(data['ancestry_counts'].values()))
            if counts.sum() > 0:
                # Shannon entropy for diversity
                probs = counts / counts.sum()
                H = -np.sum(probs * np.log(probs + 1e-10))
                H_max = np.log(len(counts))  # Maximum entropy
                diversity_score = H / H_max if H_max > 0 else 0
                
                # Apply Simpson's index for robustness
                simpson = 1 - np.sum(probs ** 2)
                
                # Combine both measures
                ancestry_score = 0.6 * diversity_score + 0.4 * simpson
            else:
                ancestry_score = 0.0
        else:
            ancestry_score = 0.0
        components['ancestry_diversity'] = ancestry_score
        
        # 2. Geographic Coverage (Burden-weighted)
        if 'geographic_coverage' in data:
            geo_score = self._calculate_geographic_equity(
                data['geographic_coverage'],
                data.get('population_by_region', {}),
                data.get('burden_by_region', {})
            )
        else:
            geo_score = 0.0
        components['geographic_coverage'] = geo_score
        
        # 3. Disease Coverage (Weighted by global burden)
        if 'disease_coverage' in data:
            disease_score = self._calculate_disease_coverage(
                data['disease_coverage']
            )
        else:
            disease_score = 0.0
        components['disease_coverage'] = disease_score
        
        # 4. Social Determinants (Comprehensiveness)
        if 'social_determinants' in data:
            # Key SDOH categories per WHO
            key_sdoh = ['education', 'income', 'employment', 'housing', 
                       'food_security', 'healthcare_access', 'social_support',
                       'discrimination', 'environment', 'transportation']
            
            covered = sum(1 for s in key_sdoh if s in data['social_determinants'])
            sdoh_score = covered / len(key_sdoh)
        else:
            sdoh_score = 0.0
        components['social_determinants'] = sdoh_score
        
        # 5. Sample Size Adequacy (Sigmoid function)
        if 'total_samples' in data and 'expected_samples' in data:
            ratio = data['total_samples'] / max(1, data['expected_samples'])
            # Sigmoid: good > 0.5, excellent > 1.0
            sample_score = 2 / (1 + np.exp(-2 * (ratio - 0.5))) - 1
            sample_score = max(0, min(1, sample_score))
        else:
            sample_score = 0.5
        components['sample_adequacy'] = sample_score
        
        # Aggregate with weighted geometric mean
        h_r = self._aggregate_component_scores(components)
        
        return h_r, components
    
    def calculate_h_s(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        H-S: Structure Component
        Governance, access, consent, participation
        """
        components = {}
        
        # 1. Governance Quality (weighted checklist)
        governance_rubric = {
            'community_board': 0.2,
            'indigenous_leadership': 0.15,
            'transparent_policies': 0.15,
            'benefit_sharing': 0.2,
            'independent_oversight': 0.15,
            'grievance_mechanism': 0.15
        }
        
        gov_score = sum(
            weight for item, weight in governance_rubric.items()
            if data.get('governance', {}).get(item, False)
        )
        components['governance'] = gov_score
        
        # 2. Access Model (Openness score)
        access_scores = {
            'open': 1.0,
            'registered': 0.85,
            'controlled': 0.65,
            'restricted': 0.4,
            'closed': 0.1
        }
        access_score = access_scores.get(
            data.get('access_model', 'restricted'), 0.4
        )
        components['access'] = access_score
        
        # 3. Consent Model (Participant agency)
        consent_scores = {
            'dynamic': 1.0,      # Participants can change consent
            'tiered': 0.85,      # Multiple consent options
            'broad': 0.7,        # General research consent
            'specific': 0.5,     # Single study consent
            'presumed': 0.2,     # Opt-out only
            'waived': 0.1        # No consent
        }
        consent_score = consent_scores.get(
            data.get('consent_model', 'specific'), 0.5
        )
        components['consent'] = consent_score
        
        # 4. Community Engagement (Depth and breadth)
        if 'community_engagement' in data:
            engagement = data['community_engagement']
            # Multiple dimensions of engagement
            depth_score = engagement.get('depth', 0) / 5  # 5-point scale
            breadth_score = engagement.get('breadth', 0) / 100  # % of communities
            frequency_score = min(1, engagement.get('meetings_per_year', 0) / 12)
            
            engagement_score = 0.4 * depth_score + 0.4 * breadth_score + 0.2 * frequency_score
        else:
            engagement_score = 0.0
        components['community_engagement'] = engagement_score
        
        # 5. Sustainability (Long-term viability)
        if 'sustainability' in data:
            sust = data['sustainability']
            funding_years = min(1, sust.get('funding_years', 0) / 10)  # 10+ years is good
            local_capacity = sust.get('local_capacity_building', False)
            
            sustainability_score = 0.7 * funding_years + 0.3 * float(local_capacity)
        else:
            sustainability_score = 0.5
        components['sustainability'] = sustainability_score
        
        # Aggregate
        h_s = self._aggregate_component_scores(components)
        
        return h_s, components
    
    def calculate_h_f(self, data: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        H-F: Function Component
        Performance equity, clinical impact, deployment
        """
        components = {}
        
        # 1. Performance Equity Across Populations
        if 'performance_by_population' in data:
            performances = list(data['performance_by_population'].values())
            if len(performances) > 1:
                # Use Gini coefficient for inequality (0=perfect equality)
                performances = np.array(performances)
                gini = self._calculate_gini_coefficient(performances)
                equity_score = 1 - gini  # Convert to equity measure
                
                # Also check minimum acceptable performance
                min_perf = performances.min()
                if min_perf < 0.7:  # Below clinical threshold
                    equity_score *= min_perf / 0.7  # Penalty
            else:
                equity_score = 0.5
        else:
            equity_score = 0.5
        components['performance_equity'] = equity_score
        
        # 2. Clinical Deployment Coverage
        if 'deployment_coverage' in data:
            deploy = data['deployment_coverage']
            
            # Weight by population served
            if isinstance(deploy, dict):
                total_pop = sum(deploy.values())
                high_burden_pop = sum(
                    pop for region, pop in deploy.items()
                    if region in data.get('high_burden_regions', [])
                )
                coverage_score = high_burden_pop / max(1, total_pop)
            else:
                coverage_score = float(deploy)
        else:
            coverage_score = 0.0
        components['deployment_coverage'] = coverage_score
        
        # 3. Health Outcome Impact (Burden-weighted)
        if 'outcome_improvements' in data:
            improvements = data['outcome_improvements']
            
            weighted_impact = 0
            total_weight = 0
            
            for disease, metrics in improvements.items():
                # Get disease burden weight
                if self.burden_normalizer and disease in self.burden_normalizer['disease_map']:
                    weight = self.burden_normalizer['disease_map'][disease]
                else:
                    weight = 1.0
                
                # Calculate impact (QALY gained, mortality reduced, etc.)
                if isinstance(metrics, dict):
                    impact = metrics.get('effect_size', 0) * metrics.get('reach', 0)
                else:
                    impact = float(metrics)
                
                weighted_impact += weight * impact
                total_weight += weight
            
            outcome_score = weighted_impact / max(1, total_weight)
            outcome_score = min(1, outcome_score)  # Cap at 1
        else:
            outcome_score = 0.0
        components['outcome_impact'] = outcome_score
        
        # 4. Bias Mitigation
        if 'bias_metrics' in data:
            biases = data['bias_metrics']
            
            # Multiple bias dimensions
            bias_types = ['demographic_parity', 'equalized_odds', 
                         'calibration', 'individual_fairness']
            
            bias_scores = []
            for bias_type in bias_types:
                if bias_type in biases:
                    # Convert bias to score (lower bias = higher score)
                    bias_val = abs(biases[bias_type])
                    score = np.exp(-bias_val * 2)  # Exponential penalty
                    bias_scores.append(score)
            
            if bias_scores:
                bias_score = np.mean(bias_scores)
            else:
                bias_score = 0.5
        else:
            bias_score = 0.5
        components['bias_mitigation'] = bias_score
        
        # 5. Validation Rigor
        if 'validation' in data:
            val = data['validation']
            
            # Check for multiple validation approaches
            external_val = float(val.get('external_validation', False))
            prospective_val = float(val.get('prospective_validation', False))
            multi_site = float(val.get('multi_site', False))
            
            validation_score = (external_val + prospective_val + multi_site) / 3
        else:
            validation_score = 0.0
        components['validation_rigor'] = validation_score
        
        # Aggregate
        h_f = self._aggregate_component_scores(components)
        
        return h_f, components
    
    def calculate_heim(self, data: Dict[str, Any], 
                      context: str = 'default',
                      return_details: bool = True) -> Dict[str, Any]:
        """
        Calculate complete HEIM score
        
        Mathematical formulation:
        HEIM = (H-R^w_r × H-S^w_s × H-F^w_f)^(1/Σw) × (1 + temporal_bonus)
        
        Using weighted geometric mean to ensure:
        - Zero in any component heavily penalizes total
        - All components must be adequate for high score
        """
        # Get context weights
        weights = self.config.weights.get(context, self.config.weights['default'])
        
        # Calculate components
        h_r, h_r_details = self.calculate_h_r(data)
        h_s, h_s_details = self.calculate_h_s(data)
        h_f, h_f_details = self.calculate_h_f(data)
        
        # Apply weighted geometric mean
        w_r, w_s, w_f = weights['H-R'], weights['H-S'], weights['H-F']
        w_sum = w_r + w_s + w_f
        
        # Handle zeros with small epsilon
        eps = 1e-10
        h_r_safe = max(eps, h_r)
        h_s_safe = max(eps, h_s)
        h_f_safe = max(eps, h_f)
        
        # Weighted geometric mean
        heim_base = (h_r_safe**w_r * h_s_safe**w_s * h_f_safe**w_f) ** (1/w_sum)
        
        # Apply sparsity penalty if applicable
        if 'sparsity_ratio' in data:
            sparsity_penalty = self._calculate_sparsity_penalty(data['sparsity_ratio'])
            heim_base *= sparsity_penalty
        
        # Calculate temporal bonus
        temporal_bonus = 0.0
        if 'temporal_improvements' in data:
            temporal_bonus = self._calculate_temporal_bonus(
                data['temporal_improvements']
            )
        
        # Final HEIM score
        heim_final = min(1.0, heim_base * (1 + temporal_bonus))
        
        # Determine interpretation
        interpretation = self._interpret_score(heim_final)
        
        result = {
            'heim_score': heim_final,
            'components': {
                'H-R': h_r,
                'H-S': h_s,
                'H-F': h_f
            },
            'weights': weights,
            'context': context,
            'interpretation': interpretation
        }
        
        if return_details:
            result['component_details'] = {
                'representation': h_r_details,
                'structure': h_s_details,
                'function': h_f_details
            }
            result['temporal_bonus'] = temporal_bonus
            result['calculation_method'] = 'weighted_geometric_mean'
        
        return result
    
    def _calculate_geographic_equity(self, coverage: Dict[str, int],
                                    population: Dict[str, int],
                                    burden: Dict[str, float]) -> float:
        """
        Calculate geographic equity with population and burden weighting
        """
        if not coverage:
            return 0.0
        
        # If we have burden data, use it for weighting
        if burden:
            weighted_coverage = 0.0
            total_burden = sum(burden.values())
            
            for region, samples in coverage.items():
                region_burden = burden.get(region, 0)
                region_pop = population.get(region, 1000000)  # Default 1M
                
                # Samples per capita per burden
                if region_burden > 0 and region_pop > 0:
                    burden_weight = region_burden / total_burden
                    per_capita_samples = samples / region_pop
                    
                    # Sigmoid normalization (good at 100 samples per million)
                    normalized = 1 / (1 + np.exp(-per_capita_samples * 1e6 / 100))
                    weighted_coverage += burden_weight * normalized
            
            return weighted_coverage
        else:
            # Fallback: simple diversity measure
            n_regions = len(coverage)
            total_samples = sum(coverage.values())
            
            # Shannon entropy of distribution
            if total_samples > 0:
                probs = np.array(list(coverage.values())) / total_samples
                H = -np.sum(probs * np.log(probs + 1e-10))
                H_max = np.log(n_regions)
                diversity = H / H_max if H_max > 0 else 0
                
                # Also reward absolute coverage
                coverage_score = min(1, n_regions / 50)  # 50+ regions is excellent
                
                return 0.6 * diversity + 0.4 * coverage_score
            return 0.0
    
    def _calculate_disease_coverage(self, coverage: Dict[str, bool]) -> float:
        """
        Calculate disease coverage weighted by global burden
        """
        if not self.burden_normalizer:
            # Simple proportion
            covered = sum(1 for v in coverage.values() if v)
            return covered / max(1, len(coverage))
        
        # Weighted by burden
        total_burden = 0.0
        covered_burden = 0.0
        
        for disease, is_covered in coverage.items():
            disease_burden = self.burden_normalizer['disease_map'].get(
                disease, self.burden_normalizer['mean']
            )
            total_burden += disease_burden
            if is_covered:
                covered_burden += disease_burden
        
        return covered_burden / max(1, total_burden)
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for inequality measurement
        0 = perfect equality, 1 = perfect inequality
        """
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def _aggregate_component_scores(self, components: Dict[str, float]) -> float:
        """
        Aggregate component scores using configured method
        """
        scores = list(components.values())
        scores = [s for s in scores if s is not None]
        
        if not scores:
            return 0.0
        
        if self.config.aggregation == 'weighted_geometric':
            # Geometric mean with small epsilon for zeros
            eps = 1e-10
            scores_safe = [max(eps, s) for s in scores]
            
            # Check for any true zeros
            n_zeros = sum(1 for s in scores if s < eps)
            
            # Geometric mean
            geom_mean = np.exp(np.mean(np.log(scores_safe)))
            
            # Apply penalty for zeros
            if n_zeros > 0:
                penalty = (len(scores) - n_zeros) / len(scores)
                geom_mean *= penalty ** 2  # Square for stronger penalty
            
            return geom_mean
            
        elif self.config.aggregation == 'arithmetic':
            return np.mean(scores)
        
        elif self.config.aggregation == 'harmonic':
            eps = 0.01
            scores_safe = [max(eps, s) for s in scores]
            return len(scores_safe) / np.sum(1.0 / np.array(scores_safe))
        
        else:
            return np.mean(scores)
    
    def _calculate_sparsity_penalty(self, sparsity_ratio: float) -> float:
        """
        Calculate penalty for sparse data
        Sigmoid function for smooth transition
        """
        if self.config.sparsity_penalty_type == 'sigmoid':
            # Sigmoid centered at threshold
            x = (sparsity_ratio - self.config.sparsity_threshold) * 10
            penalty = 1 / (1 + np.exp(-x))
            return 0.5 + 0.5 * penalty  # Scale to [0.5, 1]
        
        elif self.config.sparsity_penalty_type == 'linear':
            if sparsity_ratio >= self.config.sparsity_threshold:
                return 1.0
            else:
                return 0.5 + 0.5 * (sparsity_ratio / self.config.sparsity_threshold)
        
        else:
            return 1.0 if sparsity_ratio >= self.config.sparsity_threshold else 0.8
    
    def _calculate_temporal_bonus(self, improvements: List[Dict]) -> float:
        """
        Calculate temporal bonus for improving trends
        Recent improvements weighted more heavily
        """
        if not improvements:
            return 0.0
        
        total_bonus = 0.0
        current_year = 2024  # Or use datetime.now().year
        
        for improvement in improvements:
            year = improvement.get('year', current_year)
            magnitude = improvement.get('magnitude', 0)  # % improvement
            
            # Age decay
            years_ago = current_year - year
            decay_factor = self.config.temporal_decay ** years_ago
            
            # Bonus calculation (capped)
            bonus = min(0.1, magnitude * 0.01) * decay_factor
            total_bonus += bonus
        
        return min(self.config.temporal_bonus_max, total_bonus)
    
    def _interpret_score(self, score: float) -> Dict[str, str]:
        """
        Interpret HEIM score with actionable recommendations
        """
        if score >= 0.9:
            return {
                'level': 'Exceptional',
                'grade': 'A+',
                'description': 'Gold standard for health equity',
                'action': 'Share as best practice model'
            }
        elif score >= 0.8:
            return {
                'level': 'Excellent',
                'grade': 'A',
                'description': 'Strong equity performance',
                'action': 'Minor refinements in lowest components'
            }
        elif score >= 0.7:
            return {
                'level': 'Good',
                'grade': 'B',
                'description': 'Above threshold for ethical deployment',
                'action': 'Target specific gaps for improvement'
            }
        elif score >= 0.6:
            return {
                'level': 'Adequate',
                'grade': 'C',
                'description': 'Meets minimum standards',
                'action': 'Significant improvements needed'
            }
        elif score >= 0.4:
            return {
                'level': 'Poor',
                'grade': 'D',
                'description': 'Below ethical deployment threshold',
                'action': 'Major equity interventions required'
            }
        else:
            return {
                'level': 'Unacceptable',
                'grade': 'F',
                'description': 'Fundamental equity failures',
                'action': 'Complete redesign with equity-first approach'
            }
    
    def validate_score(self, heim_result: Dict, 
                      reference_scores: Optional[Dict] = None) -> Dict:
        """
        Validate HEIM score against benchmarks
        """
        validation = {
            'score': heim_result['heim_score'],
            'passes_threshold': heim_result['heim_score'] >= 0.7,
            'component_balance': None,
            'comparison': None
        }
        
        # Check component balance (no component should dominate)
        components = heim_result['components']
        cv = stats.variation(list(components.values()))
        validation['component_balance'] = {
            'coefficient_of_variation': cv,
            'balanced': cv < 0.3  # Less than 30% variation is good
        }
        
        # Compare to reference if provided
        if reference_scores:
            validation['comparison'] = {}
            for name, ref_score in reference_scores.items():
                diff = heim_result['heim_score'] - ref_score
                validation['comparison'][name] = {
                    'difference': diff,
                    'relative': (diff / ref_score) * 100 if ref_score > 0 else 0
                }
        
        return validation


# Example usage demonstrating the mathematical framework
if __name__ == "__main__":
    # Initialize calculator
    calculator = HEIMCalculator()
    
    # Example: UK Biobank-like data
    ukb_data = {
        'ancestry_counts': {
            'EUR': 450000,  # 90% European
            'AFR': 15000,    # 3% African
            'SAS': 20000,    # 4% South Asian
            'EAS': 10000,    # 2% East Asian
            'AMR': 5000      # 1% American
        },
        'geographic_coverage': {
            'UK': 480000,
            'Ireland': 15000,
            'Other': 5000
        },
        'disease_coverage': {
            'cardiovascular': True,
            'cancer': True,
            'diabetes': True,
            'mental_health': True,
            'rare_diseases': False  # Gap
        },
        'social_determinants': ['income', 'education', 'employment'],
        'governance': {
            'transparent_policies': True,
            'independent_oversight': True
        },
        'access_model': 'registered',
        'consent_model': 'broad',
        'performance_by_population': {
            'EUR': 0.92,
            'AFR': 0.78,  # Performance gap
            'SAS': 0.81,
            'EAS': 0.83
        },
        'temporal_improvements': [
            {'year': 2023, 'magnitude': 0.05},
            {'year': 2024, 'magnitude': 0.03}
        ]
    }
    
    # Calculate HEIM
    result = calculator.calculate_heim(ukb_data, context='genomics')
    
    print("UK Biobank HEIM Score:", result['heim_score'])
    print("Interpretation:", result['interpretation'])
    print("\nComponent Scores:")
    for comp, score in result['components'].items():
        print(f"  {comp}: {score:.3f}")
    
    # Validate
    validation = calculator.validate_score(result)
    print("\nValidation:")
    print(f"  Passes threshold (≥0.7): {validation['passes_threshold']}")
    print(f"  Component balance: {validation['component_balance']['balanced']}")