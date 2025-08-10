#!/usr/bin/env python3
"""
00-01-RETRIEVAL-VALIDATION.py

PUBMED RETRIEVAL OUTPUT ANALYZER & QUALITY VALIDATOR

Purpose: Comprehensive statistical analysis and validation of PubMed data 
retrieved by 00-00-00-pubmed-mesh-data-retrieval.py script.

This script analyzes the condition-specific CSV files to:
1. Validate data completeness and quality
2. Identify retrieval gaps and biases
3. Analyze temporal coverage and trends
4. Assess abstract/MeSH term quality
5. Generate confidence metrics for each condition
6. Identify anomalies requiring re-retrieval

Directory Structure:
- Scripts: PYTHON/
- Input: CONDITION_DATA/condition_progress/[condition]/[condition]_[year].csv
- Output: ANALYSIS/00-01-RETRIEVAL-VALIDATION/

Usage:
    python3 PYTHON/00-01-RETRIEVAL-VALIDATION.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import json
import re
from datetime import datetime
import warnings
from scipy import stats
from pathlib import Path
import glob
from typing import Dict, List, Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("📝 Install tqdm for progress bars: pip install tqdm")

warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class PubMedRetrievalValidator:
    def __init__(self, min_year=1975, max_year=2024):
        """Initialize the PubMed Retrieval Validator following user's directory structure"""
        
        # Follow user's directory structure
        self.condition_data_dir = './CONDITION_DATA/condition_progress'
        self.analysis_dir = './ANALYSIS'
        self.output_dir = os.path.join(self.analysis_dir, '00-01-RETRIEVAL-VALIDATION')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Time parameters
        self.min_year = min_year
        self.max_year = max_year
        self.year_range = max_year - min_year + 1
        
        # Storage
        self.validation_results = {}
        self.condition_stats = {}
        self.quality_scores = {}
        self.anomalies = defaultdict(list)
        self.retrieval_gaps = defaultdict(list)
        
        print(f"🔍 PUBMED RETRIEVAL OUTPUT VALIDATOR (00-01)")
        print(f"=" * 70)
        print(f"📁 Input: CONDITION_DATA/condition_progress/")
        print(f"📂 Output: ANALYSIS/00-01-RETRIEVAL-VALIDATION/")
        print(f"📅 Expected year range: {self.min_year}-{self.max_year} ({self.year_range} years)")
        print()
        
        # Initial scan
        self.available_conditions = []
        self.scan_retrieval_outputs()
    
    def scan_retrieval_outputs(self):
        """Scan and inventory all retrieved data"""
        print("📋 SCANNING RETRIEVAL OUTPUTS...")
        
        if not os.path.exists(self.condition_data_dir):
            print(f"❌ Directory not found: {self.condition_data_dir}")
            print(f"   Please run 00-00-00-pubmed-mesh-data-retrieval.py first")
            return
        
        # Scan all condition folders
        condition_summary = []
        
        for condition_folder in sorted(os.listdir(self.condition_data_dir)):
            condition_path = os.path.join(self.condition_data_dir, condition_folder)
            
            if os.path.isdir(condition_path):
                csv_files = glob.glob(os.path.join(condition_path, "*.csv"))
                
                if csv_files:
                    # Extract years from filenames
                    years_found = []
                    total_size_mb = 0
                    
                    for csv_file in csv_files:
                        # Get file size
                        size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                        total_size_mb += size_mb
                        
                        # Extract year from filename
                        year_match = re.search(r'_(\d{4})\.csv$', csv_file)
                        if year_match:
                            years_found.append(int(year_match.group(1)))
                    
                    self.available_conditions.append(condition_folder)
                    
                    condition_summary.append({
                        'condition': condition_folder,
                        'num_files': len(csv_files),
                        'years_found': sorted(years_found),
                        'year_coverage': len(years_found) / self.year_range * 100,
                        'total_size_mb': total_size_mb,
                        'min_year': min(years_found) if years_found else None,
                        'max_year': max(years_found) if years_found else None
                    })
        
        # Store summary
        self.retrieval_summary = pd.DataFrame(condition_summary)
        
        if not self.retrieval_summary.empty:
            print(f"✅ Found {len(self.available_conditions)} conditions")
            print(f"   Total files: {self.retrieval_summary['num_files'].sum():,}")
            print(f"   Total size: {self.retrieval_summary['total_size_mb'].sum():.1f} MB")
            print(f"   Average year coverage: {self.retrieval_summary['year_coverage'].mean():.1f}%")
            
            # Show conditions with best/worst coverage
            best = self.retrieval_summary.nlargest(5, 'year_coverage')
            worst = self.retrieval_summary.nsmallest(5, 'year_coverage')
            
            print(f"\n📊 BEST COVERAGE:")
            for _, row in best.iterrows():
                clean_name = row['condition'].replace('_', ' ')
                print(f"   {clean_name}: {row['year_coverage']:.1f}% ({row['num_files']} files)")
            
            print(f"\n⚠️  WORST COVERAGE:")
            for _, row in worst.iterrows():
                clean_name = row['condition'].replace('_', ' ')
                print(f"   {clean_name}: {row['year_coverage']:.1f}% ({row['num_files']} files)")
        else:
            print("❌ No data found!")
    
    def analyze_condition_completeness(self, condition: str) -> Dict:
        """Analyze completeness and quality for a specific condition"""
        
        condition_path = os.path.join(self.condition_data_dir, condition)
        if not os.path.exists(condition_path):
            return {}
        
        stats = {
            'condition': condition,
            'years': {},
            'total_papers': 0,
            'papers_with_abstracts': 0,
            'papers_with_mesh': 0,
            'missing_years': [],
            'anomalous_years': [],
            'quality_issues': [],
            'duplicate_pmids': 0,
            'unique_pmids': set()
        }
        
        # Expected years
        expected_years = set(range(self.min_year, self.max_year + 1))
        found_years = set()
        
        # Analyze each year file
        for year in range(self.min_year, self.max_year + 1):
            file_path = os.path.join(condition_path, f"{condition}_{year}.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    found_years.add(year)
                    
                    # Basic statistics
                    n_papers = len(df)
                    n_abstracts = (~df['Abstract'].isna()).sum() if 'Abstract' in df.columns else 0
                    n_mesh = (~df['MeSH_Terms'].isna()).sum() if 'MeSH_Terms' in df.columns else 0
                    
                    # Quality checks
                    abstract_rate = n_abstracts / n_papers * 100 if n_papers > 0 else 0
                    mesh_rate = n_mesh / n_papers * 100 if n_papers > 0 else 0
                    
                    # Check for duplicates within year
                    n_duplicates = 0
                    if 'PMID' in df.columns:
                        n_duplicates = df.duplicated(subset=['PMID']).sum()
                        # Track unique PMIDs across all years
                        stats['unique_pmids'].update(df['PMID'].dropna().astype(str))
                    
                    # Check for meaningful abstracts (not just "[No abstract available]" etc.)
                    meaningful_abstracts = 0
                    if 'Abstract' in df.columns:
                        abstract_series = df['Abstract'].dropna()
                        meaningful_abstracts = sum(1 for a in abstract_series 
                                                  if len(str(a)) > 100 and 
                                                  'no abstract' not in str(a).lower())
                        avg_abstract_len = abstract_series.str.len().mean() if len(abstract_series) > 0 else 0
                    else:
                        avg_abstract_len = 0
                    
                    year_stats = {
                        'papers': n_papers,
                        'abstracts': n_abstracts,
                        'meaningful_abstracts': meaningful_abstracts,
                        'abstract_rate': abstract_rate,
                        'mesh_terms': n_mesh,
                        'mesh_rate': mesh_rate,
                        'duplicates': n_duplicates,
                        'avg_abstract_length': avg_abstract_len
                    }
                    
                    stats['years'][year] = year_stats
                    stats['total_papers'] += n_papers
                    stats['papers_with_abstracts'] += meaningful_abstracts
                    stats['papers_with_mesh'] += n_mesh
                    stats['duplicate_pmids'] += n_duplicates
                    
                    # Flag quality issues
                    if n_papers == 0:
                        stats['quality_issues'].append(f"Year {year}: Empty file")
                    elif abstract_rate < 50:
                        stats['quality_issues'].append(f"Year {year}: Low abstract rate ({abstract_rate:.1f}%)")
                    elif n_duplicates > n_papers * 0.05:  # More than 5% duplicates
                        stats['quality_issues'].append(f"Year {year}: High duplicates ({n_duplicates}/{n_papers})")
                    
                except Exception as e:
                    stats['quality_issues'].append(f"Year {year}: Error reading file - {str(e)}")
            else:
                stats['missing_years'].append(year)
        
        # Calculate missing years
        stats['missing_years'] = sorted(list(expected_years - found_years))
        stats['year_coverage'] = len(found_years) / len(expected_years) * 100
        stats['unique_pmid_count'] = len(stats['unique_pmids'])
        
        # Check for cross-year duplicates
        if stats['total_papers'] > 0 and stats['unique_pmid_count'] > 0:
            stats['cross_year_duplicate_rate'] = (stats['total_papers'] - stats['unique_pmid_count']) / stats['total_papers'] * 100
        else:
            stats['cross_year_duplicate_rate'] = 0
        
        # Detect anomalous years (statistical outliers)
        if len(stats['years']) > 5:
            paper_counts = [y['papers'] for y in stats['years'].values() if y['papers'] > 0]
            if paper_counts:
                q1 = np.percentile(paper_counts, 25)
                q3 = np.percentile(paper_counts, 75)
                iqr = q3 - q1
                lower_bound = max(0, q1 - 1.5 * iqr)
                upper_bound = q3 + 1.5 * iqr
                
                for year, year_stats in stats['years'].items():
                    if year_stats['papers'] > 0 and (year_stats['papers'] < lower_bound or year_stats['papers'] > upper_bound):
                        stats['anomalous_years'].append({
                            'year': year,
                            'papers': year_stats['papers'],
                            'reason': f'Outlier (expected {lower_bound:.0f}-{upper_bound:.0f})'
                        })
        
        # Calculate overall quality score
        quality_score = self.calculate_quality_score(stats)
        stats['quality_score'] = quality_score
        
        return stats
    
    def calculate_quality_score(self, stats: Dict) -> float:
        """Calculate a quality score for the condition data"""
        
        score = 100.0
        
        # Penalize for missing years (max -30 points)
        missing_penalty = min(30, len(stats['missing_years']) * 0.6)
        score -= missing_penalty
        
        # Penalize for low abstract coverage (max -25 points)
        if stats['total_papers'] > 0:
            abstract_rate = stats['papers_with_abstracts'] / stats['total_papers'] * 100
            if abstract_rate < 70:
                score -= (70 - abstract_rate) * 0.35
        
        # Penalize for low MeSH coverage (max -20 points)
        if stats['total_papers'] > 0:
            mesh_rate = stats['papers_with_mesh'] / stats['total_papers'] * 100
            if mesh_rate < 70:
                score -= (70 - mesh_rate) * 0.28
        
        # Penalize for cross-year duplicates (max -10 points)
        if stats.get('cross_year_duplicate_rate', 0) > 5:
            score -= min(10, stats['cross_year_duplicate_rate'] * 0.5)
        
        # Penalize for quality issues (max -10 points)
        issue_penalty = min(10, len(stats['quality_issues']) * 2)
        score -= issue_penalty
        
        # Penalize for anomalous years (max -5 points)
        anomaly_penalty = min(5, len(stats['anomalous_years']) * 1)
        score -= anomaly_penalty
        
        return max(0, score)
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns across all conditions"""
        print(f"\n📈 ANALYZING TEMPORAL PATTERNS...")
        
        temporal_data = defaultdict(lambda: defaultdict(int))
        
        for condition in tqdm(self.available_conditions, desc="Processing conditions"):
            condition_path = os.path.join(self.condition_data_dir, condition)
            
            for year in range(self.min_year, self.max_year + 1):
                file_path = os.path.join(condition_path, f"{condition}_{year}.csv")
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, nrows=1)  # Just get count
                        # Read again to get actual count
                        with open(file_path, 'r') as f:
                            line_count = sum(1 for line in f) - 1  # Subtract header
                        temporal_data[year][condition] = max(0, line_count)
                    except:
                        temporal_data[year][condition] = 0
        
        # Convert to DataFrame
        self.temporal_df = pd.DataFrame(temporal_data).T
        self.temporal_df = self.temporal_df.fillna(0).astype(int)
        
        # Calculate aggregate statistics
        self.temporal_stats = {
            'total_by_year': self.temporal_df.sum(axis=1).to_dict(),
            'mean_by_year': self.temporal_df.mean(axis=1).to_dict(),
            'std_by_year': self.temporal_df.std(axis=1).to_dict(),
            'active_conditions_by_year': (self.temporal_df > 0).sum(axis=1).to_dict()
        }
        
        # Identify trends
        years = sorted(self.temporal_stats['total_by_year'].keys())
        totals = [self.temporal_stats['total_by_year'][y] for y in years]
        
        if len(years) > 1:
            # Calculate growth rate
            if len(totals) >= 10:
                early_period = totals[:5]
                late_period = totals[-5:]
            else:
                early_period = totals[:len(totals)//2]
                late_period = totals[len(totals)//2:]
            
            early_avg = np.mean(early_period) if early_period else 0
            late_avg = np.mean(late_period) if late_period else 0
            growth_rate = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
            
            # Fit linear trend
            from scipy import stats as scipy_stats
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, totals)
            
            self.temporal_stats['growth_rate'] = growth_rate
            self.temporal_stats['trend_slope'] = slope
            self.temporal_stats['trend_r2'] = r_value ** 2
            self.temporal_stats['trend_p_value'] = p_value
            
            print(f"   Total papers across all conditions: {sum(totals):,}")
            print(f"   Average annual growth: {growth_rate:.1f}%")
            print(f"   Trend R²: {r_value**2:.3f}")
            print(f"   Trend significant: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.4f})")
    
    def identify_data_gaps(self):
        """Identify systematic gaps in the data"""
        print(f"\n🔍 IDENTIFYING DATA GAPS...")
        
        gaps = {
            'missing_years': defaultdict(list),
            'low_coverage_years': defaultdict(list),
            'suspicious_patterns': [],
            'incomplete_conditions': [],
            'empty_files': []
        }
        
        for condition in self.available_conditions:
            stats = self.condition_stats.get(condition, {})
            
            # Missing years
            if stats.get('missing_years'):
                gaps['missing_years'][condition] = stats['missing_years']
            
            # Low coverage years
            for year, year_stats in stats.get('years', {}).items():
                if year_stats['papers'] == 0:
                    gaps['empty_files'].append({'condition': condition, 'year': year})
                elif year_stats['abstract_rate'] < 50:
                    gaps['low_coverage_years'][condition].append({
                        'year': year,
                        'abstract_rate': year_stats['abstract_rate'],
                        'papers': year_stats['papers']
                    })
            
            # Incomplete conditions
            if stats.get('year_coverage', 100) < 80:
                gaps['incomplete_conditions'].append({
                    'condition': condition,
                    'coverage': stats.get('year_coverage', 0),
                    'missing_years': len(stats.get('missing_years', [])),
                    'total_papers': stats.get('total_papers', 0)
                })
        
        # Check for systematic year gaps across conditions
        year_gap_counts = defaultdict(int)
        for condition, missing_years in gaps['missing_years'].items():
            for year in missing_years:
                year_gap_counts[year] += 1
        
        # Years missing for many conditions (>20% of conditions)
        systematic_gaps = {year: count for year, count in year_gap_counts.items() 
                          if count > len(self.available_conditions) * 0.2}
        
        if systematic_gaps:
            gaps['suspicious_patterns'].append({
                'type': 'Systematic year gaps',
                'details': systematic_gaps,
                'description': 'Years with missing data for >20% of conditions'
            })
        
        self.data_gaps = gaps
        
        # Report findings
        print(f"   Conditions with missing years: {len(gaps['missing_years'])}")
        print(f"   Incomplete conditions (<80% coverage): {len(gaps['incomplete_conditions'])}")
        print(f"   Empty files found: {len(gaps['empty_files'])}")
        
        if systematic_gaps:
            print(f"\n   ⚠️  SYSTEMATIC GAPS DETECTED:")
            for year, count in sorted(systematic_gaps.items(), key=lambda x: x[1], reverse=True)[:5]:
                pct = count/len(self.available_conditions)*100
                print(f"      Year {year}: Missing for {count} conditions ({pct:.1f}%)")
    
    def analyze_mesh_quality(self):
        """Analyze MeSH term quality and coverage"""
        print(f"\n🏷️  ANALYZING MESH TERM QUALITY...")
        
        mesh_stats = {
            'coverage_by_condition': {},
            'unique_terms_total': set(),
            'term_frequency': Counter(),
            'terms_per_paper': [],
            'top_terms_by_condition': {}
        }
        
        # Sample conditions for detailed analysis
        sample_size = min(30, len(self.available_conditions))
        sample_conditions = np.random.choice(self.available_conditions, sample_size, replace=False)
        
        for condition in tqdm(sample_conditions, desc="Analyzing MeSH"):
            condition_path = os.path.join(self.condition_data_dir, condition)
            condition_mesh_terms = set()
            condition_term_freq = Counter()
            papers_with_mesh = 0
            total_papers = 0
            
            # Sample recent years for better quality assessment
            sample_years = [2015, 2018, 2020, 2022, 2024]
            
            for year in sample_years:
                file_path = os.path.join(condition_path, f"{condition}_{year}.csv")
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        total_papers += len(df)
                        
                        if 'MeSH_Terms' in df.columns:
                            mesh_data = df['MeSH_Terms'].dropna()
                            papers_with_mesh += len(mesh_data)
                            
                            for mesh_str in mesh_data:
                                terms = str(mesh_str).split(';')
                                terms = [t.strip() for t in terms if t.strip()]
                                condition_mesh_terms.update(terms)
                                condition_term_freq.update(terms)
                                mesh_stats['term_frequency'].update(terms)
                                mesh_stats['terms_per_paper'].append(len(terms))
                    except:
                        pass
            
            if total_papers > 0:
                mesh_stats['coverage_by_condition'][condition] = {
                    'coverage': papers_with_mesh / total_papers * 100,
                    'unique_terms': len(condition_mesh_terms),
                    'avg_terms_per_paper': np.mean([len(t) for t in condition_term_freq.elements()]) if condition_term_freq else 0
                }
                mesh_stats['unique_terms_total'].update(condition_mesh_terms)
                mesh_stats['top_terms_by_condition'][condition] = condition_term_freq.most_common(10)
        
        # Calculate overall statistics
        if mesh_stats['coverage_by_condition']:
            coverages = [v['coverage'] for v in mesh_stats['coverage_by_condition'].values()]
            mesh_stats['mean_coverage'] = np.mean(coverages)
            mesh_stats['std_coverage'] = np.std(coverages)
            mesh_stats['total_unique_terms'] = len(mesh_stats['unique_terms_total'])
            
            if mesh_stats['terms_per_paper']:
                mesh_stats['mean_terms_per_paper'] = np.mean(mesh_stats['terms_per_paper'])
                mesh_stats['median_terms_per_paper'] = np.median(mesh_stats['terms_per_paper'])
            
            print(f"   Mean MeSH coverage: {mesh_stats['mean_coverage']:.1f}% (±{mesh_stats['std_coverage']:.1f}%)")
            print(f"   Total unique MeSH terms: {mesh_stats['total_unique_terms']:,}")
            print(f"   Mean terms per paper: {mesh_stats.get('mean_terms_per_paper', 0):.1f}")
            print(f"   Median terms per paper: {mesh_stats.get('median_terms_per_paper', 0):.1f}")
            
            # Top MeSH terms overall
            print(f"\n   TOP MESH TERMS OVERALL:")
            for term, count in mesh_stats['term_frequency'].most_common(10):
                print(f"      {term}: {count:,}")
        
        self.mesh_stats = mesh_stats
    
    def generate_confidence_scores(self):
        """Generate confidence scores for each condition"""
        print(f"\n🎯 GENERATING CONFIDENCE SCORES...")
        
        for condition in tqdm(self.available_conditions, desc="Scoring conditions"):
            stats = self.analyze_condition_completeness(condition)
            self.condition_stats[condition] = stats
            self.quality_scores[condition] = stats['quality_score']
        
        # Categorize by confidence level
        high_confidence = [c for c, s in self.quality_scores.items() if s >= 80]
        medium_confidence = [c for c, s in self.quality_scores.items() if 60 <= s < 80]
        low_confidence = [c for c, s in self.quality_scores.items() if s < 60]
        
        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        print(f"   High (≥80): {len(high_confidence)} conditions")
        print(f"   Medium (60-79): {len(medium_confidence)} conditions")
        print(f"   Low (<60): {len(low_confidence)} conditions")
        
        # Show examples from each category
        if high_confidence:
            print(f"\n✅ HIGH CONFIDENCE CONDITIONS (examples):")
            for condition in sorted(high_confidence, key=lambda x: self.quality_scores[x], reverse=True)[:5]:
                clean_name = condition.replace('_', ' ')
                print(f"   {clean_name}: {self.quality_scores[condition]:.1f}")
        
        if low_confidence:
            print(f"\n⚠️  LOW CONFIDENCE CONDITIONS (need attention):")
            for condition in sorted(low_confidence, key=lambda x: self.quality_scores[x])[:10]:
                clean_name = condition.replace('_', ' ')
                score = self.quality_scores[condition]
                stats = self.condition_stats[condition]
                print(f"   {clean_name}: {score:.1f}")
                if stats['missing_years']:
                    print(f"      Missing {len(stats['missing_years'])} years")
                if stats.get('cross_year_duplicate_rate', 0) > 5:
                    print(f"      Cross-year duplicates: {stats['cross_year_duplicate_rate']:.1f}%")
    
    def create_validation_visualizations(self):
        """Create comprehensive validation visualizations"""
        print(f"\n📊 CREATING VALIDATION VISUALIZATIONS...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Year coverage heatmap
        ax1 = plt.subplot(3, 3, 1)
        
        # Create coverage matrix for top conditions
        coverage_matrix = []
        condition_names = []
        
        # Sort conditions by total papers for better visualization
        sorted_conditions = sorted(self.available_conditions, 
                                 key=lambda x: self.condition_stats[x].get('total_papers', 0), 
                                 reverse=True)[:30]
        
        for condition in sorted_conditions:
            clean_name = condition.replace('_', ' ')
            if len(clean_name) > 20:
                clean_name = clean_name[:17] + '...'
            condition_names.append(clean_name)
            
            row = []
            stats = self.condition_stats.get(condition, {})
            
            for year in range(self.min_year, self.max_year + 1):
                if year in stats.get('years', {}):
                    papers = stats['years'][year]['papers']
                    if papers > 0:
                        row.append(1)  # Has data
                    else:
                        row.append(0.5)  # Empty file
                else:
                    row.append(0)  # Missing
            
            coverage_matrix.append(row)
        
        if coverage_matrix:
            # Show every 5th year for readability
            year_labels = [str(y) if y % 5 == 0 else '' for y in range(self.min_year, self.max_year + 1)]
            
            sns.heatmap(coverage_matrix, 
                       xticklabels=year_labels,
                       yticklabels=condition_names,
                       cmap='RdYlGn', vmin=0, vmax=1,
                       cbar_kws={'label': 'Data Status'},
                       ax=ax1)
            ax1.set_title('Data Coverage by Condition and Year (Top 30)')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Condition')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Quality score distribution
        ax2 = plt.subplot(3, 3, 2)
        scores = list(self.quality_scores.values())
        
        ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(x=80, color='green', linestyle='--', linewidth=2, label='High confidence')
        ax2.axvline(x=60, color='orange', linestyle='--', linewidth=2, label='Medium confidence')
        ax2.axvline(x=np.mean(scores), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Number of Conditions')
        ax2.set_title('Distribution of Data Quality Scores')
        ax2.legend()
        
        # 3. Temporal trends
        ax3 = plt.subplot(3, 3, 3)
        if hasattr(self, 'temporal_stats'):
            years = sorted(self.temporal_stats['total_by_year'].keys())
            totals = [self.temporal_stats['total_by_year'][y] for y in years]
            
            ax3.plot(years, totals, marker='o', linewidth=2, markersize=4, color='darkblue')
            ax3.fill_between(years, totals, alpha=0.3, color='lightblue')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Total Papers')
            ax3.set_title(f'Total Papers Retrieved by Year (Growth: {self.temporal_stats.get("growth_rate", 0):.1f}%)')
            ax3.grid(True, alpha=0.3)
            
            # Format y-axis
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else str(int(x))))
        
        # 4. Abstract coverage distribution
        ax4 = plt.subplot(3, 3, 4)
        abstract_rates = []
        for condition in self.available_conditions:
            stats = self.condition_stats.get(condition, {})
            if stats.get('total_papers', 0) > 100:  # Only include conditions with sufficient data
                rate = stats['papers_with_abstracts'] / stats['total_papers'] * 100
                abstract_rates.append(rate)
        
        if abstract_rates:
            ax4.hist(abstract_rates, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax4.axvline(x=np.mean(abstract_rates), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(abstract_rates):.1f}%')
            ax4.axvline(x=70, color='green', linestyle=':', linewidth=2, label='Target: 70%')
            ax4.set_xlabel('Abstract Coverage (%)')
            ax4.set_ylabel('Number of Conditions')
            ax4.set_title('Abstract Availability by Condition')
            ax4.legend()
        
        # 5. MeSH coverage distribution
        ax5 = plt.subplot(3, 3, 5)
        mesh_rates = []
        for condition in self.available_conditions:
            stats = self.condition_stats.get(condition, {})
            if stats.get('total_papers', 0) > 100:  # Only include conditions with sufficient data
                rate = stats['papers_with_mesh'] / stats['total_papers'] * 100
                mesh_rates.append(rate)
        
        if mesh_rates:
            ax5.hist(mesh_rates, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
            ax5.axvline(x=np.mean(mesh_rates), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(mesh_rates):.1f}%')
            ax5.axvline(x=70, color='darkgreen', linestyle=':', linewidth=2, label='Target: 70%')
            ax5.set_xlabel('MeSH Coverage (%)')
            ax5.set_ylabel('Number of Conditions')
            ax5.set_title('MeSH Term Availability by Condition')
            ax5.legend()
        
        # 6. Missing years analysis
        ax6 = plt.subplot(3, 3, 6)
        missing_counts = [len(stats.get('missing_years', [])) 
                         for stats in self.condition_stats.values()]
        
        if missing_counts:
            unique_counts = sorted(set(missing_counts))
            count_freq = [missing_counts.count(x) for x in unique_counts]
            
            colors = ['green' if x == 0 else 'orange' if x <= 10 else 'red' for x in unique_counts]
            ax6.bar(unique_counts, count_freq, color=colors, edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Number of Missing Years')
            ax6.set_ylabel('Number of Conditions')
            ax6.set_title('Distribution of Missing Years per Condition')
            ax6.set_xticks(unique_counts[::2] if len(unique_counts) > 10 else unique_counts)
        
        # 7. Top conditions by paper count
        ax7 = plt.subplot(3, 3, 7)
        condition_papers = [(c, self.condition_stats[c].get('total_papers', 0)) 
                          for c in self.available_conditions]
        condition_papers.sort(key=lambda x: x[1], reverse=True)
        
        top_conditions = [c[0].replace('_', ' ')[:20] for c in condition_papers[:15]]
        top_counts = [c[1] for c in condition_papers[:15]]
        
        y_pos = np.arange(len(top_conditions))
        ax7.barh(y_pos, top_counts, alpha=0.7, color='purple')
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(top_conditions, fontsize=9)
        ax7.set_xlabel('Total Papers')
        ax7.set_title('Top 15 Conditions by Paper Count')
        
        # Format x-axis
        ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else str(int(x))))
        
        # 8. Data quality issues
        ax8 = plt.subplot(3, 3, 8)
        
        # Count different types of issues
        issue_types = {
            'Missing Years': sum(1 for s in self.condition_stats.values() if s.get('missing_years')),
            'Low Abstracts': sum(1 for s in self.condition_stats.values() 
                               if s.get('total_papers', 0) > 0 and 
                               s.get('papers_with_abstracts', 0) / s.get('total_papers', 1) < 0.5),
            'Low MeSH': sum(1 for s in self.condition_stats.values() 
                           if s.get('total_papers', 0) > 0 and 
                           s.get('papers_with_mesh', 0) / s.get('total_papers', 1) < 0.5),
            'High Duplicates': sum(1 for s in self.condition_stats.values() 
                                 if s.get('cross_year_duplicate_rate', 0) > 5),
            'Anomalies': sum(1 for s in self.condition_stats.values() if s.get('anomalous_years'))
        }
        
        if any(issue_types.values()):
            labels = list(issue_types.keys())
            sizes = list(issue_types.values())
            colors = ['red', 'orange', 'yellow', 'pink', 'brown']
            
            # Create pie chart
            wedges, texts, autotexts = ax8.pie(sizes, labels=labels, colors=colors, 
                                               autopct=lambda pct: f'{pct:.1f}%\n({int(pct * sum(sizes) / 100)})',
                                               startangle=90)
            ax8.set_title('Data Quality Issues Distribution')
            
            # Make percentage text smaller
            for autotext in autotexts:
                autotext.set_fontsize(9)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate key metrics
        total_papers = sum(s.get('total_papers', 0) for s in self.condition_stats.values())
        total_unique_pmids = sum(s.get('unique_pmid_count', 0) for s in self.condition_stats.values())
        
        summary_text = "📊 VALIDATION SUMMARY\n" + "="*30 + "\n\n"
        summary_text += f"Total Conditions: {len(self.available_conditions)}\n"
        summary_text += f"Total Papers: {total_papers:,}\n"
        summary_text += f"Unique PMIDs: {total_unique_pmids:,}\n"
        summary_text += f"Year Range: {self.min_year}-{self.max_year}\n\n"
        
        summary_text += "📈 QUALITY METRICS:\n"
        if scores:
            summary_text += f"Mean Quality Score: {np.mean(scores):.1f}\n"
            summary_text += f"Median Quality Score: {np.median(scores):.1f}\n"
        if abstract_rates:
            summary_text += f"Mean Abstract Coverage: {np.mean(abstract_rates):.1f}%\n"
        if mesh_rates:
            summary_text += f"Mean MeSH Coverage: {np.mean(mesh_rates):.1f}%\n"
        
        summary_text += f"\n🔍 DATA COMPLETENESS:\n"
        complete = len([c for c in self.available_conditions 
                       if not self.condition_stats[c].get('missing_years')])
        summary_text += f"Complete coverage: {complete}/{len(self.available_conditions)}\n"
        summary_text += f"Conditions with gaps: {len(self.available_conditions) - complete}\n"
        
        if hasattr(self, 'temporal_stats'):
            summary_text += f"\n📊 TEMPORAL TRENDS:\n"
            summary_text += f"Growth Rate: {self.temporal_stats.get('growth_rate', 0):.1f}%\n"
            summary_text += f"Trend R²: {self.temporal_stats.get('trend_r2', 0):.3f}\n"
            
            if self.temporal_stats.get('trend_p_value', 1) < 0.05:
                summary_text += f"Trend: Significant (p<0.05)\n"
            else:
                summary_text += f"Trend: Not significant\n"
        
        # Determine overall assessment
        mean_score = np.mean(scores) if scores else 0
        if mean_score >= 80:
            assessment = "✅ HIGH CONFIDENCE\nData ready for analysis"
            box_color = 'lightgreen'
        elif mean_score >= 60:
            assessment = "⚠️ MEDIUM CONFIDENCE\nConsider re-retrieval for low-quality conditions"
            box_color = 'lightyellow'
        else:
            assessment = "❌ LOW CONFIDENCE\nSignificant data issues detected"
            box_color = 'lightcoral'
        
        summary_text += f"\n{assessment}"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
        
        plt.suptitle('PubMed Retrieval Validation Dashboard (00-01)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(self.output_dir, '00-01-validation-dashboard.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved: {output_file}")
        
        plt.show()
    
    def export_validation_report(self):
        """Export comprehensive validation report following naming convention"""
        print(f"\n💾 EXPORTING VALIDATION REPORTS...")
        
        # 1. Export condition quality scores
        condition_data = []
        for condition in self.available_conditions:
            stats = self.condition_stats[condition]
            clean_name = condition.replace('_', ' ')
            
            condition_data.append({
                'Condition': clean_name,
                'Condition_Folder': condition,
                'Quality_Score': round(stats['quality_score'], 1),
                'Total_Papers': stats['total_papers'],
                'Unique_PMIDs': stats.get('unique_pmid_count', 0),
                'Papers_With_Abstracts': stats['papers_with_abstracts'],
                'Abstract_Coverage_%': round(stats['papers_with_abstracts'] / stats['total_papers'] * 100, 1) if stats['total_papers'] > 0 else 0,
                'Papers_With_MeSH': stats['papers_with_mesh'],
                'MeSH_Coverage_%': round(stats['papers_with_mesh'] / stats['total_papers'] * 100, 1) if stats['total_papers'] > 0 else 0,
                'Year_Coverage_%': round(stats['year_coverage'], 1),
                'Missing_Years': len(stats['missing_years']),
                'Cross_Year_Duplicates_%': round(stats.get('cross_year_duplicate_rate', 0), 1),
                'Quality_Issues': len(stats['quality_issues']),
                'Anomalous_Years': len(stats['anomalous_years'])
            })
        
        condition_df = pd.DataFrame(condition_data)
        condition_df = condition_df.sort_values('Quality_Score', ascending=False)
        condition_file = os.path.join(self.output_dir, '00-01-condition-quality-scores.csv')
        condition_df.to_csv(condition_file, index=False)
        print(f"✅ Condition scores: 00-01-condition-quality-scores.csv")
        
        # 2. Export temporal data
        if hasattr(self, 'temporal_df'):
            temporal_file = os.path.join(self.output_dir, '00-01-temporal-paper-counts.csv')
            # Add totals row
            temporal_with_totals = self.temporal_df.copy()
            temporal_with_totals.loc['TOTAL'] = temporal_with_totals.sum()
            temporal_with_totals.to_csv(temporal_file)
            print(f"✅ Temporal data: 00-01-temporal-paper-counts.csv")
        
        # 3. Export data gaps analysis
        gaps_data = []
        for condition, missing_years in self.data_gaps['missing_years'].items():
            clean_name = condition.replace('_', ' ')
            for year in missing_years:
                gaps_data.append({
                    'Condition': clean_name,
                    'Condition_Folder': condition,
                    'Missing_Year': year,
                    'Type': 'Missing Data'
                })
        
        # Add empty files
        for item in self.data_gaps.get('empty_files', []):
            clean_name = item['condition'].replace('_', ' ')
            gaps_data.append({
                'Condition': clean_name,
                'Condition_Folder': item['condition'],
                'Missing_Year': item['year'],
                'Type': 'Empty File'
            })
        
        if gaps_data:
            gaps_df = pd.DataFrame(gaps_data)
            gaps_df = gaps_df.sort_values(['Condition', 'Missing_Year'])
            gaps_file = os.path.join(self.output_dir, '00-01-data-gaps.csv')
            gaps_df.to_csv(gaps_file, index=False)
            print(f"✅ Data gaps: 00-01-data-gaps.csv")
        
        # 4. Export conditions needing re-retrieval
        low_quality = [(c, self.quality_scores[c]) for c, s in self.quality_scores.items() if s < 60]
        if low_quality:
            reretrieval_data = []
            for condition, score in sorted(low_quality, key=lambda x: x[1]):
                stats = self.condition_stats[condition]
                clean_name = condition.replace('_', ' ')
                
                issues = []
                if stats['missing_years']:
                    issues.append(f"{len(stats['missing_years'])} missing years")
                if stats.get('cross_year_duplicate_rate', 0) > 5:
                    issues.append(f"{stats['cross_year_duplicate_rate']:.1f}% duplicates")
                if stats['papers_with_abstracts'] / stats['total_papers'] * 100 < 50 if stats['total_papers'] > 0 else False:
                    issues.append("Low abstract coverage")
                
                reretrieval_data.append({
                    'Condition': clean_name,
                    'Condition_Folder': condition,
                    'Quality_Score': round(score, 1),
                    'Total_Papers': stats['total_papers'],
                    'Issues': '; '.join(issues),
                    'Missing_Years_List': ', '.join(map(str, stats['missing_years'][:10])) + ('...' if len(stats['missing_years']) > 10 else '')
                })
            
            reretrieval_df = pd.DataFrame(reretrieval_data)
            reretrieval_file = os.path.join(self.output_dir, '00-01-conditions-for-reretrieval.csv')
            reretrieval_df.to_csv(reretrieval_file, index=False)
            print(f"✅ Re-retrieval list: 00-01-conditions-for-reretrieval.csv")
        
        # 5. Generate comprehensive text report
        report_file = os.path.join(self.output_dir, '00-01-validation-report.txt')
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PUBMED RETRIEVAL VALIDATION REPORT (00-01)\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {self.condition_data_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Analysis Period: {self.min_year}-{self.max_year}\n\n")
            
            # Overall statistics
            total_papers = sum(s.get('total_papers', 0) for s in self.condition_stats.values())
            total_unique = sum(s.get('unique_pmid_count', 0) for s in self.condition_stats.values())
            
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Conditions Analyzed: {len(self.available_conditions)}\n")
            f.write(f"Total Papers Retrieved: {total_papers:,}\n")
            f.write(f"Total Unique PMIDs: {total_unique:,}\n")
            f.write(f"Mean Papers per Condition: {total_papers/len(self.available_conditions):.0f}\n")
            f.write(f"Mean Quality Score: {np.mean(list(self.quality_scores.values())):.1f}\n")
            f.write(f"Median Quality Score: {np.median(list(self.quality_scores.values())):.1f}\n\n")
            
            # Quality distribution
            high = len([s for s in self.quality_scores.values() if s >= 80])
            medium = len([s for s in self.quality_scores.values() if 60 <= s < 80])
            low = len([s for s in self.quality_scores.values() if s < 60])
            
            f.write("QUALITY DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            f.write(f"High Quality (≥80): {high} conditions ({high/len(self.available_conditions)*100:.1f}%)\n")
            f.write(f"Medium Quality (60-79): {medium} conditions ({medium/len(self.available_conditions)*100:.1f}%)\n")
            f.write(f"Low Quality (<60): {low} conditions ({low/len(self.available_conditions)*100:.1f}%)\n\n")
            
            # Top quality conditions
            f.write("TOP 10 HIGH-QUALITY CONDITIONS\n")
            f.write("-"*40 + "\n")
            top_conditions = sorted(self.quality_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (condition, score) in enumerate(top_conditions, 1):
                clean_name = condition.replace('_', ' ')
                stats = self.condition_stats[condition]
                f.write(f"{i:2}. {clean_name}: {score:.1f} (Papers: {stats['total_papers']:,})\n")
            
            f.write("\n")
            
            # Data completeness
            complete = len([c for c in self.available_conditions 
                          if not self.condition_stats[c].get('missing_years')])
            
            f.write("DATA COMPLETENESS\n")
            f.write("-"*40 + "\n")
            f.write(f"Conditions with complete year coverage: {complete}\n")
            f.write(f"Conditions with missing years: {len(self.available_conditions) - complete}\n")
            f.write(f"Total missing condition-year combinations: {sum(len(s.get('missing_years', [])) for s in self.condition_stats.values())}\n\n")
            
            # Temporal analysis
            if hasattr(self, 'temporal_stats'):
                f.write("TEMPORAL ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(f"Overall Growth Rate: {self.temporal_stats.get('growth_rate', 0):.1f}%\n")
                f.write(f"Annual Trend Slope: {self.temporal_stats.get('trend_slope', 0):.1f} papers/year\n")
                f.write(f"Trend R²: {self.temporal_stats.get('trend_r2', 0):.3f}\n")
                f.write(f"Trend P-value: {self.temporal_stats.get('trend_p_value', 1):.4f}\n")
                
                if self.temporal_stats.get('trend_p_value', 1) < 0.05:
                    f.write("Statistical Significance: Yes (p < 0.05)\n\n")
                else:
                    f.write("Statistical Significance: No\n\n")
            
            # MeSH analysis
            if hasattr(self, 'mesh_stats'):
                f.write("MESH TERM ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(f"Mean MeSH Coverage: {self.mesh_stats.get('mean_coverage', 0):.1f}%\n")
                f.write(f"Total Unique MeSH Terms: {self.mesh_stats.get('total_unique_terms', 0):,}\n")
                f.write(f"Mean Terms per Paper: {self.mesh_stats.get('mean_terms_per_paper', 0):.1f}\n\n")
            
            # Issues and recommendations
            f.write("IDENTIFIED ISSUES\n")
            f.write("-"*40 + "\n")
            
            # Systematic gaps
            if hasattr(self, 'data_gaps'):
                suspicious = self.data_gaps.get('suspicious_patterns', [])
                if suspicious:
                    f.write("\nSystematic Issues:\n")
                    for pattern in suspicious:
                        f.write(f"  • {pattern['description']}\n")
                        if 'details' in pattern:
                            for year, count in sorted(pattern['details'].items())[:5]:
                                f.write(f"    - Year {year}: {count} conditions affected\n")
            
            # Conditions needing attention
            need_attention = [c for c, s in self.quality_scores.items() if s < 60]
            if need_attention:
                f.write(f"\nConditions Requiring Re-retrieval: {len(need_attention)}\n")
                f.write("Top 10 conditions needing attention:\n")
                for condition in sorted(need_attention, key=lambda x: self.quality_scores[x])[:10]:
                    clean_name = condition.replace('_', ' ')
                    score = self.quality_scores[condition]
                    stats = self.condition_stats[condition]
                    f.write(f"  • {clean_name} (Score: {score:.1f})\n")
                    if stats['missing_years']:
                        f.write(f"    - Missing {len(stats['missing_years'])} years\n")
                    if stats.get('cross_year_duplicate_rate', 0) > 5:
                        f.write(f"    - Cross-year duplicates: {stats['cross_year_duplicate_rate']:.1f}%\n")
            
            f.write("\n")
            
            # Final assessment
            f.write("FINAL ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            mean_score = np.mean(list(self.quality_scores.values()))
            if mean_score >= 80:
                f.write("✅ OVERALL CONFIDENCE: HIGH\n")
                f.write("The retrieved data is of high quality and ready for analysis.\n")
                f.write("Proceed with semantic analysis and DALY integration.\n")
            elif mean_score >= 60:
                f.write("⚠️  OVERALL CONFIDENCE: MEDIUM\n")
                f.write("The data is generally acceptable but has some quality issues.\n")
                f.write("Consider re-retrieving low-quality conditions before final analysis.\n")
                f.write(f"Conditions needing attention: {len(need_attention)}\n")
            else:
                f.write("❌ OVERALL CONFIDENCE: LOW\n")
                f.write("Significant data quality issues detected.\n")
                f.write("Review and improve the retrieval process before proceeding.\n")
                f.write(f"Conditions with major issues: {len(need_attention)}\n")
        
        print(f"✅ Report: 00-01-validation-report.txt")
        
        # 6. Create summary JSON for programmatic access
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'input_directory': self.condition_data_dir,
            'output_directory': self.output_dir,
            'year_range': f"{self.min_year}-{self.max_year}",
            'statistics': {
                'total_conditions': len(self.available_conditions),
                'total_papers': sum(s.get('total_papers', 0) for s in self.condition_stats.values()),
                'total_unique_pmids': sum(s.get('unique_pmid_count', 0) for s in self.condition_stats.values()),
                'mean_quality_score': float(np.mean(list(self.quality_scores.values()))),
                'median_quality_score': float(np.median(list(self.quality_scores.values()))),
                'high_quality_conditions': len([s for s in self.quality_scores.values() if s >= 80]),
                'medium_quality_conditions': len([s for s in self.quality_scores.values() if 60 <= s < 80]),
                'low_quality_conditions': len([s for s in self.quality_scores.values() if s < 60])
            },
            'temporal_stats': self.temporal_stats if hasattr(self, 'temporal_stats') else {},
            'mesh_stats': {
                'mean_coverage': self.mesh_stats.get('mean_coverage', 0),
                'total_unique_terms': self.mesh_stats.get('total_unique_terms', 0),
                'mean_terms_per_paper': self.mesh_stats.get('mean_terms_per_paper', 0)
            } if hasattr(self, 'mesh_stats') else {},
            'conditions_for_reretrieval': [c for c, s in self.quality_scores.items() if s < 60],
            'data_gaps_summary': {
                'conditions_with_missing_years': len(self.data_gaps.get('missing_years', {})),
                'total_missing_years': sum(len(years) for years in self.data_gaps.get('missing_years', {}).values()),
                'systematic_gaps': self.data_gaps.get('suspicious_patterns', [])
            } if hasattr(self, 'data_gaps') else {}
        }
        
        summary_file = os.path.join(self.output_dir, '00-01-validation-summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✅ Summary JSON: 00-01-validation-summary.json")
    
    def run_complete_validation(self):
        """Run complete validation pipeline"""
        print(f"\n🚀 STARTING COMPLETE VALIDATION ANALYSIS...")
        start_time = datetime.now()
        
        try:
            # Step 1: Generate confidence scores for all conditions
            self.generate_confidence_scores()
            
            # Step 2: Analyze temporal patterns
            self.analyze_temporal_patterns()
            
            # Step 3: Identify data gaps
            self.identify_data_gaps()
            
            # Step 4: Analyze MeSH quality
            self.analyze_mesh_quality()
            
            # Step 5: Create visualizations
            self.create_validation_visualizations()
            
            # Step 6: Export results
            self.export_validation_report()
            
            duration = datetime.now() - start_time
            print(f"\n✅ VALIDATION COMPLETE!")
            print(f"⏱️  Time taken: {duration}")
            
            # Final summary
            print(f"\n" + "="*70)
            print(f"📊 FINAL ASSESSMENT")
            print(f"="*70)
            
            high_quality = len([s for s in self.quality_scores.values() if s >= 80])
            total_papers = sum(s.get('total_papers', 0) for s in self.condition_stats.values())
            mean_score = np.mean(list(self.quality_scores.values()))
            
            print(f"   Conditions ready for analysis: {high_quality}/{len(self.available_conditions)}")
            print(f"   Total papers validated: {total_papers:,}")
            print(f"   Mean quality score: {mean_score:.1f}")
            
            if mean_score >= 80:
                print(f"\n   ✅ OVERALL CONFIDENCE: HIGH")
                print(f"   Data is ready for semantic analysis and further processing")
            elif mean_score >= 60:
                print(f"\n   ⚠️  OVERALL CONFIDENCE: MEDIUM")
                print(f"   Consider re-retrieving low-quality conditions before final analysis")
            else:
                print(f"\n   ❌ OVERALL CONFIDENCE: LOW")
                print(f"   Significant data quality issues - review retrieval process")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    print("🚀 PUBMED RETRIEVAL OUTPUT VALIDATOR (00-01)")
    print("Validating outputs from 00-00-00-pubmed-mesh-data-retrieval.py")
    print("Following directory structure: ANALYSIS/00-01-RETRIEVAL-VALIDATION/")
    print()
    
    # Check if data directory exists
    if not os.path.exists('./CONDITION_DATA/condition_progress'):
        print("❌ Data directory not found: ./CONDITION_DATA/condition_progress/")
        print("Please run 00-00-00-pubmed-mesh-data-retrieval.py first")
        return
    
    try:
        # Create validator and run analysis
        validator = PubMedRetrievalValidator(min_year=1975, max_year=2024)
        validator.run_complete_validation()
        
        print(f"\n📂 OUTPUT FILES CREATED IN ANALYSIS/00-01-RETRIEVAL-VALIDATION/:")
        print(f"   📊 00-01-validation-dashboard.png - Visual overview")
        print(f"   📋 00-01-condition-quality-scores.csv - Quality metrics per condition")
        print(f"   📈 00-01-temporal-paper-counts.csv - Papers by year")
        print(f"   🔍 00-01-data-gaps.csv - Missing data analysis")
        print(f"   ⚠️  00-01-conditions-for-reretrieval.csv - Low quality conditions")
        print(f"   📄 00-01-validation-report.txt - Comprehensive text report")
        print(f"   📊 00-01-validation-summary.json - Machine-readable summary")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()