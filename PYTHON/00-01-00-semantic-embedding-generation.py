"""
SEMANTIC EMBEDDING GENERATION FOR HIGH-QUALITY PUBMED CONDITIONS

Generates semantic embeddings for 60 high-quality medical conditions from PubMed literature
to enable disease similarity analysis, DALY correlation, and research gap identification.

PROJECT: Disease Semantic Mapping Using PubMed Literature (Phase 2)

APPROACH:
1. Load high-quality conditions (≥80% quality score) from validation results
2. Generate embeddings using PubMedBERT for abstracts and MeSH terms
3. Create year-wise and aggregate embeddings for temporal analysis
4. Implement efficient batch processing for 3.46M papers
5. Visualize semantic space using UMAP
6. Prepare data structure for DALY correlation

INPUT: 
- CONDITION_DATA/condition_progress/[condition]/[condition]_[year].csv
- ANALYSIS/00-01-RETRIEVAL-VALIDATION/00-01-condition-quality-scores.csv

OUTPUT: 
- ANALYSIS/00-02-SEMANTIC-EMBEDDINGS/embeddings/
- ANALYSIS/00-02-SEMANTIC-EMBEDDINGS/visualizations/
- ANALYSIS/00-02-SEMANTIC-EMBEDDINGS/metadata/

USAGE: python 00-01-00-semantic-embedding-generation.py [--conditions N] [--batch-size B]

REQUIREMENTS: 
pip install transformers torch pandas numpy scikit-learn umap-learn matplotlib seaborn tqdm h5py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
import h5py
import pickle
from tqdm import tqdm
import gc
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import argparse

warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class SemanticEmbeddingGenerator:
    """Generate semantic embeddings for medical conditions using PubMedBERT"""
    
    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 batch_size=32, max_length=512, device=None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: HuggingFace model identifier for biomedical embeddings
            batch_size: Batch size for processing
            max_length: Maximum token length for BERT input
            device: Torch device (cuda/cpu)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"🔧 Initializing {model_name}")
        print(f"   Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup directory structure for outputs"""
        self.base_dir = Path("ANALYSIS/00-02-SEMANTIC-EMBEDDINGS")
        self.embeddings_dir = self.base_dir / "embeddings"
        self.viz_dir = self.base_dir / "visualizations"
        self.metadata_dir = self.base_dir / "metadata"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        
        for dir_path in [self.embeddings_dir, self.viz_dir, self.metadata_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def load_high_quality_conditions(self, min_quality_score=80):
        """Load conditions with quality score >= threshold"""
        quality_file = "ANALYSIS/00-01-RETRIEVAL-VALIDATION/00-01-condition-quality-scores.csv"
        
        if not os.path.exists(quality_file):
            raise FileNotFoundError(f"Quality scores file not found: {quality_file}")
            
        df_quality = pd.read_csv(quality_file)
        
        # FIXED: Handle the actual column names from your CSV (with capitals and underscores)
        # Check which column naming convention is used
        if 'Quality_Score' in df_quality.columns:
            # Rename columns to match what the rest of the script expects
            df_quality = df_quality.rename(columns={
                'Quality_Score': 'quality_score',
                'Total_Papers': 'total_papers',
                'Unique_PMIDs': 'unique_pmids',
                'Papers_With_Abstracts': 'papers_with_abstracts',
                'Abstract_Coverage_%': 'abstract_coverage',
                'Papers_With_MeSH': 'papers_with_mesh',
                'MeSH_Coverage_%': 'mesh_coverage',
                'Year_Coverage_%': 'year_coverage',
                'Missing_Years': 'missing_years',
                'Cross_Year_Duplicates_%': 'cross_year_duplicates',
                'Quality_Issues': 'quality_issues',
                'Anomalous_Years': 'anomalous_years',
                'Condition': 'condition',
                'Condition_Folder': 'condition_folder'
            })
        
        # Now filter by quality score
        high_quality = df_quality[df_quality['quality_score'] >= min_quality_score].copy()
        high_quality = high_quality.sort_values('quality_score', ascending=False)
        
        print(f"\n📊 High-Quality Conditions (≥{min_quality_score}% score):")
        print(f"   Total: {len(high_quality)} conditions")
        if len(high_quality) > 0:
            print(f"   Score range: {high_quality['quality_score'].min():.1f}% - {high_quality['quality_score'].max():.1f}%")
            print(f"   Total papers: {high_quality['total_papers'].sum():,}")
        
        return high_quality
    
    def prepare_text_for_embedding(self, row):
        """
        Prepare text from paper data for embedding
        Combines abstract and MeSH terms with special tokens
        """
        text_parts = []
        
        # Add title if available
        if pd.notna(row.get('Title', '')):
            text_parts.append(f"[TITLE] {row['Title']}")
        
        # Add abstract
        if pd.notna(row.get('Abstract', '')):
            abstract = str(row['Abstract'])[:2000]  # Limit abstract length
            text_parts.append(f"[ABSTRACT] {abstract}")
        
        # Add MeSH terms
        if pd.notna(row.get('MeSH_Terms', '')):
            mesh_terms = str(row['MeSH_Terms']).replace(';', ', ')
            text_parts.append(f"[MESH] {mesh_terms}")
            
        return " ".join(text_parts) if text_parts else ""
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
        return embeddings.cpu().numpy()
    
    def process_condition_papers(self, condition_name: str, condition_dir: Path,
                                years: Optional[List[int]] = None) -> Dict:
        """
        Process all papers for a condition and generate embeddings
        
        Returns:
            Dictionary with year-wise and aggregate embeddings
        """
        print(f"\n🔬 Processing: {condition_name}")
        
        # Find all CSV files for this condition
        csv_files = list(condition_dir.glob(f"{condition_name}_*.csv"))
        
        if not csv_files:
            print(f"   ⚠️ No data files found for {condition_name}")
            return None
            
        # Filter by years if specified
        if years:
            csv_files = [f for f in csv_files 
                        if any(str(year) in f.name for year in years)]
        
        # Sort by year
        csv_files.sort()
        
        embeddings_data = {
            'condition': condition_name,
            'yearly_embeddings': {},
            'yearly_metadata': {},
            'aggregate_embedding': None,
            'total_papers': 0
        }
        
        all_embeddings = []
        all_years = []
        
        for csv_file in tqdm(csv_files, desc=f"   Years for {condition_name}"):
            # Extract year from filename
            year_str = csv_file.stem.split('_')[-1]
            try:
                year = int(year_str)
            except ValueError:
                continue
                
            # Load data
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                continue
                
            # Process in batches
            batch_embeddings = []
            for i in range(0, len(df), self.batch_size):
                batch_df = df.iloc[i:i+self.batch_size]
                
                # Prepare texts
                texts = [self.prepare_text_for_embedding(row) 
                        for _, row in batch_df.iterrows()]
                
                # Filter empty texts
                valid_texts = [t for t in texts if t]
                
                if valid_texts:
                    # Generate embeddings
                    emb = self.generate_embeddings_batch(valid_texts)
                    batch_embeddings.append(emb)
            
            if batch_embeddings:
                # Combine all batches for this year
                year_embeddings = np.vstack(batch_embeddings)
                
                # Store year-wise data
                embeddings_data['yearly_embeddings'][year] = year_embeddings.mean(axis=0)
                embeddings_data['yearly_metadata'][year] = {
                    'paper_count': len(df),
                    'embedding_count': len(year_embeddings)
                }
                
                all_embeddings.append(year_embeddings)
                all_years.extend([year] * len(year_embeddings))
                
            # Clear memory
            del df
            gc.collect()
        
        # Calculate aggregate embedding
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            embeddings_data['aggregate_embedding'] = all_embeddings.mean(axis=0)
            embeddings_data['total_papers'] = len(all_embeddings)
            embeddings_data['years_covered'] = sorted(list(embeddings_data['yearly_embeddings'].keys()))
            
            print(f"   ✅ Processed {embeddings_data['total_papers']:,} papers")
            print(f"   📅 Years: {min(embeddings_data['years_covered'])}-{max(embeddings_data['years_covered'])}")
        
        return embeddings_data
    
    def save_embeddings(self, embeddings_data: Dict, condition_name: str):
        """Save embeddings to HDF5 format for efficient storage"""
        h5_file = self.embeddings_dir / f"{condition_name}_embeddings.h5"
        
        with h5py.File(h5_file, 'w') as f:
            # Save aggregate embedding
            if embeddings_data['aggregate_embedding'] is not None:
                f.create_dataset('aggregate_embedding', 
                               data=embeddings_data['aggregate_embedding'])
            
            # Save yearly embeddings
            yearly_group = f.create_group('yearly_embeddings')
            for year, emb in embeddings_data['yearly_embeddings'].items():
                yearly_group.create_dataset(str(year), data=emb)
            
            # Save metadata
            f.attrs['condition'] = condition_name
            f.attrs['total_papers'] = embeddings_data['total_papers']
            f.attrs['years_covered'] = json.dumps(embeddings_data.get('years_covered', []))
            
        # Save metadata separately
        metadata_file = self.metadata_dir / f"{condition_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'condition': condition_name,
                'total_papers': embeddings_data['total_papers'],
                'years_covered': embeddings_data.get('years_covered', []),
                'yearly_metadata': embeddings_data['yearly_metadata']
            }, f, indent=2)
            
        print(f"   💾 Saved: {h5_file.name}")
    
    def load_all_embeddings(self) -> Dict:
        """Load all saved embeddings for analysis"""
        all_embeddings = {}
        
        for h5_file in self.embeddings_dir.glob("*_embeddings.h5"):
            condition = h5_file.stem.replace('_embeddings', '')
            
            with h5py.File(h5_file, 'r') as f:
                all_embeddings[condition] = {
                    'aggregate': f['aggregate_embedding'][:] if 'aggregate_embedding' in f else None,
                    'yearly': {}
                }
                
                if 'yearly_embeddings' in f:
                    for year in f['yearly_embeddings'].keys():
                        all_embeddings[condition]['yearly'][int(year)] = f['yearly_embeddings'][year][:]
                        
        return all_embeddings
    
    def create_semantic_visualization(self, conditions_to_plot: Optional[List[str]] = None):
        """Create UMAP visualization of condition embeddings"""
        print("\n📊 Creating semantic space visualization...")
        
        # Load embeddings
        all_embeddings = self.load_all_embeddings()
        
        if conditions_to_plot:
            all_embeddings = {k: v for k, v in all_embeddings.items() 
                            if k in conditions_to_plot}
        
        # Prepare data for visualization
        condition_names = []
        embedding_matrix = []
        
        for condition, data in all_embeddings.items():
            if data['aggregate'] is not None:
                condition_names.append(condition)
                embedding_matrix.append(data['aggregate'])
        
        if len(embedding_matrix) < 2:
            print("   ⚠️ Not enough conditions with embeddings for visualization")
            return
        
        # FIX: Skip UMAP if too few conditions
        if len(embedding_matrix) < 5:
            print(f"   ⚠️ Only {len(embedding_matrix)} conditions - skipping UMAP (needs ≥5)")
            print(f"   Conditions processed: {', '.join(condition_names)}")
            return None, condition_names
            
        embedding_matrix = np.vstack(embedding_matrix)
        
        # FIX: Adjust n_neighbors based on sample size
        n_samples = len(embedding_matrix)
        n_neighbors = min(15, n_samples - 1)  # Ensure n_neighbors < n_samples
        
        # Apply UMAP
        print(f"   Reducing {embedding_matrix.shape} to 2D...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, 
                        min_dist=0.1, 
                        random_state=42)
        embedding_2d = reducer.fit_transform(embedding_matrix)
        
        # Rest of the visualization code...
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: UMAP projection
        scatter = ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            s=100, alpha=0.7, c=range(len(condition_names)), 
                            cmap='viridis')
        
        # Add labels
        for i, name in enumerate(condition_names):
            ax1.annotate(name[:20], (embedding_2d[i, 0], embedding_2d[i, 1]),
                       fontsize=8, alpha=0.8)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Semantic Space of Medical Conditions\n(UMAP Projection of PubMedBERT Embeddings)')
        
        # Plot 2: Temporal evolution for top conditions
        # Select top 10 conditions by paper count
        top_conditions = sorted(all_embeddings.keys(), 
                              key=lambda x: len(all_embeddings[x].get('yearly', {})),
                              reverse=True)[:10]
        
        # Calculate temporal distances
        for condition in top_conditions:
            yearly_data = all_embeddings[condition]['yearly']
            if len(yearly_data) > 5:  # Only plot if enough years
                years = sorted(yearly_data.keys())
                
                # Calculate distances between consecutive years
                distances = []
                for i in range(1, len(years)):
                    prev_emb = yearly_data[years[i-1]]
                    curr_emb = yearly_data[years[i]]
                    dist = np.linalg.norm(curr_emb - prev_emb)
                    distances.append(dist)
                
                if distances:
                    ax2.plot(years[1:], distances, label=condition[:20], alpha=0.7, marker='o', markersize=3)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Semantic Distance from Previous Year')
        ax2.set_title('Temporal Evolution of Disease Concepts\n(Year-to-Year Semantic Drift)')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        output_file = self.viz_dir / 'semantic_space_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        
        return embedding_2d, condition_names
    
    def create_similarity_matrix(self):
        """Create and visualize condition similarity matrix"""
        print("\n📊 Creating similarity matrix...")
        
        # Load embeddings
        all_embeddings = self.load_all_embeddings()
        
        # Filter conditions with embeddings
        conditions = []
        embeddings = []
        
        for condition, data in sorted(all_embeddings.items()):
            if data['aggregate'] is not None:
                conditions.append(condition)
                embeddings.append(data['aggregate'])
        
        if len(embeddings) < 2:
            print("   ⚠️ Not enough conditions for similarity matrix")
            return
            
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings_matrix = np.vstack(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot heatmap
        sns.heatmap(similarity_matrix, 
                   xticklabels=[c[:20] for c in conditions],
                   yticklabels=[c[:20] for c in conditions],
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, cbar_kws={'label': 'Cosine Similarity'})
        
        ax.set_title('Medical Condition Similarity Matrix\n(Based on PubMed Literature Embeddings)', 
                    fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save
        output_file = self.viz_dir / 'condition_similarity_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        
        # Save similarity matrix as CSV
        similarity_df = pd.DataFrame(similarity_matrix, 
                                    index=conditions, 
                                    columns=conditions)
        similarity_df.to_csv(self.metadata_dir / 'similarity_matrix.csv')
        
        return similarity_matrix, conditions
    
    def prepare_daly_correlation_data(self):
        """Prepare embedding data for DALY correlation analysis"""
        print("\n📊 Preparing data for DALY correlation...")
        
        # Load all embeddings
        all_embeddings = self.load_all_embeddings()
        
        # Create structured output for DALY analysis
        daly_ready_data = []
        
        for condition, data in all_embeddings.items():
            if data['aggregate'] is not None:
                record = {
                    'condition': condition,
                    'embedding': data['aggregate'].tolist(),
                    'embedding_dim': len(data['aggregate']),
                    'years_available': sorted(data['yearly'].keys()) if data['yearly'] else []
                }
                daly_ready_data.append(record)
        
        # Save for downstream analysis
        output_file = self.metadata_dir / 'embeddings_for_daly_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(daly_ready_data, f, indent=2)
            
        print(f"   ✅ Prepared {len(daly_ready_data)} conditions for DALY correlation")
        print(f"   💾 Saved: {output_file}")
        
        return daly_ready_data
    
    def generate_summary_report(self, conditions_processed: List[str]):
        """Generate comprehensive summary report"""
        print("\n📋 Generating summary report...")
        
        # Collect statistics
        all_embeddings = self.load_all_embeddings()
        
        total_conditions = len(conditions_processed)
        successful_conditions = len([c for c in all_embeddings if all_embeddings[c]['aggregate'] is not None])
        
        # Calculate temporal coverage
        temporal_coverage = {}
        for condition, data in all_embeddings.items():
            if data['yearly']:
                years = sorted(data['yearly'].keys())
                temporal_coverage[condition] = {
                    'start': min(years),
                    'end': max(years),
                    'count': len(years)
                }
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'model_used': self.model_name,
            'total_conditions_attempted': total_conditions,
            'successful_embeddings': successful_conditions,
            'embedding_dimension': 768,  # PubMedBERT dimension
            'temporal_coverage': temporal_coverage,
            'output_structure': {
                'embeddings': str(self.embeddings_dir),
                'visualizations': str(self.viz_dir),
                'metadata': str(self.metadata_dir)
            }
        }
        
        # Save summary
        summary_file = self.base_dir / 'embedding_generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print summary
        print(f"""
SEMANTIC EMBEDDING GENERATION SUMMARY
{'='*50}
Model: {self.model_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CONDITIONS PROCESSED:
  Attempted: {total_conditions}
  Successful: {successful_conditions}
  Success Rate: {(successful_conditions/total_conditions)*100:.1f}%

EMBEDDING CHARACTERISTICS:
  Dimension: 768
  Type: Contextual (BERT-based)
  Aggregation: Mean pooling

TEMPORAL COVERAGE:
  Conditions with yearly data: {len(temporal_coverage)}
  Average years per condition: {np.mean([v['count'] for v in temporal_coverage.values()]):.1f if temporal_coverage else 0}

OUTPUT FILES:
  Embeddings: {self.embeddings_dir}
  Visualizations: {self.viz_dir}
  Metadata: {self.metadata_dir}

READY FOR:
  ✅ Disease similarity analysis
  ✅ DALY correlation studies
  ✅ Temporal evolution analysis
  ✅ Research gap identification
""")
        
        print(f"✅ Summary saved: {summary_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate semantic embeddings for PubMed conditions')
    parser.add_argument('--conditions', type=int, default=60, 
                       help='Number of top conditions to process')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--min-quality', type=float, default=80.0,
                       help='Minimum quality score for conditions')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with limited data')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SEMANTIC EMBEDDING GENERATION FOR PUBMED CONDITIONS")
    print("Phase 2: Disease Semantic Mapping")
    print("="*70)
    
    # Initialize generator
    generator = SemanticEmbeddingGenerator(batch_size=args.batch_size)
    
    # Load high-quality conditions
    high_quality_conditions = generator.load_high_quality_conditions(
        min_quality_score=args.min_quality
    )
    
    # Limit conditions if specified
    conditions_to_process = high_quality_conditions.head(args.conditions)
    
    if args.test_mode:
        print("\n⚠️ TEST MODE: Processing only first 3 conditions with limited years")
        conditions_to_process = conditions_to_process.head(3)
        test_years = [2020, 2021, 2022, 2023, 2024]
    else:
        test_years = None
    
    print(f"\n🚀 Processing {len(conditions_to_process)} conditions...")
    
    # Process each condition
    processed_conditions = []
    
    for idx, row in conditions_to_process.iterrows():
        # Handle both 'condition' and 'Condition' column names
        condition_name = row.get('condition', row.get('Condition', ''))
        
        # Also handle 'condition_folder' or 'Condition_Folder' if present
        if 'condition_folder' in row:
            folder_name = row['condition_folder']
        elif 'Condition_Folder' in row:
            folder_name = row['Condition_Folder']
        else:
            folder_name = condition_name
            
        condition_dir = Path(f"CONDITION_DATA/condition_progress/{folder_name}")
        
        if not condition_dir.exists():
            print(f"⚠️ Directory not found for {condition_name} at {condition_dir}")
            continue
        
        try:
            # Generate embeddings
            embeddings_data = generator.process_condition_papers(
                folder_name,  # Use folder name for file matching
                condition_dir,
                years=test_years
            )
            
            if embeddings_data and embeddings_data['aggregate_embedding'] is not None:
                # Save embeddings
                generator.save_embeddings(embeddings_data, condition_name)
                processed_conditions.append(condition_name)
                
                # Save checkpoint
                checkpoint = {
                    'processed': processed_conditions,
                    'current_index': idx,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint_file = generator.checkpoint_dir / 'processing_checkpoint.json'
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                    
        except Exception as e:
            print(f"   ❌ Error processing {condition_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n✅ Successfully processed {len(processed_conditions)} conditions")
    
    # Generate visualizations and analyses
    if len(processed_conditions) >= 2:
        print("\n📊 Generating visualizations and analyses...")
        
        # Create semantic space visualization
        generator.create_semantic_visualization()
        
        # Create similarity matrix
        generator.create_similarity_matrix()
        
        # Prepare DALY correlation data
        generator.prepare_daly_correlation_data()
    
    # Generate summary report
    generator.generate_summary_report(processed_conditions)
    
    print("\n🎯 Semantic embedding generation complete!")
    print(f"📂 All outputs saved to: {generator.base_dir}")

if __name__ == "__main__":
    main()