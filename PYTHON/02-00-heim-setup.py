#!/usr/bin/env python3
"""
02-00-HEIM-SETUP: Create HEIM Directory Structure and Core Files

RUN AS: python PYTHON/02-00-heim-setup.py

This script sets up the HEIM infrastructure following the exact directory hierarchy:
- Scripts in PYTHON/ with 02-XX- prefix
- Output in ANALYSIS/02-00-HEIM-ANALYSIS/
- All directory names in CAPITALS
- Run from root directory (.)
"""

import os
import sys
from pathlib import Path

def create_heim_structure():
    """Create HEIM directory structure following the naming convention"""
    
    print("="*70)
    print("HEIM SETUP - CREATING DIRECTORY STRUCTURE")
    print("="*70)
    print("Following directory hierarchy:")
    print("  Scripts: PYTHON/02-XX-heim-*.py")
    print("  Output: ANALYSIS/02-00-HEIM-ANALYSIS/")
    print("  Run from: . (root directory)")
    print()
    
    # Define directories to create
    directories = [
        "ANALYSIS/02-00-HEIM-ANALYSIS",
        "ANALYSIS/02-00-HEIM-ANALYSIS/BASELINE-SCORES",
        "ANALYSIS/02-00-HEIM-ANALYSIS/DISEASE-SCORES", 
        "ANALYSIS/02-00-HEIM-ANALYSIS/VISUALIZATIONS",
        "ANALYSIS/02-00-HEIM-ANALYSIS/REPORTS",
        "DATA/HEIM-PROCESSED"
    ]
    
    # Create directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print()
    print("Directory structure created successfully!")
    
    # Create README file
    readme_content = """HEIM ANALYSIS DIRECTORY STRUCTURE
====================================

02-00-HEIM-ANALYSIS/
├── BASELINE-SCORES/     # Initial HEIM calculations
├── DISEASE-SCORES/      # Disease-specific HEIM scores
├── VISUALIZATIONS/      # HEIM dashboards and plots
└── REPORTS/            # Final reports and summaries

PYTHON SCRIPTS:
- 02-00-heim-setup.py           # Setup script (this file)
- 02-01-heim-core.py           # Core HEIM calculator
- 02-02-heim-integration.py    # Data integration
- 02-03-heim-disease-analysis.py # Disease-level analysis
- 02-04-heim-visualization.py   # Visualization dashboard

RUN ORDER:
1. python PYTHON/02-00-heim-setup.py (setup)
2. python PYTHON/02-02-heim-integration.py (baseline)
3. python PYTHON/02-03-heim-disease-analysis.py (diseases)
4. python PYTHON/02-04-heim-visualization.py (plots)

DATA DEPENDENCIES:
- DATA/pubmed_complete_dataset.csv
- DATA/GBD-DISEASES/master_gbd_diseases_papers.csv
- ANALYSIS/FULL-DATASET-GBD2021-ANALYSIS/comprehensive_research_gaps_gbd2021_full_dataset.csv
"""
    
    readme_path = Path("ANALYSIS/02-00-HEIM-ANALYSIS/README.txt")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created README: {readme_path}")
    
    # Check for required data files
    print("\nChecking for required data files...")
    required_files = [
        "DATA/pubmed_complete_dataset.csv",
        "DATA/GBD-DISEASES/master_gbd_diseases_papers.csv",
        "ANALYSIS/FULL-DATASET-GBD2021-ANALYSIS/comprehensive_research_gaps_gbd2021_full_dataset.csv"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_present = False
    
    if all_present:
        print("\n✅ All required data files present - ready for HEIM analysis!")
    else:
        print("\n⚠️ Some data files missing - HEIM analysis may be limited")
    
    print("\n📋 NEXT STEPS:")
    print("1. Save HEIM core calculator as: PYTHON/02-01-heim-core.py")
    print("2. Run integration: python PYTHON/02-02-heim-integration.py")
    print("3. Run disease analysis: python PYTHON/02-03-heim-disease-analysis.py")
    print("4. Create visualizations: python PYTHON/02-04-heim-visualization.py")
    
    return all_present

if __name__ == "__main__":
    # Ensure we're running from root directory
    if not os.path.exists("PYTHON") or not os.path.exists("DATA"):
        print("ERROR: Must run from root directory containing PYTHON/ and DATA/")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    all_present = create_heim_structure()
    
    if all_present:
        print("\n✓ Setup complete! Ready to proceed with HEIM analysis.")
    else:
        print("\n⚠️ Setup complete but some data files are missing.")