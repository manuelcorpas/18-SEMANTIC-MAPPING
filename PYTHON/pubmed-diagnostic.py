#!/usr/bin/env python3
"""
Diagnostic script to test PubMed API connection and identify issues
"""

import os
import sys
import time
from Bio import Entrez
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Entrez
Entrez.email = "mc@manuelcorpas.com"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"
Entrez.tool = "DiagnosticTool"

def test_basic_connection():
    """Test basic PubMed connection"""
    print("\n" + "="*60)
    print("TEST 1: Basic PubMed Connection")
    print("="*60)

    try:
        # Simple search for a single article
        query = '"cancer"[Title] AND "2023"[Date - Publication]'
        print(f"Query: {query}")

        handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
        result = Entrez.read(handle)
        handle.close()

        count = int(result.get("Count", 0))
        print(f"✅ SUCCESS: Found {count:,} articles")
        print(f"   IdList: {result.get('IdList', [])[:5]}")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_condition_search():
    """Test searching for a specific condition"""
    print("\n" + "="*60)
    print("TEST 2: Condition-Specific Search")
    print("="*60)

    condition = "Breast cancer"
    year = 2023

    try:
        query = (
            '"Breast Neoplasms"[MeSH] OR "breast cancer"[Title/Abstract] '
            'AND "2023"[Date - Publication] '
            'AND "journal article"[Publication Type] '
            'AND english[Language] '
            'AND humans[MeSH Terms] '
            'AND hasabstract[text]'
        )

        print(f"Condition: {condition}")
        print(f"Year: {year}")
        print(f"Query length: {len(query)} characters")

        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        result = Entrez.read(handle)
        handle.close()

        count = int(result.get("Count", 0))
        print(f"✅ SUCCESS: Found {count:,} articles for {condition} in {year}")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.error(f"Full error: ", exc_info=True)
        return False

def test_fetch_articles():
    """Test fetching article details"""
    print("\n" + "="*60)
    print("TEST 3: Fetching Article Details")
    print("="*60)

    try:
        # First get some PMIDs
        query = '"cancer"[Title] AND "2023"[Date - Publication]'
        handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
        search_result = Entrez.read(handle)
        handle.close()

        pmids = search_result.get("IdList", [])
        if not pmids:
            print("❌ No PMIDs found to fetch")
            return False

        print(f"Fetching details for PMIDs: {pmids[:3]}")

        # Fetch the articles
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids[:3],
            rettype="xml",
            retmode="xml"
        )
        records = handle.read()
        handle.close()

        print(f"✅ SUCCESS: Fetched {len(records)} bytes of XML data")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting with multiple requests"""
    print("\n" + "="*60)
    print("TEST 4: Rate Limiting (5 rapid requests)")
    print("="*60)

    try:
        delay = 0.1 if Entrez.api_key else 0.34
        print(f"Using delay: {delay} seconds between requests")

        for i in range(5):
            start = time.time()

            query = f'"cancer"[Title] AND "202{i}"[Date - Publication]'
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()

            elapsed = time.time() - start
            count = int(result.get("Count", 0))
            print(f"  Request {i+1}: {count:,} results in {elapsed:.2f}s")

            time.sleep(delay)

        print("✅ SUCCESS: All requests completed")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_checkpoint_system():
    """Test checkpoint file operations"""
    print("\n" + "="*60)
    print("TEST 5: Checkpoint System")
    print("="*60)

    checkpoint_file = "test_checkpoint.json"

    try:
        # Write test checkpoint
        test_data = {
            "test_condition": {
                "2023": {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat()
                }
            }
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(test_data, f)
        print("✅ Checkpoint write successful")

        # Read test checkpoint
        with open(checkpoint_file, 'r') as f:
            loaded_data = json.load(f)
        print("✅ Checkpoint read successful")

        # Clean up
        os.remove(checkpoint_file)
        print("✅ Checkpoint cleanup successful")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_all_conditions_quick():
    """Quick test of first 5 conditions"""
    print("\n" + "="*60)
    print("TEST 6: Quick Test of Multiple Conditions")
    print("="*60)

    conditions = [
        "Breast cancer",
        "COVID-19",
        "Diabetes mellitus",
        "Alzheimer's disease and other dementias",
        "Stroke"
    ]

    success_count = 0
    for condition in conditions:
        try:
            # Simple query for 2023
            if condition == "COVID-19":
                query = '"COVID-19"[MeSH] AND "2023"[Date - Publication]'
            elif condition == "Breast cancer":
                query = '"Breast Neoplasms"[MeSH] AND "2023"[Date - Publication]'
            elif condition == "Diabetes mellitus":
                query = '"Diabetes Mellitus"[MeSH] AND "2023"[Date - Publication]'
            elif "Alzheimer" in condition:
                query = '"Alzheimer Disease"[MeSH] AND "2023"[Date - Publication]'
            else:
                query = f'"{condition}"[Title/Abstract] AND "2023"[Date - Publication]'

            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()

            count = int(result.get("Count", 0))
            print(f"  {condition}: {count:,} articles")
            success_count += 1

            time.sleep(0.1)

        except Exception as e:
            print(f"  {condition}: ❌ FAILED - {e}")

    print(f"\n✅ Successfully queried {success_count}/{len(conditions)} conditions")
    return success_count == len(conditions)

def main():
    print("="*60)
    print("PUBMED API DIAGNOSTIC TEST")
    print("="*60)
    print(f"Email: {Entrez.email}")
    print(f"API Key: {'Set' if Entrez.api_key else 'Not set'}")
    print(f"Time: {datetime.now()}")

    tests = [
        ("Basic Connection", test_basic_connection),
        ("Condition Search", test_condition_search),
        ("Fetch Articles", test_fetch_articles),
        ("Rate Limiting", test_rate_limiting),
        ("Checkpoint System", test_checkpoint_system),
        ("Multiple Conditions", test_all_conditions_quick)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠️  Some tests failed. Possible issues:")
        print("1. Check your internet connection")
        print("2. Verify your NCBI API key is valid")
        print("3. Check if PubMed/NCBI services are accessible")
        print("4. Try reducing the number of parallel threads")
        print("5. Check file permissions in the output directory")
    else:
        print("\n✅ All tests passed! The main script should work.")
        print("If it's still stuck, try:")
        print("1. Reduce MAX_PARALLEL_THREADS to 2 or 3")
        print("2. Clear the checkpoint file and restart")
        print("3. Check available disk space")
        print("4. Monitor system resources (CPU/Memory)")

if __name__ == "__main__":
    main()
