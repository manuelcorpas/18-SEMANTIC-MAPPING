# PubMed Semantic Embedding Pipeline

## Overview
This project retrieves, validates, and processes PubMed literature data for a wide range of medical conditions, generating semantic embeddings using PubMedBERT. 
The pipeline is designed for GPU acceleration on AWS to significantly reduce processing time.

### Pipeline Stages
1. **Retrieval (`00-00-00-pubmed-mesh-data-retrieval.py`)**
   - Retrieves PubMed records per condition and year using MeSH terms.
   - Outputs stored under `CONDITION_DATA/`.

2. **Validation (`00-01-pubmed-analyzer.py`)**
   - Analyzes retrieval results and generates completeness scores for each condition.
   - Output: `ANALYSIS/00-01-RETRIEVAL-VALIDATION/00-01-condition-quality-scores.csv`.

3. **Embedding Generation (`00-01-00-semantic-embedding-generation.py`)**
   - Generates semantic embeddings for high-quality conditions using PubMedBERT.
   - Designed for AWS GPU acceleration (g4dn.xlarge recommended).
   - Output stored under `ANALYSIS/00-02-SEMANTIC-EMBEDDINGS/`.

---

## Current Progress
- **Retrieval stage:** Complete for selected conditions.
- **Completeness report:** Generated (`completeness_report.txt`), most conditions show >98% completeness. Some anomalies noted (e.g., conditions with <95% completeness or >105% totals).
- **Validation stage:** Pending — `00-01-pubmed-analyzer.py` needs to be run on retrieval outputs.
- **Embedding generation:** Pending — AWS GPU instance not yet launched due to account restrictions.

---

## AWS GPU Setup Status
- **Blocked**: Cannot yet access AWS Deep Learning AMI GPU PyTorch (Ubuntu 20.04).
- **Quota needed**: g4dn.xlarge instance type quota set to 0.
- **Action required**: Open AWS Support case to enable AMI access and request quota increase to 1 in `eu-west-2`.

---

## Next Actions
1. **AWS Support Request**
   - Enable AWS-provided Deep Learning AMI GPU PyTorch (Ubuntu 20.04).
   - Increase g4dn.xlarge quota to 1 in `eu-west-2`.

2. **(Optional) Preprocessing**
   - Review `completeness_report.txt` for low-quality conditions (<95% completeness) or anomalies.
   - Exclude these from embedding generation.

3. **Run Validation**
   ```bash
   python3 00-01-pubmed-analyzer.py
   ```

4. **Launch AWS GPU Instance**
   - AMI ID (London): `ami-0c90462c3f6ebf6e0`
   - Instance type: `g4dn.xlarge`
   - Storage: 100 GB gp3
   - IAM Role: AmazonS3FullAccess + AmazonSSMManagedInstanceCore

5. **Run Embedding Generation**
   ```bash
   screen -S embeddings
   python3 00-01-00-semantic-embedding-generation.py \
       --conditions 60 \
       --batch-size 64 \
       --min-quality 80.0
   ```

6. **Upload Results to S3 and Terminate Instance**
   ```bash
   aws s3 cp ANALYSIS/00-02-SEMANTIC-EMBEDDINGS/ s3://<bucket>/ --recursive
   ```

---

## Memory Jogger
> Continue from AWS GPU enablement → check completeness_report anomalies → run analyzer to produce quality scores → run embedding generation on AWS GPU with `--min-quality 80.0` → upload results to S3.

