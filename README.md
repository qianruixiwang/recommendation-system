# Recommendation System
This project implements a simple but extensible recommendation system using a combination of:
- Collaborative Filtering (Matrix Factorization via SVD)
- Content-Based Filtering (TF-IDF on item categories)
- Hybrid Recommendation (linear combination of CF and content scores)

## Overview

The pipeline consists of the following steps:
1. Load user–item interaction data and item metadata
2. Encode user and item IDs into contiguous integer indices
3. Construct a sparse user–item interaction matrix
4. Train a collaborative filtering model using truncated SVD
5. Build item content representations using TF-IDF
6. Generate recommendations using:
   - collaborative filtering
   - content-based similarity
   - a hybrid of both approaches

## Methods
**Collaborative Filtering**
- Uses matrix factorization (TruncatedSVD) on a sparse user–item matrix
- Learns low-dimensional latent representations for users and items
- Produces personalized recommendations based on latent similarity

**Content-Based Filtering**
- Encodes item categories using TF-IDF
- Computes item–item similarity via cosine similarity
- Recommends items similar in content space

**Hybrid Model**
- Combines collaborative and content-based scores: score = $\alpha$ $\cdot$ CF $\times$ (1 - $\alpha$) $\cdot$ Content
- Helps mitigate cold-start issues and improves robustness

## Dataset
This project is configured to work with the Amazon CD & Vinyl dataset. 

Source: https://amazon-reviews-2023.github.io/
