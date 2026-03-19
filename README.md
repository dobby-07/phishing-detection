# phishing-detection
An Attention-Enhanced Multi-Branch Framework for Intelligent Phishing Detection on Mobile and Web Platforms.

🚀 FusionPhishGuard

Attention-Enhanced Multi-Branch Framework for Phishing Detection
Overview

FusionPhishGuard is an open-source deep learning framework for detecting phishing URLs across web and mobile platforms.

The system is designed to handle modern phishing attacks by combining multiple representation techniques and using an attention-based fusion mechanism to improve detection accuracy and robustness.

It is optimized for real-world deployment, providing high accuracy while maintaining reasonable latency.

Why FusionPhishGuard?

Traditional phishing detection approaches struggle with:

❌ Zero-day phishing attacks

❌ URL obfuscation techniques

❌ Poor generalization across platforms

❌ Dependence on handcrafted features

FusionPhishGuard addresses these issues by:

✅ Learning from multi-level representations (lexical, contextual, semantic)

✅ Using attention-based fusion to prioritize important features

✅ Supporting both mobile and web phishing scenarios

✅ Providing high accuracy with strong generalization
Datasets

FusionPhishGuard is evaluated on:

CatchPhish

~82K URLs

Balanced dataset

Includes real-world phishing patterns

PhishDump

~331K URLs

Includes mobile redirect phishing

Large-scale real-world dataset

Results

| Model                | CatchPhish Accuracy | PhishDump Accuracy |
| -------------------- | ------------------- | ------------------ |
| Word Embeddings      | ~91%                | ~93%               |
| Transformer Models   | ~93–95%             | ~95–96%            |
| LLM Models           | ~95–97%             | ~96–97%            |
| Fusion (Final Model) | **96.85%**          | **98.11%**         |


Performance

Latency: ~110–120 ms per URL

Parameters: ~585M

Optimized for: Real-time phishing detection

Installation
Python
pip install -r requirements.txt

Train the Model
python train.py

Evaluate
python evaluate.py
