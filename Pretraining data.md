# Pretraining Data

The first step in training an LLM is to build the dataset.

## FineWeb

- The best open dataset today is [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) by HuggingFace, currently at 18.5T tokens.
- The data comes from 96 [CommonCrawl](https://commoncrawl.org/) snapshots (2013-2024). Filtered to be mostly English.
- To process it they created [datatrove](https://github.com/huggingface/datatrove), an open-source text processing library. The processing code is fully published.
- To evaluate data quality and ablate the processing pipeline, they train small models on a representative subset of the dataset and evaluate them on a set of early-signal benchmark tasks. The models are trained via their pretraining library [nanotron](https://github.com/huggingface/nanotron) and evaluated via [Lighteval](https://github.com/huggingface/lighteval/).
- Details on how FineWeb was built in the [blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). Processing pipeline: 1) URL via block-lists and subword detection, (2) Trafilatura for HTML text extraction, (3) FastText language filtering (English score ≥0.65), (4) quality filtering combining Gopher repetition checks, C4 filters, and custom FineWeb heuristics (5) MinHash deduplication per crawl (5-grams, 14×8 hash functions), and (6) PII anonymization for emails and IP addresses.
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is a dataset built by filtering the most educational data from FineWeb. It comes in 1.3T tokens and 5.4T tokens. This was [used by Karpathy](https://github.com/karpathy/nanochat/discussions/1) to train nanochat.

![FineWeb processing pipeline](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/05cbb1ce-e7bc-4534-883e-2fb6a0abf9a9/image.png)
