# Tokenization

[Karpathy Explainer](https://www.youtube.com/watch?v=zduSFxRajkE).

[Gwern's tokenization directory](https://gwern.net/doc/ai/nn/tokenization/index).

### Introduction

[Tiktokenizer](https://tiktokenizer.vercel.app/) app to play with tokens.

- Tokenization is the process of translating between text strings and tokens (encoding) and the reverse (decoding).
- Ideally we'd skip tokenization and feed single characters/bytes to the models. But that would fill up the context window too quickly and shoot up computational costs.
- We cannot use words as tokens because the vocab would be too big and the model won't be able to deal with new words.
- So we use subword tokenization with algorithms like Byte Pair Encoding (BPE).
- Tokenization is an independent step in the LLM pipeline. You gather a dataset, (typically smaller than the pretraining dataset), train a tokenizer, then use that tokenizer to encode/decode text.

| **Feature** | **Word-level** | **Character-level** | **Subword (LLMs)** |
|-------------|----------------|---------------------|-------------------|
| **Vocabulary Size** | Massive (1M+) | Tiny (~256) | Balanced (32k–128k) |
| **Sequence Length** | Shortest | Longest | **Optimized** |
| **Handles New Words** | No (OOV) | Yes | **Yes (breaks them down)** |
| **Semantic Meaning** | High | Low | **High** |

### Unicode

Blog: [A Programmer's Introduction to Unicode.](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)

- [Unicode](https://www.wikiwand.com/en/articles/Unicode) is the universal standard for text encoding, currently at ~160k characters.
- Each character has a unique Code Point. For example, the letter `k` is `U+006B`. `6B` is a hexadecimal number: in decimal 107.
- The Code Point can be represented via three different encodings: UTF-8, UTF-16 and UTF-32.
- UTF-8 is the most common encoding and is backwards compatible with ASCII ([video explainer for text encoding](https://www.youtube.com/watch?v=GMF2Z1EZHXk&t=6s)).
- UTF-8 is a variable-length encoding using between 1 and 4 bytes. Common characters (e.g. ASCII) have an efficient representation in memory using only 1 byte, while newer characters (like emoji) expand to use the full 4 bytes. The first byte header communicates how many bytes are being used.

Encoding the letter `k` (U+006B):

| **Encoding** | **Binary Representation** | **Hexadecimal** | **Total Bytes** |
|--------------|---------------------------|-----------------|-----------------|
| **UTF-8** | `01101011` | `6B` | **1 Byte** |
| **UTF-16** | `00000000 01101011` | `006B` | **2 Bytes** |
| **UTF-32** | `00000000 00000000 00000000 01101011` | `0000006B` | **4 Bytes** |

Encoding ghost emoji (U+1F47B):

| **Encoding** | **Binary Representation** | **Hexadecimal** | **Total Bytes** |
|--------------|---------------------------|-----------------|-----------------|
| **UTF-8** | `11110000 10011111 10010001 10111011` | `F0 9F 91 BB` | **4 Bytes** |
| **UTF-16** | `11011000 00111101 11011100 01111011` | `D83D DC7B` | **4 Bytes** |
| **UTF-32** | `00000000 00000001 11110100 01111011` | `0001F47B` | **4 Bytes** |

### UTF-8 and the base vocabulary

Nearly all LLM tokenizers start from the UTF-8 encoding.

All UTF-8 characters are represented via 1-4 bytes. There are 256 possible bytes. They constitute our base vocabulary.

So the first step in training the tokenizer is to map each of these 256 bytes to an index in the token dictionary.

Since every character in existence—no matter how rare—can be represented as a sequence of these 256 bytes, the model can always spell out whatever it sees in bytes. This is known as **byte fallback**.

![Byte fallback example](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/9a987f30-bb9c-4750-bf4b-7b5c0a7131d6/image.png)

*In this example, the tokenizer encounters a rare character (Egyptian Hieroglyph). Because of its rarity, the tokenizer has not learned a shortcut for this character. Instead, it falls back to reading the 4 raw bytes (using the tokens from the initial vocabulary of 256 bytes). Thanks to this fallback, models can still read any character in Unicode no matter how rare.*

If we stopped here, we'd have a complete tokenizer with a vocabulary of 256 tokens, able to represent any existing text. However, we need a bigger vocabulary size for training an LLM. That's where BPE kicks in.

### The vocab size hyperparameter

The most important hyperparameter when training a tokenizer: what should be the vocabulary size `V`?

The vocabulary contains:
- The 256 raw bytes (base vocabulary)
- Special tokens like `<|endoftext|>` or `<|eos|>`
- Learned BPE tokens like `dog`

We control `V` by deciding for how long to run BPE.

The size of `V` creates a tradeoff between different factors.

| **Feature** | **Small Vocab (32k)** | **Large Vocab (128k+)** | **The "Why"** |
|-------------|----------------------|------------------------|---------------|
| Sequence Length | Longer. Words are split into more sub-tokens. | Shorter. More words/phrases fit into 1 token. | Larger V increases "tokenization fertility" (more information per token). |
| Effective Context | Reduced. The context window fills up with "word fragments." | Expanded. You can fit more "actual text" into the same 8k window. | A 128k vocab can fit ~3-4x more non-English text in the same context window than a 32k vocab. |
| Model Size | Leaner. Embedding and Head layers are small. | Heavier. Embedding and Head can be 20%+ of total parameters. | These layers scale linearly with V. For small models (e.g., 2B), a large V dominates the parameter count. |
| Inference Speed | Faster per-token. The final Softmax math is "light." | Slower per-token. The final Softmax must score 100k+ options. | The Output Head is a bottleneck; calculating 128k logits takes significantly more GPU cycles than 32k. |
| Training Quality | Dense. Common tokens get millions of updates. | Sparse. Rare tokens may suffer from "under-fitting." | In a massive vocab, rare tokens (the "long tail") might not appear enough during training to learn a good embedding. |
| Multilingualism | Poor. Non-English text is over-fragmented (The "Language Tax"). | Excellent. Better representation of diverse scripts and specialized terms. | Small vocabs are usually English-centric; large vocabs allow "dedicated" tokens for other languages. |

In [nanochat](https://github.com/karpathy/nanochat/discussions/1), `V` is `2**16 = 65,536`. More examples:

| **Model Family** | **Tokenizer Type** | **Vocab Size (V)** | **Primary Focus** |
|------------------|--------------------|--------------------|-------------------|
| **BERT** | WordPiece | **30,522** | Scientific/Academic English |
| **Llama 2** | SentencePiece (BPE) | **32,000** | Efficient, English-centric |
| **Mistral 7B** | Llama-compatible | **32,768** | High efficiency, common words |
| **GPT-2 / GPT-3** | Byte-Level BPE | **50,257** | General English & Code |
| **GPT-4 / 4o** | Tiktoken (cl100k_base) | **~100,256** | Balanced Multilingual & Logic |
| **Llama 3 / 3.1** | Tiktoken (BPE) | **128,256** | High Compression, Multilingual |
| **Gemma 2** | SentencePiece (BPE) | **256,000** | Deep Multilingual & Cultural |
| **Gemma 3** | SentencePiece (BPE) | **~262,000** | Native Multimodal & Global |

### How vocab size impacts model parameters

At the level of the transformer, the dimension of `V` affects two layers:
- Embedding layer (input). This is the first layer. Lookup table of shape (`V`, `d`). It has a row for each token. It embeds each token in a vector of dimension `d`.
- Transformer head (output). Last layer before Softmax. It has shape (`d`, `V`). It projects the hidden state on each possible output token to get the logits (which are then converted to probabilities).

![Model head example](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/b8249f06-5fff-4526-8789-43079bf830b9/image.png)

*Toy example of model head for a model with **`V=4`** and **`d=3`**.*

Consequently, each new token in the vocabulary adds `2d` parameters to the model. In a model with a hidden dimension (`d`) of 4,096, **every 1,000 tokens you add to the vocabulary adds about 8 Million parameters to the model** (1000 * 4096 * 2 for the embedding and the head).

### Byte Pair Encoding

To apply BPE, you need a training corpus. This is an independent stage from the LLM training.

Take your text corpus and access the UTF-8 encoding to convert it into a stream of bytes. These are then tokenized into the base vocabulary (e.g. 0-255).

The [BPE algorithm](https://www.wikiwand.com/en/articles/Byte-pair_encoding) works by recursively finding the most frequent byte-pair and assigning it a unique code.

Note: because BPE is greedy (always picks the most frequent pair) **the final result is optimized for the tokenizer training corpus**. If the corpus has a lot of code, the tokenizer will represent code efficiently; if it has very little Chinese, it will not compress that language well. In general, current tokenizers are very efficient with English but spend a lot of tokens on other languages.

At the end you get a dictionary like this:

| **ID** | **Token** | **Origin (Merge Rule)** |
|--------|-----------|-------------------------|
| ... | ... | ... |
| **104** | `h` | Base Byte |
| **101** | `e` | Base Byte |
| **256** | `he` | Merge `104` + `101` (`h` + `e`) |
| **257** | `ll` | Merge `108` + `108` (`l` + `l`) |
| **258** | `hello` | Merge `256` + `257` + `111` |

The vocab contains the base bytes (0-255). The merges are recursive: token 258 merges tokens 256 and 257, token 256 merges tokens 104 and 101, etc.

To encode, split the text into bytes and apply the merge rules in the exact order they were learned.

![BPE encoding example](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/873003a5-a092-4510-9ed7-0b99e79c8ddf/image.png)

To decode, we use a simple vocabulary that maps each token ID to its original byte(s) sequence: `{258: [104, 101, 108, 108, 111]}`.

After conversion, concatenate all the bytes into a single byte stream, then use UTF-8 to convert it back to text.

Note: the model could generate invalid byte sequences that fail the UTF-8 standard. There are different ways to deal with this issue, like swapping illegal bytes with a replacement character.

Note: UTF-8 encodes characters as a sequence of 1-4 bytes. When you feed a 4-byte character like an emoji or a rare hieroglyph into the tokenizer, the Pre-tokenization Regex doesn't see a "character"—it sees a sequence of bytes. Since BPE's base vocabulary is only the 256 individual bytes, those 4 bytes start their life as four separate tokens. Even though they represent one single symbol, the model initially treats them as four distinct "words" in its language. But if the character is common, there's a good chance that those 4 bytes will be merged back together by BPE.

### **Pre-tokenization**

In the above explanation, the entire text is converted to a single byte stream and then BPE is applied. In reality, a pre-tokenization phase is used to force splits across character categories, and prevent certain tokens from being formed.

**Pre-tokenization** uses a regex to split the text into buckets. Tokenization is only allowed to happen **within** buckets but not **between** them. This prevents the BPE from wasting vocab space on suboptimal tokens: merging words with punctuation, merging letters and numbers, using a single token for each whitespace etc.

The GPT-4 pretokenization string ([source](https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py)):

```javascript
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

![Pre-tokenization example](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/81756f80-349e-422e-9eac-225c1c7aa380/image.png)

*We can see the regex rules at work here. The contractions **`'t`** and **`'s`** are tokenized separately, so is the punctuation mark **`!`**. **`2024`** is split in two tokens, since the rules forbid having a single token for numbers above 3 digits.*

The pre-tokenization stage is applied both in training and inference.

### Pipeline example

Let's tokenize the sentence `Don't wait! It's 2024.`

First, the regex pre-tokenization will split it into the following buckets:

`["Don", "'t", " wait", "!", " It", "'s", " 20", "24", "."]`

Notice how 2024 was split in two parts. This is to prevent the tokenizer from wasting its vocabulary on suboptimal number tokens like `2024`.

Next, each bucket is converted into its raw UTF-8 byte sequence.

`_wait` → `[32, 119, 97, 105, 116]`

Next, the tokenizer will apply the merge rules **within each bucket**. So `_wait` might be reduced to a single token.

Finally, all the token IDs are returned: `[342, 470, 1432, 0, 552, 338, 2024, 13]`.

### Special tokens

Special tokens are added to the vocabulary after the BPE training is complete. They are used to give extra information to models and direct their behavior. Here are a few examples.

| **Token** | **Name** | **Function** |
|-----------|----------|--------------|
| **BOS / `<s>`** | Beginning of Sequence | Signals the start of a text. It helps the model reset its "internal state" for a new document. |
| **EOS / `</s>`** | End of Sequence | Tells the model to stop generating. Without this, the model would produce endless gibberish until hitting its max limit. |
| **PAD** | Padding | Used in batching to make all sequences the same length. The model is taught to ignore these "empty" slots. |
| **UNK / `<unk>`** | Unknown | A fallback for characters that literally don't exist in the vocabulary (though rare in Byte-Level BPE). |
| **Role Tokens** | Control Tokens | Used in chat models (like `<|user|>`) |

The tokenizer will not further break down these tokens (they are atomic).

Special tokens are generally sanitized from user inputs to prevent jailbreaks.

![User input mimicking special token](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/bcd28cc4-a93f-43c4-9b44-38a12006d50f/image.png)

*User input mimicking a special token. Since we don't want users injecting special tokens, this is not treated as such. Instead, it is tokenized as a standard piece of text.*

![Assistant conversation with special tokens](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/f6e47935-e5b5-4769-83d5-a3fa1a370e26/image.png)

*An assistant conversation. Special tokens are used to delimit system message, chat roles and messages (IM = internal message). The models are fine-tuned to handle conversations using these special tokens. These special tokens are not shown to the user.*

You may want to add new special tokens to your model, e.g. when taking a pretrained model and doing assistant fine-tuning. In this case, you need to expand the embedding layer and the model head as explained above (the rest of the transformer can stay identical).

### Tokenization woes

Karpathy [has spoken at length](https://www.youtube.com/watch?v=zduSFxRajkE) about the issues created by tokenization.

![Tokenization issues](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/20816e02-d09d-4f05-b6ee-564a5d4f8ec2/image.png)

### Codebases

![Tokenizer codebases](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/a2826fa0-472c-4be6-9240-8a5852b0d656/image.png)

Here's a quick overview of techniques used to make tokenization efficient:

![Tokenization efficiency techniques](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/76347fb1-0952-4f7d-b23f-44d111fdc29d/image.png)
