# AI Notes - Learning Style Guide

## Goal

Build a deep and wide understanding of AI. Not surface-level explanations, but 100% comprehension of how things actually work.

## Teaching Style: The Karpathy Method

My primary reference is Andrej Karpathy's approach to explaining AI concepts. Here's what makes it effective:

1. **State the goal first** - Say what we want to accomplish before diving in
2. **Start with intuition** - Build understanding from first principles, not formulas
3. **Simplest possible solution** - Begin with the most basic version that could work
4. **Show what breaks** - Demonstrate why the simple solution fails or has limitations
5. **Add layers incrementally** - Each new concept builds on the previous foundation
6. **Always show the code** - Build the thing to understand it. Theory alone is not enough.

## What I Want From Every AI Topic

- **Intuition first**: Explain the "why" and the general idea before any details
- **Go deep**: Don't stop at the surface. I want to understand the internals.
- **Show the code**: Always include implementation. I understand by building.
- **100% comprehension**: If something is unclear, break it down further rather than hand-waving

## No Jargon Without Explanation

**Every technical term must be explained when first used.** Don't assume the reader knows what a "convolution" is, what a "CLS token" is, or why we use "positional embeddings." If a term appears, explain it. If it's a concept from another domain (like convolutions from CNNs), give enough context that someone unfamiliar can follow.

**No hand-waving.** Phrases like "we prepend a CLS token and add positional embeddings" are useless without explaining:
- What is a CLS token? (A learnable vector that...)
- Where exactly does it go? (Position 0 of the sequence...)
- Why do we need it? (To aggregate information for...)
- What are positional embeddings? (Vectors that encode position because...)
- How are they added? (Element-wise addition to each token...)

**Concrete steps, not vague summaries.** When describing a process, list the actual operations:
1. Take input X
2. Apply operation Y, producing Z
3. The loss is computed as...
4. Gradients flow back through...

**Don't be afraid to write more.** Thoroughness beats brevity. If explaining something properly takes 500 words instead of 50, write the 500 words. These are learning materials, not tweets.

## Writing Style: Mini-Textbooks, Not Reference Notes

These notes are how I learn, not quick references. Write them like mini-textbooks:

**Discursive paragraphs, not bullet points.** Explain things in flowing prose. Use paragraphs that walk through ideas, not lists of facts. The reader should feel like they're being taught, not scanning documentation.

**Code is purposeful, not decorative.** Code is essential, but it has a reading cost. Every code block should earn its place. Surround code with explanation—what problem does it solve? What would you see if you ran it? Trace through concrete examples with actual numbers.

**Visual explanations over walls of code.** When explaining a concept like "ViT splits images into patches," show it:

```
Original image (224×224):          After patching (14×14 patches of 16×16):
┌────────────────────────┐         ┌──┬──┬──┬──┬──┬──┬──┐
│                        │         │1 │2 │3 │4 │5 │6 │7 │ ... 14
│     [photo of cat]     │   →     ├──┼──┼──┼──┼──┼──┼──┤
│                        │         │15│16│17│  │  │  │  │
│                        │         ├──┼──┼──┼──┼──┼──┼──┤
└────────────────────────┘         │  │  │  │  │  │  │  │ ... 196 patches total
```

ASCII diagrams, concrete examples—these often communicate better than another code block.

**Use proper markdown tables, not ASCII tables.** For tabular data, use markdown table syntax:

```
| Column A | Column B | Column C |
|----------|----------|----------|
| value 1  | value 2  | value 3  |
```

ASCII box-drawing tables (with `┌`, `│`, `└` characters) often render poorly depending on fonts and display contexts. Markdown tables are more reliable.

**Generated diagrams with Python.** For more complex visuals (architecture diagrams, data flow illustrations, plots), generate them with matplotlib or PIL and save to the `assets/` folder. Embed in notes with `![[assets/filename.png]]`. This gives precise control over technical diagrams and doesn't require external image generation services.

**Conversational tone.** "You might think X, but actually Y." "Why doesn't this cause problems? Two reasons." "That's it." The writing should feel like a knowledgeable friend explaining something, not a textbook being formal.

**No empty phrases.** Eliminate AI slop patterns—sentences that sound important but contain no information:
- "Here's the key insight:"
- "This is important to understand:"
- "Let's break this down:"
- "It's worth noting that..."
- "Interestingly enough..."

These are filler. If something is a key insight, just state it. The insight itself should make its importance obvious. Delete the throat-clearing and get to the point.

**No dialogue artifacts.** Notes are standalone documents, not transcripts. Don't include phrases that reference a conversation:
- "So you're exactly right:"
- "To your specific question:"
- "As I mentioned earlier..."
- "Great question!"

The reader has no context for who "you" is or what was said earlier. Write as if the reader is encountering the material fresh.

## Editing Process

**Reread for global coherence after edits.** After making changes to a note, reread the entire document to ensure it still flows as a coherent whole. Incremental edits can create Frankenstein notes—redundant sections, inconsistent examples, or awkward transitions. If something feels off, fix it. If unsure whether a fix is right, ask.

## Mathematical Formulas

I'm not fluent in mathematical notation. When formulas are necessary:

1. **Always include them** - They're part of the complete picture
2. **Break them down** - Explain each symbol and what it represents
3. **Translate to Python** - Show the formula as actual code. This is how I truly understand it.

Example of what I need:

```
Formula: L = -E[log P(y|x)]

Breakdown:
- L is the loss we want to minimize
- E[...] means "expected value" or "average over all examples"
- log P(y|x) is the log probability the model assigns to the correct answer y given input x
- The negative sign flips it so minimizing loss = maximizing probability

In Python:
log_probs = model.log_prob(y, given=x)  # shape: (batch_size,)
loss = -log_probs.mean()  # average over batch, negate
```

This combination of notation + explanation + code is how formulas should always be presented.
