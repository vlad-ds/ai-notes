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

ASCII diagrams, tables for comparison, concrete examples—these often communicate better than another code block.

**Generated diagrams with Python.** For more complex visuals (architecture diagrams, data flow illustrations, plots), generate them with matplotlib or PIL and save to the `assets/` folder. Embed in notes with `![[assets/filename.png]]`. This gives precise control over technical diagrams and doesn't require external image generation services.

**Conversational tone.** "You might think X, but actually Y." "Why doesn't this cause problems? Two reasons." "That's it." The writing should feel like a knowledgeable friend explaining something, not a textbook being formal.

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
