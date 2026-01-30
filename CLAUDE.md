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
