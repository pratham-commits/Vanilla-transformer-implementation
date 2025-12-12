Question 1: "If we break the 512 vector into 64-chunks, doesn't that break the meaning?"
Short Answer: It would... IF we just chopped the raw input vector. But we don't. We use a Linear Projection first.

The Misconception: You are imagining the Input Vector (for "Love") like a book with chapters:

Indices 0-63: "Chapter 1: The Spelling"

Indices 64-127: "Chapter 2: The Meaning"

...

If this were true, slicing it would indeed give Head 1 only "The Spelling" and Head 2 only "The Meaning."

The Reality (The Prism Analogy): Look at the code in src/attention.py again:

Python

# 1. Linear Projection (The Prism)
query = self.w_q(q) 
# 2. THEN split
query = query.view(..., heads, d_k)
Think of the input vector (512) as a beam of White Light. It contains everything mixed together. The Linear Layer (self.w_q) acts like a Prism.

It takes that white light and refracts it.

It learns to sort the information.

It pushes all the "Grammar" info into the first 64 numbers.

It pushes all the "Gender" info into the next 64 numbers.

It pushes all the "Tense" info into the next 64 numbers.

So, when we split it into heads:

Head 1 receives a complete packet of Grammar info.

Head 2 receives a complete packet of Gender info.

Head 3 receives a complete packet of Tense info.

The 64-sized vector isn't a "broken fragment" of the word; it is a specialized view of the word.

Question 2: "How do we ensure each head learns something different?"
You might worry: "What if all 8 heads just decide to focus on Grammar? Then we wasted 7 heads!"

We do not write any code to force them to be different. We rely on two powerful forces:

1. Random Initialization (The "Nature" factor)
When we created the model:

Python

self.w_q = nn.Linear(d_model, d_model)
PyTorch filled this matrix with Random Numbers.

Head 1 starts with a random bias towards looking at the previous word.

Head 2 starts with a random bias towards looking at the next word.

Head 3 starts with a random bias towards looking at verbs.

Because they start in different places on the "map," they tend to walk down different paths.

2. Gradient Descent (The "Nurture" factor)
This is the most important part. Imagine a soccer team where all 8 players run to the ball. The team will lose. The coach (Loss Function) will yell at them.

To win (lower the loss), the players realize: "I need to cover the goal, and you need to cover the wing."

The Math of Efficiency: If Head 1 and Head 2 are doing the exact same thing, they are redundant. The model is effectively "wasting" capacity. The Gradient Descent process is greedy. It will realize: "Hey, Head 1 has Grammar covered perfectly. Head 2, you are useless right now. Change your weights to look for synonyms instead."

The Research Reality: Interestingly, researchers have analyzed what the heads learn.

Early Layers (1-2): Heads usually focus on position (looking at the previous word).

Middle Layers (3-4): Heads focus on syntax (Subject-Verb relationships).

Deep Layers (5-6): Heads focus on semantic meaning (Context, tone, global structure).

So, they naturally specialize because that is the most efficient way to solve the translation puzzle.