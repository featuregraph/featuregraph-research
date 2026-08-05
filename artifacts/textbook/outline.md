Representation: A First-Principles Approach to Algorithms
Part I — Representation
Chapter 1 — Computation is Representation

The central idea:

Every computation begins by choosing a representation.

Examples:

numbers
strings
trees
graphs
time series
images

Question:

What information should survive?

Chapter 2 — State

What is a state?

Examples:

position in an array

(node)

(i,j)

(board, constraints)

(environment state)

State is

everything necessary to continue the computation.

Chapter 3 — Graphs

Every computation induces a graph.

Examples:

Tree recursion

DP lattice

Control flow

FeatureGraph pipeline

Neural network

The graph tells us

where information may flow.

Part II — Information
Chapter 4 — Contracts

Probably my favorite chapter.

Definition:

Every state promises to return one object.

Examples:

subtree size

subtree height

list of visible nodes

(sum,count)

best reward

This chapter is where recursion finally becomes obvious.

Chapter 5 — Information Flow

Edges transport information.

Nodes transform information.

Exactly what your notebook pages have become.

Chapter 6 — Local Contributions

Every node asks

What do I contribute?

Examples

+1

append(root)

reward

current cost

current letter
Chapter 7 — Aggregation Operators

This is the chapter that grew out of our climbing-stairs discussion.

Same graph.

Different contracts.

Different operators.

+

or

max

min

union

intersection

concatenate

Rule:

The contract determines the operator.

Part III — Families of Computation

Notice these are no longer presented as isolated techniques.

Chapter 8 — Recursive Aggregation

Trees.

Subtree summaries.

Bottom-up information flow.

Chapter 9 — Recursive Construction

Gray Code.

Catalan objects.

Grammar generation.

Objects building larger objects.

Chapter 10 — Search

Backtracking.

Constraint satisfaction.

Decision trees.

Here information flows downward.

Chapter 11 — Dynamic Programming

Memoized recursion.

Tables.

Value propagation.

Bellman's Principle.

DP becomes

recursion with remembered states.

Chapter 12 — Graph Algorithms

DFS

BFS

Shortest paths

Topological order

All described as information moving through graphs.

Part IV — Representation Design

This is where I think things become unusual.

Chapter 13 — Canonical Representations

Anagrams

Hashing

Normalization

Equivalence classes

Chapter 14 — Multiple Views

Sudoku

Indexes

Databases

Caches

One object.

Many representations.

Chapter 15 — Constraint Systems

Sudoku

Type inference

Logic

SAT

Constraint propagation.

Chapter 16 — Information Compression

Feature extraction.

Summary statistics.

Minimal sufficient representations.

This is where FeatureGraph naturally appears.

Part V — Learning Systems

Everything comes together.

Chapter 17 — Markov Decision Processes

State

Transition

Reward

Policy

Value

Chapter 18 — Reinforcement Learning

Bellman equations.

Dynamic programming.

Approximation.

Chapter 19 — Neural Networks

Representations.

Latent spaces.

Embeddings.

Graph computation.

Chapter 20 — Building New Representations

This chapter is almost philosophical.

The question becomes

How do we invent a useful computational representation?

Appendix

Instead of LeetCode categories

the appendix becomes

Aggregation

Transformation

Construction

Propagation

Canonicalization

Constraint

Composition

Search

Optimization

