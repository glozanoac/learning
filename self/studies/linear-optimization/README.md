# Linear Optimization

Analysis of the book:

```bibtex
@book{bertsimas1997introduction,
  title     = {Introduction to Linear Optimization},
  author    = {Bertsimas, Dimitris and Tsitsiklis, John N.},
  year      = {1997},
  publisher = {Athena Scientific},
  address   = {Belmont, MA},
  series    = {Athena Scientific Series in Optimization and Neural Computation},
  volume    = {6}
}
```

Aligned to the objetive and due that many of the decision variables in open-pit mining are discrete, we are going to focus on "discrete linear optimization", a reading plan was defined at [planning](./planning/)

## Structure

| Path | Contents |
|---|---|
| `planning/` | Reading plan for discrete linear optimization: reading paths and the open-pit mining problems they lead to |
| `N*-*.rnote` | Handwritten notes, numbered by topic: LP models, optimality of extreme points, simplex method |
| `NNN-*.tex/.pdf` | Typeset proofs (e.g. Theorem 2.3: extreme point implies basic feasible solution) |
| `extra/chapterN/` | Complementary examples and numerical checks of a chapter, named `NNN__<result>__<description>` |
| `src/` | Python prototypes (simplex method) |

[<- Back to studies](../README.md)
