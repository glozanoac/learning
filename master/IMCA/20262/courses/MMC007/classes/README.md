# MMC007 – Optimization: Class Notes

Lecture notes for **MMC007 – Optimización**, Maestría en Modelamiento Matemático y Computacional, IMCA-UNI (2026-II).
Main references: Nocedal & Wright, *Numerical Optimization* (2nd ed.) **[NW]**, and Bertsekas, *Nonlinear Programming* (3rd ed.) **[B]**.

## Structure

```
classes/
├── syllabus.pdf        Course syllabus (16 weeks)
├── ch*.pdf   Lesson notes, one file per chapter
├── extra/              Complementary proofs (see below)
└── exercises/          Exercises
```

## Chapters

| File | Topic | Reference |
|------|-------|-----------|
| `ch1.pdf` | Introduction, convex sets and functions | NW 1–2, B 1.1 |
| `ch2-v2.pdf` | Unconstrained optimality conditions, line search | NW 2–3, B 1.2 |
| `ch3.pdf` | Newton's method, Hessian modification, quasi-Newton (DFP, BFGS) | NW 3, 6 |
| `ch3_13-15.pdf` | Revision of pp. 13–15 of Ch. 3 (global convergence proof, Powell) | NW 6.5 |
| `ch4.pdf` | Constrained optimality: tangent cone, constraint qualifications, Farkas, KKT, Lagrange duality | NW 12, B 3, 5 |
| `ch5.pdf`, `ch5-v2.pdf` | Trust-region methods: exact solution, Cauchy point, dogleg, global convergence | NW 4 |
| `ch6.pdf` | Large-scale methods (conjugate gradient) | NW 5, 7 |

`-v2` and later revisions replace the original file; `ch5-v2.pdf` adds the motivation section.

## `extra/`

Complementary resources (partial or complete proofs) to understand a specific hypothesis or step of the lessons. There is one subfolder per chapter (`extra/chN/`), added as needed; browse the folder of the chapter you are studying.

Files are named `NNN__<result>__<description>`: `NNN` is the order of appearance in the chapter, `<result>` is the statement being proved (e.g. `prop-2.1`, `Theorem-3.1`), and `PAGEn` marks the page of the notes where the step is used.
