# One Layer Deeper Researcher Track

This track is the owner-training system for the project. Its goal is not to
turn the owner into the fastest PyTorch programmer. Its goal is to make the
owner capable of choosing the question, recognizing a confounded experiment,
and directing a mediocre coding agent toward trustworthy results.

## Three versions

1. [`01-self-contained-handbook.md`](01-self-contained-handbook.md): the whole
   project taught from first principles.
2. [`02-annotated-syllabus.md`](02-annotated-syllabus.md): primary papers,
   lectures, repository evidence, and what to extract from each.
3. [`03-mastery-practicum.md`](03-mastery-practicum.md): exercises, agent
   management, experiment operations, tests, and promotion gates.

## Learning-mode rule

Until the owner completes the Foundation Gate in the practicum, the project is
in learning mode:

- finish monitoring already-submitted jobs;
- preserve results and operational safety;
- allow CPU learning exercises and read-only audits;
- do not rent a GPU, start a model sweep, or submit competition quota;
- do not let an agent reinterpret “learning exercise” as authorization to run
  the real experiment.

The owner can explicitly override this rule, but an override should name the
new evidence or deadline that justifies it.

