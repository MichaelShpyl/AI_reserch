# Dissertation document generator

Builds `dissertation/Dissertation_Shpyl_progress_draft.docx` from the markdown chapters in
`dissertation/chapters/`, in ATU thesis style (title pages, declaration, acknowledgements,
abstract, an updatable table of contents, then the chapters with the detection figures
embedded).

## Rebuild

```bash
cd dissertation/docgen
npm install        # first time only; installs the docx library (node_modules is gitignored)
node build_dissertation.js
```

The output `.docx` is written to `dissertation/` and is gitignored (it is a generated binary;
the source of truth is the markdown chapters plus this script).

## Notes

- `figdims.json` holds the pixel dimensions of the embedded figures so the script can keep
  their aspect ratios. Regenerate it if the figures change (see the snippet in the project
  progress log entry for 16 June 2026).
- `atu_logo.jpg` is the ATU logo used on the title page.
- The table of contents is a Word field. Open the document in Word, select all, and press F9
  to populate the page numbers.
- This is a working-draft generator. The chapter prose is to be rewritten in the author's own
  words before final submission, per the project writing rules.

## Checking the document against the results

Two audits, both run from the repository root.

```bash
python dissertation/docgen/audit_consistency.py
python dissertation/docgen/audit_numbers.py --deck
```

`audit_consistency.py` checks a named list of headline figures, the citation set and its
alphabetical order, and the figure numbering and cross-references. It is precise about the things
it knows about.

`audit_numbers.py` is the opposite. It pulls every number out of the chapters and, with `--deck`,
the slides, pulls every value out of `outputs/`, and lists the numbers the text asserts that no
results file supports. It cannot know what a number means, so the output is a list to read rather
than a list of defects: sample sizes, values quoted from other people's papers and figures derived
by addition all show up and are all fine. Known constants live in `CONSTANTS` at the top of the
file, with a note saying what each one is, so the list stays short enough to actually read.

The point of the second one is that the first only checks what somebody remembered to add to it. A
summary claim about the hybrid detector was wrong in three places for weeks and no hand-kept list
would ever have caught it.
