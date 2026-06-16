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
