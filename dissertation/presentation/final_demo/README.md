# Final demo

One self-contained page, `index.html`. Open it in any browser, no server needed. Everything on
screen is real output from the trained system: the two essays, the detection scores, the habit
measurements, the sentence shading, the claims and the questions all come from `data.js`, which
was extracted from the models and result files. Nothing is mocked.

## Two modes

- **Walkthrough** (default): the interactive demo to drive live. Switch between the AI-written
  version and the real student's essay. Click a phrase chip to light up every place the phrase
  appears in the document. Click a claim to see its questions and where in the document it came
  from. Hover a shaded sentence for the reason it is shaded.
- **Play the story**: an auto-playing film, eighteen scenes, about three and a quarter minutes.
  Written to be understood with no narration. Space pauses, arrow keys move, R restarts, the
  dots at the bottom left jump to a scene.

## Recording the video

1. Open `index.html`, press F11 for fullscreen, pick the theme you want (the page follows the
   system light or dark setting).
2. Start a screen recording (Xbox Game Bar is enough: Win+Alt+R).
3. Click "Play the story". It runs on its own and ends on the credit card.
4. Stop the recording after the closing scene.

For the walkthrough part of a recording, a good route is: AI version, dial and verdict, hover
two shaded sentences, click the "meanwhile" chip, scroll the rhythm skylines, open the first
claim, then switch to the real student's essay and let the contrast speak.

## Files

- `index.html`, the whole demo, inline CSS and JS.
- `data.js`, the extracted real data. Regenerate only if the models or results change.
- `demo_sent_meta.json`, intermediate output of the per-sentence occlusion run, kept for
  reference; `data.js` already contains what the page uses.

`data.js` and `demo_sent_meta.json` are deliberately not committed: they embed the text of a
BAWE essay, and BAWE-derived text never goes into git. They live only on this machine. If they
are ever lost, rebuild them from the pipeline: `explain_submission` for the card rows and
scores, the question generator for claims and questions, and the occlusion module
(`src/explainability/sentence_occlusion.py`) for the per-sentence drops, for essay 3108a and
its AI twin.
