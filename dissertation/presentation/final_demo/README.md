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

There is a hands-free mode for this, so the recording contains no cursor wandering and no tab
clicking. It plays the lecturer session first, then the film, then a closing card, and stops.

1. Open `index.html` in a browser and press **F11** for fullscreen. The page follows the system
   light or dark theme, so set that first if you have a preference.
2. Start a screen recording. On Windows the Xbox Game Bar is enough: **Win+Alt+R**.
3. Press **P**. Everything plays by itself. Do not touch the mouse or keyboard after this.
4. When the closing card appears ("End of demonstration"), stop the recording.

Total runtime is roughly five and a half minutes: about two minutes of the lecturer session and
three and a half of the film.

Three ways to start it, in case one is awkward on the day:

- press **P** on the page (most reliable, works however the file was opened)
- open `index.html?record=1`
- open `index.html#record`

The query string and hash are stripped by some embedded viewers, which is why the keypress exists.

### If you would rather drive it yourself

Skip record mode and use the tabs. A good live route: open **Watch a lecturer use it** and let it
run, then switch to **Explore it yourself** and click a claim to show the provenance highlighting,
then **Play the story** if there is time.

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
