// Build the dissertation progress-draft .docx from the markdown chapters, in ATU style.
// Front matter (title pages, declaration, acknowledgements, abstract), an updatable TOC,
// then Chapters 1 to 3 with the detection figures embedded.
//
//   node build_dissertation.js
//
// Output: dissertation/Dissertation_Shpyl_progress_draft.docx

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel,
  LevelFormat, TableOfContents, Header, Footer, PageNumber, PageBreak, BorderStyle,
  Table, TableRow, TableCell, WidthType, ShadingType, VerticalAlign, ExternalHyperlink,
} = require("docx");

const REPO = path.resolve(__dirname, "..", "..");
const CH = path.join(REPO, "dissertation", "chapters");
const FIGS = path.join(REPO, "dissertation", "figures");
// Default output path; override with DISS_OUT (useful when the .docx is open/locked in Word).
const OUT = process.env.DISS_OUT
  ? path.resolve(process.env.DISS_OUT)
  : path.join(REPO, "dissertation", "Dissertation_Shpyl_progress_draft.docx");
const figdims = JSON.parse(fs.readFileSync(path.join(__dirname, "figdims.json"), "utf8"));

// Match the official 2026 template: Calibri body, Office-blue headings, 12pt 1.5 justified.
const ARIAL = "Calibri";       // body font (template theme minor font)
const TEAL = "2F5496";         // heading colour (Word default Office blue)
const INK = "222831";

// ---- repository linking ----
// Every file this document names in backticks becomes a live link into the public repository, so an
// examiner reading a claim can open the code that produced it in one click.
//
// The one rule that matters here: a dead link is worse than no link. Some paths the text mentions
// are deliberately not published (the corpus itself, model checkpoints, the human-essay guide), so
// the candidate is checked against `git ls-files` before it becomes a link. Anything not actually
// in the repository is left as plain monospaced text, exactly as it read before.
const REPO_URL = "https://github.com/MichaelShpyl/AI_reserch";
const REPO_REF = "main";
const REPO_TOP = /^(src|tests|config|data|outputs|models|dissertation|notebooks)(\/|$)/;

const TRACKED = new Set();   // every published file path
const TRACKED_DIRS = new Set();  // every directory that contains at least one published file
try {
  const listed = require("child_process")
    .execSync("git ls-files", { cwd: REPO, encoding: "utf8", maxBuffer: 8 << 20 })
    .split("\n").map(s => s.trim()).filter(Boolean);
  for (const f of listed) {
    TRACKED.add(f);
    const parts = f.split("/");
    for (let i = 1; i < parts.length; i++) TRACKED_DIRS.add(parts.slice(0, i).join("/"));
  }
} catch (e) {
  console.warn("git ls-files failed, so no path will be linked:", e.message);
}

const LINKED = new Map();    // path -> times linked, for the build summary
const UNLINKED = new Set();  // repo-shaped paths that are not published, for the build summary

function repoLink(pathText) {
  const clean = pathText.trim().replace(/[.,;:)]+$/, "").replace(/\/$/, "");
  if (!REPO_TOP.test(clean) || clean.includes(" ")) return null;
  const isFile = TRACKED.has(clean);
  const isDir = TRACKED_DIRS.has(clean);
  if (!isFile && !isDir) { UNLINKED.add(clean); return null; }
  LINKED.set(clean, (LINKED.get(clean) || 0) + 1);
  return `${REPO_URL}/${isFile ? "blob" : "tree"}/${REPO_REF}/${clean}`;
}

// ---- inline parsing: **bold** and `code`, with repo paths becoming hyperlinks ----
function inlineRuns(text, base = {}) {
  const runs = [];
  let i = 0, buf = "", bold = false, code = false;
  // Consolas runs wide, and a long unbreakable path in a justified paragraph stretches the line it
  // sits on. Setting code a point smaller than the body narrows the paths without making them hard
  // to read, and keeps the justification from opening up rivers of white space.
  const mk = (t, isCode, link) => new TextRun({
    text: t, bold: bold || base.bold, italics: base.italics,
    font: isCode ? "Consolas" : ARIAL,
    size: isCode ? Math.round((base.size || 24) * 0.9) : (base.size || 24),
    color: link ? "1F4E5F" : (base.color || INK),
    underline: link ? {} : undefined,
  });
  const flush = () => {
    if (!buf) { buf = ""; return; }
    const url = code ? repoLink(buf) : null;
    if (url) runs.push(new ExternalHyperlink({ link: url, children: [mk(buf, true, true)] }));
    else runs.push(mk(buf, code, false));
    buf = "";
  };
  while (i < text.length) {
    if (text.startsWith("**", i)) { flush(); bold = !bold; i += 2; }
    else if (text[i] === "`") { flush(); code = !code; i += 1; }
    else { buf += text[i]; i += 1; }
  }
  flush();
  return runs.length ? runs : [new TextRun({ text: "", font: ARIAL, size: base.size || 24 })];
}

function imageParagraphs(altAndPath) {
  // altAndPath: { caption, file }
  const dim = figdims[altAndPath.file];
  const aspect = dim.h / dim.w;
  let wIn = aspect > 0.85 ? 3.8 : aspect > 0.6 ? 4.7 : 5.7;
  const px = Math.round(wIn * 96);
  const img = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(FIGS, altAndPath.file + ".png")),
      transformation: { width: px, height: Math.round(px * aspect) },
      altText: { title: altAndPath.file, description: altAndPath.caption, name: altAndPath.file },
    })],
  });
  // The caption already begins "Figure N.M: ...", so split the label off for the front-matter list.
  const m = altAndPath.caption.match(/^(Figure\s+[0-9]+\.[0-9]+):\s*(.+)$/s);
  if (m) CAPTIONS.figures.push([m[1], m[2]]);
  const cap = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: inlineRuns(altAndPath.caption, { italics: true, size: 20, color: "52616B" }),
  });
  return [img, cap];
}

// ---- markdown chapter -> docx paragraphs ----
// The chapter markdown is hard-wrapped, so consecutive non-blank text lines belong to one
// logical paragraph (or list item) and must be joined until a blank line or a structural
// element (heading, bullet, image) ends it.
// Captions are numbered across the whole document, in reading order, and collected here so the
// front-matter lists are generated rather than hand-maintained. A hand-kept list silently went
// three figures stale, so nothing in the front matter is typed by hand any more.
const CAPTIONS = { figures: [], tables: [], listings: [] };

const capBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const capBorders = { top: capBorder, bottom: capBorder, left: capBorder, right: capBorder };

function captionParagraph(label, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 200 },
    children: [t(label + "  ", { bold: true, size: 20, color: "52616B" }),
      ...inlineRuns(text, { italics: true, size: 20, color: "52616B" })],
  });
}

// A markdown pipe table becomes a real Word table with a shaded header row and a numbered caption.
function tableParagraphs(rows, caption) {
  const widths = rows[0].map(() => Math.floor(9026 / rows[0].length));
  const cell = (txt, head) => new TableCell({
    borders: capBorders, width: { size: widths[0], type: WidthType.DXA },
    shading: head ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { after: 0 }, children: inlineRuns(txt, { size: 20, bold: head }) })],
  });
  const table = new Table({
    width: { size: 9026, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0,
      children: r.map(c => cell(c, i === 0)) })),
  });
  const n = CAPTIONS.tables.length + 1;
  const label = "Table " + n;
  CAPTIONS.tables.push([label, caption]);
  // Table captions sit above the table; figure captions sit below the figure.
  return [captionParagraph(label, caption), table, new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// A fenced code block becomes monospaced, unjustified, line-preserving text with a caption.
function codeParagraphs(lines, caption) {
  const n = CAPTIONS.listings.length + 1;
  const label = "Code Listing " + n;
  CAPTIONS.listings.push([label, caption]);
  const paras = lines.map(l => new Paragraph({
    spacing: { after: 0, line: 240 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: l.length ? l : " ", font: "Consolas", size: 18, color: INK })],
  }));
  return [...paras, captionParagraph(label, caption)];
}

function parseChapter(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let buf = "", mode = null;  // mode: 'para' | 'bullet' | 'num'
  let tableRows = null, codeLines = null;

  const flush = () => {
    const text = buf.trim();
    buf = "";
    if (!text) { mode = null; return; }
    if (mode === "bullet") {
      // Match body spacing (1.5 lines) so lists do not collapse: supervisor feedback, Meeting 4.
      out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
        spacing: { after: 100, line: 360 }, children: inlineRuns(text) }));
    } else if (mode === "num") {
      out.push(new Paragraph({ numbering: { reference: "nums", level: 0 },
        spacing: { after: 100, line: 360 }, children: inlineRuns(text) }));
    } else {
      out.push(new Paragraph({ spacing: { after: 160, line: 360 },
        alignment: AlignmentType.JUSTIFIED, children: inlineRuns(text) }));
    }
    mode = null;
  };

  // Caption for the next table or listing, set by a "Table:" / "Listing:" line in the markdown.
  let pendingCaption = null;

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");

    // Fenced code blocks come first: blank lines and indentation inside them must survive.
    if (/^```/.test(line.trim())) {
      if (codeLines === null) { flush(); codeLines = []; }
      else {
        out.push(...codeParagraphs(codeLines, pendingCaption || "Code listing"));
        codeLines = null; pendingCaption = null;
      }
      continue;
    }
    if (codeLines !== null) { codeLines.push(raw.replace(/\t/g, "    ")); continue; }

    // Pipe tables: collect contiguous rows, drop the |---|---| separator.
    if (/^\s*\|.*\|\s*$/.test(line)) {
      flush();
      const cells = line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
      if (!cells.every(c => /^:?-{2,}:?$/.test(c))) {
        (tableRows = tableRows || []).push(cells);
      }
      continue;
    }
    if (tableRows) {
      out.push(...tableParagraphs(tableRows, pendingCaption || "Table"));
      tableRows = null; pendingCaption = null;
    }

    const capLine = line.match(/^(?:Table|Listing):\s*(.+)$/);
    if (capLine) { flush(); pendingCaption = capLine[1].trim(); continue; }

    if (line.trim() === "") { flush(); continue; }
    if (line.startsWith(">")) { flush(); continue; }   // drop draft-note banners
    const img = line.match(/^!\[(.+?)\]\((.+?)\)$/);
    if (img) {
      flush();
      const file = path.basename(img[2]).replace(/\.png$/, "");
      out.push(...imageParagraphs({ caption: img[1], file }));
      continue;
    }
    if (line.startsWith("### ")) {
      flush();
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 },
        children: inlineRuns(line.slice(4), { bold: true, size: 24, color: INK }) }));
    } else if (line.startsWith("## ")) {
      flush();
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 },
        children: inlineRuns(line.slice(3), { bold: true, size: 28, color: TEAL }) }));
    } else if (line.startsWith("# ")) {
      flush();
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
        spacing: { before: 240, after: 200 },
        children: inlineRuns(line.slice(2), { bold: true, size: 36, color: TEAL }) }));
    } else if (/^[-*] /.test(line)) {
      flush(); mode = "bullet"; buf = line.slice(2);
    } else if (/^\d+\.\s/.test(line)) {
      flush(); mode = "num"; buf = line.replace(/^\d+\.\s/, "");
    } else {
      // plain text: start a paragraph or continue the current one (joins wrapped lines)
      if (mode === null) mode = "para";
      buf += (buf ? " " : "") + line.trim();
    }
  }
  if (tableRows) out.push(...tableParagraphs(tableRows, pendingCaption || "Table"));
  if (codeLines) out.push(...codeParagraphs(codeLines, pendingCaption || "Code listing"));
  flush();
  return out;
}

function readChapter(name) {
  return parseChapter(fs.readFileSync(path.join(CH, name), "utf8"));
}

// ---- front matter helpers ----
const center = (children, opts = {}) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: opts.spacing || { after: 200 }, children });
const t = (text, o = {}) => new TextRun({ text, font: ARIAL, size: o.size || 24,
  bold: o.bold, italics: o.italics, color: o.color || INK });
const body = (text) => new Paragraph({ alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 160, line: 360 }, children: inlineRuns(text) });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
  spacing: { before: 240, after: 200 }, children: [t(text, { bold: true, size: 36, color: TEAL })] });

const logo = new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new ImageRun({ type: "jpg", data: fs.readFileSync(path.join(__dirname, "atu_logo.jpg")),
    transformation: { width: 320, height: 98 },
    altText: { title: "ATU", description: "Atlantic Technological University", name: "ATU" } })] });

const TITLE = "An Explainable AI Pipeline for Academic Integrity Verification";
const SUBTITLE = "Transparent AI-Text Detection with Argument-Aware Question Generation";
const AWARD = "Master of Science in Artificial Intelligence and Big Data Analytics";

const titlePage1 = [
  center([t(TITLE, { bold: true, size: 36, color: INK })], { spacing: { before: 600, after: 200 } }),
  center([t(SUBTITLE, { italics: true, size: 28, color: "52616B" })], { spacing: { after: 500 } }),
  center([t("Mykhailo Shpyl", { size: 28 })]),
  center([t("M.Sc. in Artificial Intelligence and Big Data Analytics, 2026", { size: 24 })], { spacing: { after: 400 } }),
  logo,
  center([t("Department of Computing, ATU Donegal, Port Road, Letterkenny, Co. Donegal, Ireland.", { size: 22, color: "52616B" })], { spacing: { after: 300 } }),
  center([t("August 2026", { size: 22, color: "52616B" })]),
];

const titlePage2 = [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 },
    children: [t(TITLE, { bold: true, size: 32, color: INK })] }),
  center([t(SUBTITLE, { italics: true, size: 26, color: "52616B" })], { spacing: { after: 500 } }),
  center([t("Author: Mykhailo Shpyl", { size: 26 })]),
  center([t("Supervised by: Dr. Vini Vijayan", { size: 26 })], { spacing: { after: 400 } }),
  center([t("A thesis submitted in partial fulfilment of the requirements for the", { size: 24 })]),
  center([t(AWARD, { bold: true, size: 24 })], { spacing: { after: 300 } }),
  center([t("Submitted to Atlantic Technological University", { size: 24 })]),
  center([t("Arna chur isteach chuig Ollscoil Teicneolaiochta an Atlantaigh", { italics: true, size: 22, color: "52616B" })]),
  center([t("August 2026", { size: 24 })]),
];

const declaration = [
  h1("Declaration"),
  body("I hereby certify that the material, which I now submit for assessment on the programmes of study leading to the award of " + AWARD + ", is entirely my own work and has not been taken from the work of others except to the extent that such work has been cited and acknowledged within the text of my own work. No portion of the work contained in this thesis has been submitted in support of an application for another degree or qualification to this or any other institution. I understand that it is my responsibility to ensure that I have adhered to ATU's rules and regulations."),
  body("I hereby certify that the material on which I have relied on for the purpose of my assessment is not deemed as personal data under the GDPR Regulations. Personal data is any data from living people that can be identified. Any personal data used for the purpose of my assessment has been pseudonymised and the data set and identifiers are not held by ATU. Alternatively, personal data has been anonymised in line with the Data Protection Commissioner's Guidelines on Anonymisation. All datasets used in this work are publicly available and openly licensed, and no human participants were involved in the study."),
  body("I consent that my work will be held for the purposes of education assistance to future students and will be shared on the ATU Donegal (Computing) website (atucomputingdonegal.com) and Research THEA website (https://research.thea.ie/). I understand that documents once uploaded onto the website can be viewed throughout the world and not just in Ireland. Consent can be withdrawn for the publishing of material online by emailing Jade Lyons, Head of Department, at Jade.Lyons@atu.ie to remove items from the ATU Donegal Computing website, and by emailing Denise McCaul, Systems Librarian, at denise.mccaul@atu.ie to remove items from the Research THEA website. Material will continue to appear in printed formats once published, and as websites are a public medium, ATU cannot guarantee that the material has not been saved or downloaded."),
  new Paragraph({ spacing: { before: 400, after: 80 },
    children: [t("Signature of Candidate: Mykhailo Shpyl", { size: 24 }),
      t("\t\tDate: __________________", { size: 24 })] }),
];

const acknowledgements = [
  h1("Acknowledgements"),
  body("I would like to thank my supervisor, Dr. Vini Vijayan, for her guidance and feedback throughout this project, and the staff of the MSc in Artificial Intelligence and Big Data Analytics at Atlantic Technological University for their support."),
];

// ---- Acronyms table ----
const ACRONYMS = [
  ["AI", "Artificial Intelligence"],
  ["NLP", "Natural Language Processing"],
  ["LLM", "Large Language Model"],
  ["GenAI", "Generative Artificial Intelligence"],
  ["BAWE", "British Academic Written English (corpus)"],
  ["TTR", "Type-Token Ratio"],
  ["POS", "Part of Speech"],
  ["BIO", "Begin-Inside-Outside (sequence labelling scheme)"],
  ["QLoRA", "Quantised Low-Rank Adaptation"],
  ["SHAP", "SHapley Additive exPlanations"],
  ["IG", "Integrated Gradients"],
  ["FPR", "False-Positive Rate"],
  ["HPC", "High-Performance Computing"],
];
const cellBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
function acroCell(text, opts = {}) {
  return new TableCell({
    borders: cellBorders, width: { size: opts.w, type: WidthType.DXA },
    shading: opts.head ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    children: [new Paragraph({ children: [t(text, { bold: opts.head, size: 22 })] })],
  });
}
const acronymsTable = new Table({
  width: { size: 9026, type: WidthType.DXA }, columnWidths: [2200, 6826],
  rows: [
    new TableRow({ tableHeader: true, children: [acroCell("Acronym", { w: 2200, head: true }), acroCell("Definition", { w: 6826, head: true })] }),
    ...ACRONYMS.map(([a, d]) => new TableRow({ children: [acroCell(a, { w: 2200 }), acroCell(d, { w: 6826 })] })),
  ],
});
const acronyms = [h1("Acronyms"), acronymsTable, new Paragraph({ spacing: { after: 100 }, children: [] })];

// ---- Front-matter lists, generated from what the chapters actually contain ----
// Built after the chapters are parsed, so these can never drift from the document again.
function captionList(heading, entries, emptyNote) {
  const out = [h1(heading)];
  if (!entries.length) {
    out.push(new Paragraph({ spacing: { after: 120 },
      children: [t(emptyNote, { italics: true, size: 22, color: "52616B" })] }));
    return out;
  }
  for (const [n, c] of entries) {
    const short = c.length > 110 ? c.slice(0, 107).replace(/[\s,;.]+$/, "") + "..." : c;
    out.push(new Paragraph({ spacing: { after: 80 },
      children: [t(n + "   ", { bold: true, size: 22 }), ...inlineRuns(short, { size: 22 })] }));
  }
  return out;
}

const abstract = [
  h1("Abstract"),
  body("Universities increasingly use automatic detectors to judge whether student work was written with generative AI. Most return one percentage with nothing behind it: a lecturer cannot defend it if challenged, and it says nothing about whether the student understands what they submitted. This dissertation builds a pipeline that detects likely AI text, explains the decision in plain language, extracts the submission's own claims, and writes verification questions from them, so understanding is checked in conversation rather than by accusation. It runs on one laptop, so student work never leaves the institution."),
  body("The detector was trained on 640 essays from the British Academic Written English corpus paired with 640 topic-matched and length-matched AI essays. A first version scored perfectly; an audit traced that to formatting markup, and the honest F1 after cleaning is 0.99. It catches every test essay from two unseen generators without falsely flagging their human counterparts, but on unfamiliar human writing false positives rise sharply, so fusing it with stylometric features cuts them three to eight times. Token-level attribution fails a faithfulness test that SHAP passes."),
  body("Question quality is measured without a judge: a question works if a model that has read the essay answers it far better than one that has not. Across thirty essays and 901 questions, a QLoRA fine-tune of a 3B model on one 8 GB laptop beat the free commercial tier on 24 of 29 shared essays (p = 0.0003). Three of this project's own headline numbers were retracted by its own checks."),
];

// The chapter files, in document order. Named once so the contents list and the document body
// cannot fall out of step with each other.
const CHAPTER_FILES = [
  "01_introduction.md", "02_literature_review.md", "03_detection.md", "04_implementation.md",
  "05_explainability.md", "06_robustness.md", "07_question_generation.md",
  "08_evaluation_questions.md", "09_discussion.md", "10_conclusions.md",
  "11_references.md", "12_appendices.md",
];

// ---- Table of contents ----
// A Word TableOfContents field only fills in when someone opens the file and presses F9. Nobody
// presses F9 on a PDF, so the submitted document showed a contents page containing the instruction
// to generate a contents page. This builds a real one instead: the entries are read from the same
// chapter headings the document is built from, and the page numbers come from a first pass over
// the rendered PDF (docgen/toc_pages.json, written by build_toc_pages.py).
//
// The two passes stay in step because the entry text is identical in both and the page number sits
// on a right-aligned tab at the margin, so a number changing width cannot re-wrap a line and move
// the pagination underneath it.
function tocEntries() {
  // The front-matter sections are Heading 1 in the document, so a Word contents field
  // would list them. Listing them here keeps the literal version faithful to that.
  const out = ["Declaration", "Acknowledgements", "Abstract", "Acronyms",
               "Table of Figures", "Table of Tables", "Table of Code Listings"]
    .map((text) => ({ level: 1, text }));
  for (const f of CHAPTER_FILES) {
    const src = fs.readFileSync(path.join(CH, f), "utf8");
    for (const line of src.split(/\r?\n/)) {
      if (line.startsWith("## ")) out.push({ level: 2, text: line.slice(3).trim() });
      else if (line.startsWith("# ")) out.push({ level: 1, text: line.slice(2).trim() });
    }
  }
  return out;
}

let TOC_PAGES = {};
try {
  TOC_PAGES = JSON.parse(fs.readFileSync(path.join(__dirname, "toc_pages.json"), "utf8"));
} catch { /* first pass: no page numbers yet */ }

const TOC_TAB = 9000;   // twips, just inside the right margin
function tocLine(e) {
  const page = TOC_PAGES[e.text];
  return new Paragraph({
    spacing: { after: e.level === 1 ? 60 : 20, before: e.level === 1 ? 120 : 0 },
    indent: { left: e.level === 1 ? 0 : 340 },
    tabStops: [{ type: "right", position: TOC_TAB, leader: "dot" }],
    children: [
      t(e.text, { bold: e.level === 1, size: e.level === 1 ? 24 : 22,
                  color: e.level === 1 ? INK : "3A4750" }),
      new TextRun({ text: "\t" + (page === undefined ? "0" : String(page)),
                    font: ARIAL, size: e.level === 1 ? 24 : 22,
                    bold: e.level === 1, color: e.level === 1 ? INK : "3A4750" }),
    ],
  });
}

const toc = [h1("Table of Contents"), ...tocEntries().map(tocLine)];

const ch1 = readChapter("01_introduction.md");
const ch2 = readChapter("02_literature_review.md");
const ch3 = readChapter("03_detection.md");
const ch4 = readChapter("04_implementation.md");
const ch5 = readChapter("05_explainability.md");
const ch6 = readChapter("06_robustness.md");
const ch7 = readChapter("07_question_generation.md");
const ch8 = readChapter("08_evaluation_questions.md");
const ch9 = readChapter("09_discussion.md");
const ch10 = readChapter("10_conclusions.md");
const ch11 = readChapter("11_references.md");
const ch12 = readChapter("12_appendices.md");

const doc = new Document({
  creator: "Mykhailo Shpyl",
  title: TITLE,
  styles: {
    default: { document: { run: { font: ARIAL, size: 24, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: ARIAL, color: TEAL },
        paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: ARIAL, color: TEAL },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: ARIAL, color: INK },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: { size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ font: ARIAL, size: 20, color: "52616B", children: [PageNumber.CURRENT] })] })] }),
    },
    children: [
      ...titlePage1, ...titlePage2, ...declaration, ...acknowledgements, ...abstract,
      ...acronyms, ...toc,
      ...captionList("Table of Figures", CAPTIONS.figures, "No figures."),
      ...captionList("Table of Tables", CAPTIONS.tables, "No tables."),
      ...captionList("Table of Code Listings", CAPTIONS.listings, "No code listings."),
      ...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...ch9, ...ch10, ...ch11, ...ch12,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Saved", OUT);
  let total = 0;
  for (const n of LINKED.values()) total += n;
  console.log(`Repository links: ${total} across ${LINKED.size} distinct paths.`);
  if (UNLINKED.size) {
    // Not an error. These are paths the text names that are deliberately unpublished (corpus,
    // checkpoints) or typos. Printed so a real typo does not sit unnoticed as plain text.
    console.log(`Not linked (${UNLINKED.size}, left as plain text): ${[...UNLINKED].sort().join(", ")}`);
  }
});
