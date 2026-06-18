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
  Table, TableRow, TableCell, WidthType, ShadingType, VerticalAlign,
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

// ---- inline parsing: **bold** and `code` ----
function inlineRuns(text, base = {}) {
  const runs = [];
  let i = 0, buf = "", bold = false, code = false;
  const flush = () => {
    if (buf) runs.push(new TextRun({ text: buf, bold: bold || base.bold,
      italics: base.italics, font: code ? "Consolas" : ARIAL,
      size: base.size || 24, color: base.color || INK }));
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
function parseChapter(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let buf = "", mode = null;  // mode: 'para' | 'bullet' | 'num'

  const flush = () => {
    const text = buf.trim();
    buf = "";
    if (!text) { mode = null; return; }
    if (mode === "bullet") {
      out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
        spacing: { after: 80 }, children: inlineRuns(text) }));
    } else if (mode === "num") {
      out.push(new Paragraph({ numbering: { reference: "nums", level: 0 },
        spacing: { after: 80 }, children: inlineRuns(text) }));
    } else {
      out.push(new Paragraph({ spacing: { after: 160, line: 360 },
        alignment: AlignmentType.JUSTIFIED, children: inlineRuns(text) }));
    }
    mode = null;
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
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
  center([t("Working draft for supervisor review, June 2026", { italics: true, size: 22, color: "52616B" })]),
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

// ---- Table of Figures (manual list; the detector chapter figures) ----
const FIGURE_LIST = [
  ["Figure 3.1", "What each detection signal achieves on its own"],
  ["Figure 3.2", "The words the cleaned-text model keys on (style, not topic)"],
  ["Figure 3.3", "DeBERTa on the held-out test set after markup removal"],
  ["Figure 3.4", "Essays in function-word style space: two clusters"],
  ["Figure 3.5", "Detector F1 with 95% bootstrap confidence intervals"],
  ["Figure 5.1", "Integrated Gradients token attributions for a matched essay pair"],
  ["Figure 5.2", "Faithfulness by ablation: the signal is diffuse"],
  ["Figure 5.3", "SHAP on the stylometric detector (feature-level explanation)"],
  ["Figure 6.1", "Transfer to unseen generators on essays"],
  ["Figure 6.2", "In-domain vs cross-generator vs cross-domain F1"],
  ["Figure 6.3", "Cross-domain failure modes by domain"],
];
const tableOfFigures = [
  h1("Table of Figures"),
  ...FIGURE_LIST.map(([n, c]) => new Paragraph({ spacing: { after: 80 },
    children: [t(n + "   ", { bold: true, size: 22 }), t(c, { size: 22 })] })),
];

const abstract = [
  h1("Abstract"),
  body("Universities increasingly rely on automatic detectors to judge whether student work was written with generative AI. Most detectors return a single percentage with nothing behind it, which a lecturer cannot defend if a student challenges it, and which says nothing about whether the student understands the work they submitted. This dissertation develops an explainable pipeline for academic integrity verification. It detects likely AI-generated text, explains which features drove each decision, extracts the claims and evidence in a flagged submission, and generates verification questions tied to those claims so a lecturer can check understanding in a short conversation. The detector is the first component reported here. It was trained on a balanced corpus of 640 human essays from the British Academic Written English corpus and 640 length-matched and topic-matched AI essays generated locally with Llama 3.1. An initial model reached a perfect score, which was treated as a warning rather than a result. A reproducible audit found that the original corpus leaked formatting markup that gave the answer away, removed it from both classes, and confirmed on cleaned text that the two classes remain separable on writing style alone, with a function-words-only model still reaching 99.5 percent. On the cleaned corpus the transformer detector scores an F1 of 0.99. The high in-domain score is consistent with the literature on single-generator detection, and the work now turns to explainability and robustness on harder settings. This document is a working draft submitted for supervisor review; the prose will be revised in the author's own words for final submission."),
];

const toc = [
  h1("Table of Contents"),
  new Paragraph({ spacing: { after: 120 },
    children: [t("Update this field in Word (select all, then press F9) to populate page numbers.",
      { italics: true, size: 20, color: "52616B" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
];

// Chapter 2 status note (it is a planned structure, not finished prose).
const ch2note = new Paragraph({
  spacing: { before: 120, after: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: TEAL, space: 12 } },
  children: [t("This chapter is presented as a planned structure and reading plan. The section headings below show the intended coverage of the review. The prose and the verified 2021 to 2026 citations are in progress and will be completed once the corresponding components are built.",
    { italics: true, size: 22, color: "52616B" })],
});

const ch1 = readChapter("01_introduction.md");
let ch2 = readChapter("02_literature_review.md");
ch2 = [ch2[0], ch2note, ...ch2.slice(1)];  // insert the note right after the H1
const ch3 = readChapter("03_detection.md");
const ch4 = readChapter("04_implementation.md");
const ch5 = readChapter("05_explainability.md");
const ch6 = readChapter("06_robustness.md");

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
      ...acronyms, ...toc, ...tableOfFigures,
      ...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("Saved", OUT); });
