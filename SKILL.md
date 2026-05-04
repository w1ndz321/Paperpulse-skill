---
name: paperpulse-skill
description: Read local CS/AI/LLM academic PDFs, crop important PDF screenshots, and produce a Chinese paper brief as `report.md`, `images/`, and GitHub Pages-ready `report.html`.
---

# PaperPulse Skill

Turn one local CS / AI / LLM paper PDF into a shareable Chinese paper brief:

```text
PDF -> source_text.md + images/ + captions.json -> report.md -> report.html
```

Final folder, created under this skill's `outputs/` directory by default:

```text
outputs/<paper-slug>/
├── report.md
├── report.html
└── images/
```

`<paper-slug>` must come from keywords in the paper title, not from opaque PDF filenames such as `2510.04618v3.pdf`. `pdf_process.py` does this by default. Use `--slug` only to override with a clearer title-keyword folder name.

Use relative image paths like `images/main_results.png` so the folder works locally and on GitHub Pages.

## Fast Path

If `report.md` already exists and all referenced `images/...` files exist, do not re-read the PDF or re-crop screenshots. Run only:

```bash
python scripts/render_report_html.py "<output-dir>/report.md" "<output-dir>/report.html"
```

HTML rendering should be seconds or less. Multi-minute runtime belongs to PDF extraction, screenshot selection/cropping, or report writing.

## Workflow

1. Prepare the paper assets in one script:

```bash
python scripts/pdf_process.py "<pdf-path>"
```

This creates:

```text
outputs/<paper-slug>/
├── source_text.md
├── captions.json
└── images/
```

`pdf_process.py` directly uses `pymupdf4llm.to_markdown()`, removes `References` / `Bibliography` / `参考文献` and everything after it, detects arXiv / code links from PDF annotations and first-page text, extracts正文中的 Figure/Table screenshots into `images/`, and writes their captions/nearby text to `captions.json`. If extraction fails, report the dependency/error; do not use OCR or fallback extraction.

Do not search the internet to fill missing paper/code links. If `captions.json` does not contain a paper or code/project link, write `未在 PDF 中提取到` in `report.md` and continue.

Use `--output-root <dir>` only when the user explicitly wants to put all output folders somewhere else. The default is the skill-local `outputs/` folder because it is writable in Codex.

If the generated folder name still looks like an arXiv id or a meaningless filename, rerun with a title-keyword slug, for example:

```bash
python scripts/pdf_process.py "<pdf-path>" --slug ace-agentic-context-engineering
```

2. Read `source_text.md` and `captions.json`. Do not read every image visually. Use the paper body plus each figure/table caption and nearby text to decide which images belong in the brief. Select by article logic, not by fixed keywords. Image count should follow paper length and evidence density:

- short papers: usually 3 to 5 evidence images
- regular papers: usually 4 to 7 evidence images
- long papers, surveys, benchmark papers, or system papers: usually 7 to 12 evidence images, and more if every image carries a distinct argument

- choose problem/task/motivation figures when they shape the story
- choose method/framework/pipeline/algorithm/objective figures when they explain the contribution
- choose main result tables or benchmark figures when they prove the claim
- choose ablation, scaling, cost, latency, case-study, error, safety, or limitation figures only when they change the interpretation

Screenshot quality is a hard requirement:

- Every report screenshot must include the complete figure/table body.
- Preserve in-figure legends, color keys, axis labels, table headers, notes, and the paper's original Figure/Table caption.
- Do not crop so tightly that a title, legend, footnote, or caption is cut off. Prefer a slightly larger crop over a cleaner but incomplete crop.
- Visually inspect only the images selected for the final report. If any image is incomplete, re-crop it before writing or rendering the report.
- When the original caption is too long, keep the full original caption in the screenshot and use a shorter Chinese alt/caption in Markdown.

If a selected screenshot is incomplete, manually re-crop or re-run PDF processing before writing or rendering the report. Use only final `images/...` paths in `report.md`.

3. Write `report.md` in Chinese following `references/reportstyle.md`. Load that file before drafting the report.

4. Render HTML:

```bash
python scripts/render_report_html.py "<output-dir>/report.md" "<output-dir>/report.html"
```

After rendering, check the HTML hero information cards:

- `PAPER` should contain arXiv / DOI / paper links only.
- `CODE` should contain GitHub / GitLab / Hugging Face / project/demo links only.
- `AUTHORS` should contain authors, institutions, or author team text only; it must not contain URLs or keywords.
- `KEYWORDS` should contain about 5 keywords only; it must not contain author names or links.

If a card is missing expected content or contains the wrong kind of text, fix the metadata near the top of `report.md` and rerender. The template also has automatic cleanup, but the source Markdown should still be correct.

Keep `source_text.md` and `captions.json` because they explain how the report was built. In the final response, state output paths and whether selected screenshots were visually inspected.

## Scripts

- `scripts/pdf_process.py`: prepare `source_text.md`, `images/`, and `captions.json`.
- `scripts/render_report_html.py`: render `report.md` into PaperPulse-style `report.html`.
