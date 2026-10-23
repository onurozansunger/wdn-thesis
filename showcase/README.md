# Thesis and defence archive

Completed thesis by Onur Ozan Sünger. This branch keeps a small presentation archive after the local research workspace was retired on 23 October 2026.

- [Final thesis PDF](Thesis_Final.pdf)
- [Print version](Thesis_Print.pdf)
- [Defence slides — PDF](Defence.pdf) · [editable PowerPoint](Defence.pptx)
- [Final LaTeX sources](thesis-source/)
- [Final result summary](results/Final_Results.csv) · [detailed summary JSON](results/final_results.json)
- [Last local code snapshot](local-code-snapshot.zip); the curated research code also remains at the repository root.

The defence files are the final local versions modified on 20 October 2026. The final thesis PDF is byte-identical to the recorded final build. All 66 local source inputs listed by that build matched their recorded hashes. The print PDF is preserved separately.

## Scope

This is a presentation archive. Raw datasets, fitted models, feature banks, prediction arrays, experiment workspaces, earlier drafts and temporary renders are intentionally omitted at the author's request. It is not a complete rerun environment. The code ZIP preserves the last local source files, including historical scripts; those scripts may depend on deleted datasets and runs.

The original `main` branch and the exact commits cited in the thesis are unchanged:

- `49036da8016097f2ebb627b3b58cea1a07432a47` — thesis repository snapshot
- `300f7274d1158a4e1929f7f97028ed54f975baec` — supplementary evidence
- `d50fdcc376d41efa20772af02e2fc9fafd4d206d` — factorial evidence

No links or content in the thesis were rewritten. See `preservation-manifest.json`, `code-manifest.json` and `SHA256SUMS` for file checksums.

To rebuild the text, use a LaTeX installation with the dependencies required by Sapthesis, then run `latexmk -pdf main.tex` inside `thesis-source/`. Figures and tables are already included; no model training is needed to compile the manuscript. The original template license is retained. The existing PDFs are the archived deliverables.
