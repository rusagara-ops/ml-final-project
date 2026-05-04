# RouteRight labeling rules (Yale-focused)

These six labels are the **only** valid `label` values in `data/raw/collected_questions.csv` and in `data/processed/final_dataset.csv`. When scraping or copying FAQ text, assign **one** primary office per row. If a question truly spans two offices, pick the office that would **first** triage the case, or skip the row and note it for error analysis.

Allowed labels (exact spelling):

| `label` | Use this label when the question is primarily about… |
|-----------|------------------------------------------------------|
| `Registrar` | Academic records, transcripts (official/unofficial), enrollment verification, registration deadlines, add/drop, course withdrawal, major/degree declarations, transfer credit, diplomas, exam schedules tied to **records/registration policy** (not course content tutoring). |
| `Financial_Aid` | FAFSA/CSS Profile, aid offers, scholarships, grants, loans, work-study, financial aid verification, appeals, **aid eligibility**—not routine tuition billing unless the text is clearly about **aid posting** vs a bursar bill (if billing-only with no aid context, prefer Registrar/bursar policy if you add that later; for this 6-class task, billing tied to aid packages stays here). |
| `Housing` | Residence halls, room assignment/draw, roommate changes, move-in/out, dorm policies, graduate/undergraduate housing portals, **mail keys in residence**—not off-campus landlord disputes (skip or label best-effort as Housing only if about Yale housing contract). |
| `IT` | NetID/password, Duo/MFA, email, Wi‑Fi, VPN, software licensed by the university, phishing reports, lab printing, **Canvas/LMS access** and learning-tech login issues, device network registration. |
| `Dining` | Meal plans, dining halls/retail, menus, dietary stations, guest meals, dining points/dollars, Yale Hospitality policies—**not** student health nutrition therapy (that is `Health_Wellness`). |
| `Health_Wellness` | Yale Health / student health, insurance and waivers, immunizations, counseling and mental health, accessibility/disability services **as health/campus care** (if purely academic accommodations with no clinical angle, still often `Health_Wellness` when it is the accessibility office; if unclear, use the Yale page section you took the text from as the guide). |

## Yale vs supplemental examples

- **Primary:** `school` = `Yale` and `url` should point to a **Yale** `.yale.edu` (or official Yale subdomain such as `yalehealth.yale.edu`) page you used as evidence for the label.
- **Supplemental (allowed):** If Yale-only pages yield too few examples, you may add questions from **other universities’ public FAQ pages**. Those rows **must** have `school` set to something other than `Yale` (e.g. the institution’s short name) and `url` set to the **non-Yale** page. In the final report, summarize how many supplemental rows were used and from which domains.

## Quality checks before merging

1. **No PII:** paraphrase or use only published FAQ lines; do not paste ticket numbers or student IDs.
2. **One label per row;** no multi-line concatenations of unrelated FAQs.
3. **Language:** English only for this project unless the team extends the scope in writing.
4. **Duplicates:** exact duplicate questions (after text cleaning) are removed by `src/clean_dataset.py`; near-duplicates may remain—dedupe manually if needed.

## Automated crawl rows

`src/collect_yale_faqs.py` can append machine-extracted lines from Yale HTML pages. Those rows still use the six labels, but **heuristic extraction is imperfect** (navigation blurbs, long policy paragraphs, or duplicate headings). Before analysis and grading, **spot-check** each category and delete or relabel obvious mistakes; keep the `url` column truthful for anything you retain.

## Source metadata

Every row in `collected_questions.csv` should set:

- `source_name`: short human-readable name of the page or section (e.g. “Yale Registrar — Registration FAQs”).
- `school`: `Yale` or the non-Yale institution for supplemental rows.
- `url`: canonical URL of the page where the text appeared (or the closest parent FAQ page).

Track high-level sources in `data/raw/faq_sources.csv` for the report’s Data section.
