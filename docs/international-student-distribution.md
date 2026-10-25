# International-student distribution and registry selection

## What the statistics say

The registry expansion is guided by country-level international-student distribution data and the availability of official doctoral/admissions sources. These measures are not directly comparable in every country: some count foreign citizenship, others count prior country of education or internationally mobile students, and national reporting years differ.

Eurostat's 2024 learning-mobility statistics report 1.83 million students from abroad undertaking tertiary study in the EU. Germany hosted 438,300 (24.0% of the EU total), France 284,900 (15.6%), and the Netherlands 176,500 (9.7%, excluding doctoral students because those data are incomplete). Across the EU, 43% of students from abroad came from Europe, 26% from Asia, and 17% from Africa. At doctoral level, students from abroad represented 22.2% of EU students overall, illustrating that international mobility is especially relevant to research-degree discovery.

These figures describe destinations and student populations, not the international-student enrollment of each named university. They guide geographic coverage and institution selection; this project does not add enrollment counts or percentages to its university records or generated output.

## Data sources

| Coverage | Source | Notes |
| --- | --- | --- |
| EU, including Austria, Belgium, Finland, France, Germany, Italy, the Netherlands, Spain, and Sweden | [Eurostat: Learning mobility statistics](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Learning_mobility_statistics) | 2024 statistics; dataset `educ_uoe_mobs02`. Definitions and missing-data notes are in the source. |
| United States | [IIE Open Doors data](https://opendoorsdata.org/data/international-students/) | U.S. Department of State-sponsored annual survey of international students and scholars. |
| United Kingdom | [HESA student statistics](https://www.hesa.ac.uk/data-and-analysis/students) | Official higher-education student statistics, including student domicile. |
| Canada | [Statistics Canada: Postsecondary enrolments](https://www150.statcan.gc.ca/n1/en/subjects/education_training_and_learning/postsecondary_education) | National enrolment statistics; consult table notes for citizenship/residency definitions and reporting year. |
| Australia | [Australian Government international education data](https://www.education.gov.au/international-education-data-and-research) | Official international-student enrolment and sector data. |
| Germany | [DAAD: Wissenschaft weltoffen](https://www.wissenschaft-weltoffen.de/en/) | Annual indicators on international students and researchers in Germany. |
| Japan | [JASSO international student statistics](https://www.jasso.go.jp/en/statistics/) | Official survey statistics; use the corresponding year and survey definitions. |
| India | [AISHE reports](https://aishe.gov.in/aishe-final-report/) | Government higher-education statistics; check the report year and foreign-student definition. |
| Switzerland | [Swiss Federal Statistical Office: education](https://www.bfs.admin.ch/bfs/en/home/statistics/education-science.html) | Higher-education statistics by citizenship and institution. |
| South Korea | [Study in Korea](https://www.studyinkorea.go.kr/en/main.do) | Government information portal; no country total is included in the comparison above. |
| Malaysia | [Education Malaysia Global Services](https://educationmalaysia.gov.my/) | Official international-education portal; no country total is included in the comparison above. |

## Validated registry coverage

The approved expansion goal was approximately 10 additional universities in each of the 10 existing countries and 10–20 universities in each of 10 new countries. That goal is **not yet met**: many candidate URLs were unavailable or blocked, and blocked pages were not bypassed. The following are the current registry totals after adding only sources that survived the full ingestion pipeline:

| Country | Sources |
| --- | ---: |
| Australia | 15 |
| Austria | 5 |
| Belgium | 6 |
| Canada | 18 |
| China | 16 |
| Finland | 2 |
| France | 15 |
| Germany | 13 |
| India | 16 |
| Italy | 4 |
| Japan | 15 |
| Malaysia | 4 |
| Netherlands | 11 |
| Norway | 4 |
| South Korea | 1 |
| Spain | 3 |
| Sweden | 4 |
| Switzerland | 7 |
| United Kingdom | 19 |
| United States | 16 |
| **Total** | **194** |

Of the 194 original proposed additions, 93 produced at least one chunk in the multi-source ingestion-and-FAISS build. One initially fetchable source (KAIST) failed normal TLS certificate verification and was excluded rather than weakening verification. A separate official-page check found Örebro University, which also passed a full single-source build. The registry therefore adds 94 validated sources, yielding 559 chunks across the isolated builds. Other candidates were excluded after fetch failures or unusable text. Counts above represent validated sources, not the requested per-country target.

## Selection and source-validation method

- Select research universities using international-student destination evidence as context, alongside research prominence and geographic coverage; do not treat enrollment volume as a quality ranking.
- Keep the application registry format unchanged: `country`, `university`, and official `url`.
- Add only official, public graduate, doctoral, or admissions pages that return usable text with the project's existing fetcher and produce one or more chunks in a full ingestion run. The 94 additions produced 559 chunks in isolated full-build runs.
- Do not bypass access controls, anti-bot responses, or other restrictions. Replace a blocked or unavailable URL with another official page, or leave that candidate out.
- University pages and enrollment statistics change. Re-check the official source before consequential use; this document's Eurostat comparison refers to 2024 data.
