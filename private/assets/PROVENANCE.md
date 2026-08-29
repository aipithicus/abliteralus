# Assets

Externally sourced reference data used to build stimulus corpora. The files are
tracked, so the derivation chain is complete in-repo: raw source here, stores in
`experiments/studies/datasets/`, and nothing in between depends on a fetch that
might not reproduce. Record anything added so provenance survives a dead URL.

| File | Source | Retrieved | Notes |
| --- | --- | --- | --- |
| `brysbaert-concreteness.txt` | https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt | 2026-08-29 | Brysbaert, Warriner & Kuperman (2014), *Behavior Research Methods*. 39,955 rows, TSV. Columns include `Word`, `Conc.M`, `Conc.SD`, `SUBTLEX`, `Dom_Pos` — concreteness, frequency and dominant POS in one file. Third-party mirror; the Ghent CRR original 404s. Not yet unpacked. |
| `arxiv-taxonomy.html` | https://arxiv.org/category_taxonomy | 2026-08-29 | Category tree, 155 categories under archive groups. |
