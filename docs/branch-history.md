# Branch history

The final snapshot was prepared on 21 September 2026 from the current research workspace. It starts a new `main` history so the public default checkout contains the curated final code, reports and manuscript without the old binary experiment archive.

| Reference | Role | Preserved commit |
|:---|:---|:---|
| `dev` | Original self-play research; unchanged | `531fe7c0466002e718a1e4b9da0c42caead90a0f` |
| `archive/main-before-final-2026-09-21` | Tag preserving the former `main` | `ed97b7a685169bb0d1b9f35c2a04ce86f2ec8815` |

The former `main` and `dev` were not identical. Both are preserved at the references above. The local research workspace and large artifact archive were left intact; the publication snapshot was assembled separately.

Earlier implementations in `src/wdn/` and experimental drivers provide research context. The final result record is `results/final/final_results.json`; historical results are not substituted for it.
