# Inpaint/collage workflow checkpoint

- Backend is currently using replace-background-and-relight endpoint for subject+background generation.
- Prompt blueprint, palette, and analysis logic are all working and up to date.
- User wants to try a two-step process: generate background with core model, then composite subject on top for sharper, more collage-like results.
- If needed, revert to this state by restoring backend/app.py and prompt logic as of this checkpoint.
- No changes to frontend or analysis logic yet.

Date: 2026-05-30
