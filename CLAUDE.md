# LLM Wiki Schema for Gaming Tutorials, Gameplays, Patches, and Walkthroughs

## Overview
This schema defines how the LLM maintains a wiki for gaming content. The wiki consists of three layers:
1. **Raw Sources**: Immutable collection of source documents (articles, papers, images, data files).
2. **Wiki**: LLM-generated markdown files (summaries, entity pages, concept pages, comparisons, overview, synthesis).
3. **Schema**: This document (CLAUDE.md) that tells the LLM how to structure the wiki and maintain consistency.

## Directory Structure
- `raw_sources/` : Place source materials here (e.g., game manuals, patch notes, gameplay videos transcripts, tutorial texts).
- `wiki/` : LLM-generated markdown files. The LLM owns this directory entirely.
  - `wiki/entities/` : Pages for specific games, characters, items, etc.
  - `wiki/concepts/` : Pages for game mechanics, genres, terminology.
  - `wiki/comparisons/` : Pages comparing games, versions, strategies.
  - `wiki/overviews/` : High-level overviews (e.g., "Overview of Game X", "Overview of Patch Y").
  - `wiki/synthesis/` : Synthesis pages that combine multiple sources (e.g., "Complete Walkthrough of Game Z").
  - `wiki/index.md` : Main index page linking to all other pages.

## Naming Conventions
- Use lowercase with hyphens for file names (e.g., `game-overview.md`, `patch-notes-1.2.md`).
- Entity pages: `entities/<game-name>-<entity-type>.md` (e.g., `entities/zelda-breath-of-the-wild-character-link.md`).
- Concept pages: `concepts/<concept-name>.md` (e.g., `concepts/open-world-exploration.md`).
- Comparison pages: `comparisons/<item1>-vs-<item2>.md` (e.g., `comparisons/zelda-ocarina-of-time-vs-majoras-mask.md`).
- Overview pages: `overviews/<topic>-overview.md` (e.g., `overviews/gameplay-overview.md`).
- Synthesis pages: `synthesis/<topic>-synthesis.md` (e.g., `synthesis/complete-walkthrough-final-fantasy-vii.md`).

## Workflow for Ingesting New Sources
1. When a new source is added to `raw_sources/`, the LLM should:
   a. Read the source to understand its content.
   b. Determine which wiki pages need to be created or updated.
   c. Update existing wiki pages to reflect new information, maintaining consistency.
   d. Create new wiki pages for new entities, concepts, or topics not covered.
   e. Update cross-references between pages.
   f. Update the index (`wiki/index.md`) if necessary.

## Workflow for Answering Questions
1. When asked a question about gaming tutorials, gameplays, patches, or walkthroughs:
   a. First, consult the wiki (`wiki/`) for relevant information.
   b. If the wiki does not have sufficient information, consult the raw sources.
   c. Synthesize an answer based on the wiki and raw sources, citing sources when possible.
   d. If the answer reveals gaps in the wiki, note them for future updates.

## Maintenance
- The LLM should periodically review the wiki for consistency, outdated information, and broken links.
- When multiple sources conflict, the LLM should note the discrepancy and present both viewpoints if appropriate.
- The LLM should not modify raw sources; they are immutable.

## Example Pages
- `wiki/entities/mario-character-mario.md`: Information about Mario from the Mario series.
- `wiki/concepts/power-up.md`: Explanation of power-up mechanics in games.
- `wiki/comparisons/super-mario-bros-3-vs-super-mario-world.md`: Comparison of two Mario games.
- `wiki/overviews/mario-series-overview.md`: Overview of the Mario series.
- `wiki/synthesis/super-mario-bros-3-walkthrough.md`: Step-by-step walkthrough of Super Mario Bros 3.

## Revision History
- 2026-06-02: Initial schema created.