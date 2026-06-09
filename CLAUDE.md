# LLM Wiki Schema for AmazonIA Travel

## Overview

This schema defines how the LLM maintains a vertical tourism wiki for Amazonas travel planning. The wiki has three layers:

1. **Raw Sources**: Source notes, public references, transcripts, guides, and curated travel material.
2. **Wiki**: Markdown pages generated or maintained by the LLM for RAG retrieval.
3. **Schema**: This document, which defines structure, naming, workflow, and quality rules.

## Directory Structure

- `raw_sources/`: Source materials for tourism in Amazonas. Keep original source notes here.
- `wiki/`: Curated Markdown knowledge base used by the RAG agent.
  - `wiki/entities/`: Destinations, attractions, operators, travel profiles, and experience types.
  - `wiki/concepts/`: Planning concepts such as seasonality, logistics, safety, responsible tourism, and budget framing.
  - `wiki/comparisons/`: Comparisons between destinations, seasons, route options, or travel profiles.
  - `wiki/overviews/`: High-level domain overviews.
  - `wiki/synthesis/`: Itineraries, decision guides, and synthesized travel playbooks.
  - `wiki/index.md`: Main index linking to the most important pages.

## Naming Conventions

- Use lowercase with hyphens for file names.
- Entity pages: `entities/<amazonas-topic>.md`, such as `entities/amazonas-destinations.md`.
- Concept pages: `concepts/<planning-topic>.md`, such as `concepts/amazonas-travel-planning.md`.
- Comparison pages: `comparisons/<item1>-vs-<item2>.md`.
- Overview pages: `overviews/<topic>-overview.md`.
- Synthesis pages: `synthesis/<topic>.md`, such as `synthesis/amazonas-itineraries.md`.

## Workflow for Ingesting New Sources

1. Read the source and identify its tourism scope.
2. Decide whether it updates destinations, logistics, seasonality, safety, culture, sustainability, or itineraries.
3. Update existing wiki pages before creating duplicates.
4. Create new pages only when the topic deserves independent retrieval.
5. Add cross-links and update `wiki/index.md` when a page becomes important.
6. Mark dynamic information, such as prices and schedules, as requiring current verification.

## Workflow for Answering Questions

1. Consult `wiki/` first.
2. If the answer is incomplete, consult `raw_sources/` if relevant.
3. For current prices, schedules, availability, regulations, weather alerts, road conditions, or event calendars, use current sources or clearly state that verification is required.
4. Synthesize practical guidance in Brazilian Portuguese.
5. Ask for missing traveler constraints when they materially change the answer, such as dates, number of days, budget, comfort level, mobility, and interests.

## Quality Rules

- Do not invent prices, schedules, legal requirements, medical guidance, or operator availability.
- Distinguish stable travel knowledge from information that changes frequently.
- Prefer responsible tourism: local benefit, environmental care, respect for communities, consent for photos, and safe operators.
- When information conflicts, present the uncertainty and recommend verification.

## Revision History

- 2026-06-04: Converted the project schema from gaming content to Amazonas tourism.
