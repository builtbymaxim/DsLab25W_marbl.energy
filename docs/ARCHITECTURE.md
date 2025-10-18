# Architektur-Skeleton

Schichten
- core: Domänenmodelle, Config, Typen
- io: Datenzugriff (API, CSV, Caching)
- processing: Cleaning, Harmonisierung, Qualitätsmetriken
- features: Ereignis-/Merkmalslogik (Spikes, Saisonalität)
- pipelines: Orchestrierung pro Task/Report
- viz: Standardplots
- utils: Hilfsfunktionen

Cli
- `marbl` (Typer) mit Kommandos für ingest, quality, spikes, eda
