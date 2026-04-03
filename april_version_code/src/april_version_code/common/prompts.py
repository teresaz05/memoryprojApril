"""Human-readable experiment descriptions used by the package README and wrapper help text.

The heavy prompt templates still live in the copied experiment cores so the behavior matches
the original runners. This module is only for the concise descriptions a collaborator needs
when choosing which script to run.
"""

EXPERIMENT_DESCRIPTIONS = {
    'prose_merge2': 'Per-document prose cluster-bank summaries plus merge2 answering.',
    'structured_merge2': 'Per-document structured cluster-bank summaries plus merge2 answering.',
    'docsummaryaux_merge2': 'Query-aware doc summaries plus cluster-bank merge2 answering.',
    'rlm_promptdocs_from_docsummaryaux': (
        'Official RLM with raw support documents kept in prompt context and one extra '
        'docsummaryaux companion document appended per source document.'
    ),
}
