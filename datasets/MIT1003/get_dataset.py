import pysaliency

stimuli, fixations = pysaliency.external_datasets.get_mit1003_with_initial_fixation(
    location="output", replace_initial_invalid_fixations=True
)
