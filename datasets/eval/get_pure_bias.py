"""
Get subset of BBQ that isn't in bbq_subset_100.jsonl
Let's say 100 on each section as well.

Then, want to fill in purposefully-biased completions -- can grab these
from the metadata so it won't be hard.

Can also test prompting model with biased completion, and asking to
create a biased explanation --> then use this as the fine-tuning 
dataset.
"""