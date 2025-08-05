# Relation Info
This file (`rel_info_full.json`) includes the data augmentations that are used in the experiments.
The format is an extension of the rel_info.json file from the original DocRED data.
The name and description were pulled from WikiData pages on those relations.
`prompt_xy` is the prompt used in the experiments, and `prompt_yx` is so far unused.
XY and YX refer to the "direction" of the text, and whether the subject of the relation appears first (XY) or second (YX) in the sentence.
The fields `domain` and `range` are empty; they get populated during runtime by a different process.
There are several additional fields with logicl restrictions (`reflexive`, `symmetric`, etc.).
These have been partially filled in, but should be considered incorrect and are unused.
The `tokens` field is based on tokenization using `bert-based-uncased`, and as such isn't directly used in the paper.
It is intended to allow for easily identifying which tokens belong to the prompt and which belong to entities.
This was used in a discarded experiment which did not find a good way to use entity-only/prompt-only scores to correlate to support.
`verb` is a pointer to the token in that list that acts as the "primary" verb in the sentence, and was used in a discarded experiment that did not show a relationship between verb scores and support.

