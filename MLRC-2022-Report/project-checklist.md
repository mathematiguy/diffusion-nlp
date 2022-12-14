# Reproducibility Project
## Reproducibility Summary

-[X] Scope of reproducibility: State main claim of reproduced paper
-[X] Scope of reproducibility: Place work in context to tell reader the objective
-[x] Methodology: Describe what you did
-[] Results: Describe overall conclusions
-[] Results: Use specific and precise language
-[] Results: Used judgment to decide if results support original claim
-[] What was easy: Summarize which part of original paper are easy to apply
-[] What was difficult: Indicate which parts of original paper are difficult to re-use
-[] What was difficult: Indicate which parts of original paper requirement significant
-[] work/resources to verify
-[] Communication with original authors: Describe concisely contact with original authors

## Introduction
-[x] Introduce and motivate the problem and discuss its importance.
-[x] At a high-level, describe the contributions of the original paper and their significance.
-[x] In addition, at a high-level, describe the additional experiments or analysis that were performed.
-[x] Briefly describe the conclusions drawn.

## Scope of reproducibility
-[x] State the main claims of the paper they are reproducing.
-[x] State whether they are using the original codebase or not.
-[x] State, specifically, which results from the original paper they are replicating.

## Methodology
-[x] Describe the methods you used for reproducing the results of the original paper.
-[x] Describe the computational resources used to reproduce the results.
-[] Describe if the authors’ original code was used. If only parts of the authors’ code was used, describe which parts.
-[x] State the compute budget required for reproduction (e.g., how many GPU hours used).

## Model descriptions
-[x] Provide a brief description of each model and algorithm used in reproduction.
  -[x] Relevant citations should be included.
-[x] For each model used, state the following information:
  -[x] How large is it (i.e., how many parameters)?
  -[x] What data was the model trained on?
  -[x] Was the model pre-trained?
  -[x] Are there any major limitations?

## Datasets
-[xx] For each dataset used, include the dataset statistics:
  -[xx] How many examples?
  -[xx] Average example length (e.g., number of tokens).
  -[xx] Distribution of labels (in the case of a classification dataset).
-[xx] For each dataset used, state the train/validation/test split sizes. If the dataset does not have a training split, mention this.
-[xx] Include a link or reference for each dataset used.
-[xx] At a high-level, describe how each dataset was collected (e.g., was it crowdsourced?).

## Hyperparameters
-[x] Describe the settings of the hyperparameters used (e.g., we used a learning rate of 2e-5).
-[x] If a hyperparameter search was conducted, describe how it was done:
  -[x] How did you search the hyperparameter space? Linear search? Random search? Grid search?
  -[x] State how many repetitions (i.e., seeds) were used to select the best hyperparameter configuration.
  -[x] State what evaluation metric was used to select the best hyperparameter configuration (e.g., we determined the best hyperparameter configuration by computing the validation accuracy).

## Experimental setup and code
-[x] Describe how experiments were set up
-[x] Written clearly enough for readers to replicate
-[] Include description of evaluation measures
-[x] Provide link to code (see code requirements in section “Code”)

## Computational requirements
-[x] Description of hardware used (GPU, CPU, other)
-[x] For each model, include a measure of the average runtime
-[x] For each experiment, include the total computational requirements
-[x] Considered the perspective of a reader that wants to use the approach reproduced
-[x] List what a reader would find useful

## Results
-[o] High-level overview of your results
-[o] Discuss objectively if results support the main claims of the original paper
-[o] Section is factual and precise
-[o] Did not include judgment and discussion (which should be in the “Discussions” section)

## Results reproducing original paper
-[o] Had sufficient number of experiments with respect to the original paper
-[o] For each experiment:
  -[o] Indicate which claim in Section 2 it supports
  -[o] Indicate if it successfully reproduced the associated experiment in the original paper
-[o] Logically group related results into sections

## Results beyond original paper
-[o] Indicate whether the original paper failed to fully specify some experiment(s)
-[o] Indicate if additional experimentation was necessary
-[o] Include results of any additional experiments (only if it’s necessary for this reproduction)

## Discussion
-[o] Judge if experimental results support the claims of the paper
-[x] Discuss strengths of your approach
-[x] Discuss weaknesses of your approach
-[x] Indicate if you did not have time to run all experiments and why
-[x] Explain how additional experiments further strengthen the claims in the paper

## What was easy
-[x] Judge what was easy to reproduce
 -[x] Author’s code is clearly written
 -[x] Author’s code was easy to run
-[x] Avoid giving sweeping generalizations
-[x] Explain why something was easy (e.g. code had good API and lots of examples)

## What was difficult
-[x] List part of the study that:
 -[x] took more time than anticipated
 -[x] felt were difficult
-[x] Put discussion in context (e.g., avoid saying “math was difficult to follow”, instead “math require advanced knowledge of X to follow”)

## Communication with original authors

-[x] Documented the extend of communication with the original authors
 -[x] If they responded, summarize their responses
 -[x] If they did not, indicate which other means you used to reach out (email, social media, GitHub issues)
-[x] Either:
 -[x] List specific questions that were asked, or
 -[x] Sent full report to get their feedback

## Code
-[] Release usable code
-[] Avoid instructions that only work on specific platforms (e.g. SLURM)
-[] Provide self-contained instructions that are straightforward to follow
-[] Release code on GitHub (if it’s private, invite @xhluca or @ncmeade)
-[] Include instructions in Readme, or link to separate markdown file with full instructions
-[] Indicate what the TA should be reading
-[] Wrote the code in Python
-[] Released a PyPi package with instructions (if it’s not relevant, explain why in the GitHub readme, in a section called “Notes about release”)
 -[] For others to run the experiments
 -[] To import and reuse parts of the code as a standalone library
 -[] Can be installed via `pip` from the official PyPi

## Quality of Writing

-[] Clear and concise writing.
-[] Coherent sentences that follow a natural and logical flow
-[] Define important technical terms clearly such that any student in the class can understand
-[] If you have abbreviation, define it the first time you mention it
-[] Avoid repeating the same statement/idea multiple time when unnecessary
-[] Writing grammatically correct sentences
-[] Avoid typo/missing words and phrases that are difficult to read
-[] If you have text in different language (e.g. in a figure, table), translate it
-[] If you have math equations, define them clearly right before or after the equation.
-[] Each figure/table should be referred to from the text

## Formatting

-[] When citing, the link should be clickable and link to the bibliography
-[] Avoid including pre-print citations when the paper has a published version available (e.g. don't cite "ArXiv" if it's available on ACL anthology)
-[] Use \citet for textual citations, \citep for parenthetical citations. For example, "Devlin et al. (2018) proposed a..." vs "We used the BERT model (Devlin et al. (2018)"
-[] Math equations should be correctly formatted
-[] Figures should be clearly readable (e.g. use PDF, SVG instead of PNG when possible, otherwise use high DPI)
-[] Tables should not be images
-[] You should have clickable links to figures and tables
