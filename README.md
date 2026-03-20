# Chef d'oeuvre

Code and experiments for the M2 Chef d'oeuvre project on author profiling with Large Language Models, with a focus on bias analysis on song lyrics.

## Overview

This repository contains the code, notebooks, and experimental material used in our project on zero-shot author profiling from song lyrics. The main goal is to study how Large Language Models infer demographic attributes from lyrical text and to analyze the biases that appear in these predictions.

The project covers:

* data processing and pilot experiments
* ground truth construction and topic modeling
* informed prompt experiments
* earlier project data reused for reference

## Repository structure

```text
chef-doeuvre/
├── Data Processing and Pilot Experiments/
├── Ground Truth & Topic Modeling/
├── Informed Prompt Experiments/
├── Previous datas - Projet DEBIAR/
└── .gitignore
```

### Folder description

#### `Data Processing and Pilot Experiments/`

Code and notebooks related to dataset preparation, cleaning, early prompting experiments, and baseline evaluations.

#### `Ground Truth & Topic Modeling/`

Material for metadata preparation, label construction, and topic-level analysis of the lyrics dataset.

#### `Informed Prompt Experiments/`

Experiments using guided prompting strategies for author profiling and bias analysis.

#### `Previous datas - Projet DEBIAR/`

Archived or reused material from earlier related project work.

## Project goals

The project investigates:

* zero-shot author profiling from song lyrics
* prediction of demographic attributes from textual signals
* the effect of prompting strategies on model behavior
* cultural and demographic bias in LLM predictions

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/niyarrbarman/chef-doeuvre.git
cd chef-doeuvre
```

### 2. Create an environment

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Usage

This repository is organized by experiment stage. A typical workflow is:

1. Prepare and clean the data in `Data Processing and Pilot Experiments/`
2. Build or inspect labels and topic structure in `Ground Truth & Topic Modeling/`
3. Run prompt-based experiments from `Informed Prompt Experiments/`
4. Compare outputs, analyze errors, and study bias patterns

Because the repository contains multiple experiment tracks, the exact entry point may differ by folder. Start from the notebooks or scripts in the relevant subdirectory.

## Outputs

Depending on the experiment, outputs may include:

* processed datasets
* metadata tables
* prompt predictions
* evaluation summaries
* figures for reports or presentations

## Contributors

* Niyar R. Barman
* Krish Sharma
* Elouan Vuichard
* Youssef Zidan

