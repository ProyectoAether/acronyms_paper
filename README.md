# How does GPT-2 Predict Acronyms? Extracting and Understanding a Circuit via Mechanistic Interpretability

This repository contains the data and code used to perform the experiments of the paper: How does GPT-2 Predict Acronyms? Extracting and Understanding a Circuit via Mechanistic Interpretability. It also includes the generated figures.

# How to use

First, clone the repository and install the required dependencies:

```
git clone https://github.com/jgcarrasco/acronyms_paper.git
cd acronyms_paper
pip install -r requirements.txt
```

Then, run the different scripts to replicate the figures presented in the paper. You can specify the number of samples used via an argument, for example `python positional_experiments.py -n 500` to run with a dataset of 500 samples. Each script generates the figures specified below:

- `patching_experiments.py`: Figures 1, 2, 4, 5, 6, 7. It is used to perform the activation patching experiments to identify and isolate the underlying circuit associated to the acronym prediction task.
- `histogram.py`: Figure 3. It is used to visualize the attention paid to the different tokens.
- `evaluation.py`: Figure 8. Evaluates the identified circuit by ablating every attention head and then iteratively adding the components of the circuit, showing that the performance is recovered.
- `mover_heads.py`: Figures 9, 10, 11. Plots the OV circuits of the individual letter mover heads, as well as the combined OV circuit. It also shows a scatter plot that provides evidence about their copying behavior.
- `positional_experiments.py`: Figures 12, 13, 14. Experiments performed to study how the positional information is propagated across the circuit.

# Citation

For any question and/or suggestion, do not hesitate to reach out. If you use our work, you can reference it as follows:

```
@inproceedings{garcia2024does,
  title={How does GPT-2 Predict Acronyms? Extracting and Understanding a Circuit via Mechanistic Interpretability},
  author={Garc{\'\i}a-Carrasco, Jorge and Mat{\'e}, Alejandro and Trujillo, Juan Carlos},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={3322--3330},
  year={2024},
  organization={PMLR}
}
```
