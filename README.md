# 4Geeks data science project boilerplate
[![Codespaces Prebuilds](https://github.com/gperdrizet/4Geeks_datascience_project/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/gperdrizet/4Geeks_datascience_project/actions/workflows/codespaces/create_codespaces_prebuilds)

Minimal Python 3.11 repository for 4Geeks data science assignments. Several useful Python packages and VSCode extensions are installed on Codespace boot-up. Directories for models and data are created within the Codespace but excluded from tracking. The notebooks directory contains `notebook.ipynb`, run this notebook to verify the environment. It can then be deleted or renamed to use for your project.

## 1. Getting started

### Option 1: GitHub Codespaces (Recommended)

1. **Fork the Repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - 4Geeks students: set 4GeeksAcademy as the owner - 4Geeks pays for your codespace usage. All others, set yourself as the owner
   - Give the fork a descriptive name. 4Geeks students: I recommend including your GitHub username to help in finding the fork if you loose the link
   - Click "Create fork"
   - 4Geeks students: bookmark or otherwise save the link to your fork

2. **Create a GitHub Codespace**
   - On your forked repository, click the "Code" button
   - Select "Create codespace on main"
   - If the "Create codespace on main" option is grayed out - go to your codespaces list from the three-bar menu at the upper left and delete an old codespace
   - Wait for the environment to load (dependencies are pre-installed)

3. **Start Working**
   - Open `notebooks/notebook.ipynb` in the Jupyter interface
   - Run the notebook to verify that your environment is working correctly - if there are no errors, you are all set!

### Option 2: Local Development

1. **Prerequisites**
   - Git
   - Python >= 3.10

2. **Fork the repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - Optional: give the fork a new name and/or description
   - Click "Create fork"

3. **Clone the repository**
   - From your fork of the repository, click the green "Code" button at the upper right
   - From the "Local" tab, select HTTPS and copy the link
   - Run the following commands on your machine, replacing `<LINK>` and `<REPO_NAME>`

   ```bash
   git clone <LINK>
   cd <REPO_NAME>
   ```

4. **Set Up Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Launch Jupyter & start the notebook**
   ```bash
   jupyter notebook notebooks/notebook.ipynb
   ```

   Once the notebook opens in your web browser, run it once to verify that your environment is working correctly - if there are no errors, you are all set!

## 2. Environment

### 2.1. Repository structure

```text
.
├──.devcontainer
│   └── devcontainer.json  # Codespace/devcontainer configuration
│
├── .gitignore             # Files and directories listed will be ingored by git
├── LICENSE                # Open source GNU license - copy, modify and distribute this repo freely
├── README.md              # This file
├── data/                  # Empty directory for data
├── models/                # Empty directory for models
├── notebooks              # Notebooks directory
│   └── notebook.ipynb     # Test notebook with library version checks
│
└── requirements.txt       # List of Python packages installed during Codespace creation
```

### 2.2. Python
**Base image**: [Python 3.11](https://github.com/devcontainers/images/tree/main/src/python)

Packages installed via `requirements.txt`:

1. [ipykernel 6.30.0](https://pypi.org/project/ipykernel)
2. [Jupyter](https://jupyter.org)
3. [matplotlib 3.10.3](https://matplotlib.org/stable/index.html)
4. [numpy 2.3.2](https://numpy.org/doc/stable/index.html)
5. [pandas 2.3.1](https://pandas.pydata.org/docs)
6. [pyarrow 21.0.0](https://arrow.apache.org/docs/python/index.html)
7. [scipy 1.16.1](https://scipy.org)
8. [scikit-learn 1.7.1](https://scikit-learn.org/stable/index.html)
9. [seaborn 0.13.2](https://seaborn.pydata.org)

If you need to install additional Python packages, you can do so via the terminal with: `pip install packagename`.

### 2.3. VSCode extensions

Sepcified via `devcontainier.json`.

1. [ms-python.python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
2. [ms-toolsai.jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
3. [streetsidesoftware.code-spell-checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)

VSCode extensions can be added via the *Extensions* tab located on the activities panel at the left once inside the Codespace.
