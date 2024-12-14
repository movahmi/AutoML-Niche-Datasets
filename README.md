
# AutoML for Niche Datasets

This project is designed to provide an AutoML solution tailored to niche datasets using TPOT and Auto-Sklearn.

## Features
- Domain-specific preprocessing for datasets from various fields (e.g., finance, art).
- Integration with TPOT and Auto-Sklearn for automated machine learning.

## Setup
1. Clone this repository.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script using the following command:
```bash
python main.py --file <path_to_dataset> --domain <domain_name> --tool <tool_name>
```
- `<path_to_dataset>`: Path to your dataset file (CSV format).
- `<domain_name>`: Domain type (e.g., finance, art).
- `<tool_name>`: AutoML tool to use (`tpot` or `autosklearn`).

## Example
```bash
python main.py --file data.csv --domain finance --tool tpot
```

## Requirements
- Python 3.8 or later
- TPOT
- Auto-Sklearn
- scikit-learn
- pandas

## License
This project is open-source and free to use.
