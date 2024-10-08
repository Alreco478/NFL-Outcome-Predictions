NOTES: 
- These files are customized to my workspace. It is necessary to change filepaths and names in order to match your workspace.
- Between the dataset generated in pbp_ag_script and the dataset required by DATEX, I manually trimmed the number of columns and removed a large number of features.

FILES: 
- pbp_ag_script.py: Script that takes a folder of csv files downloaded from original data source, then combines them into a raw csv file.
- DATEX.ipynb: Notebook for data exploration. Takes a trimmed version of the dataset that pbp_ag_script.py generates and explores/ cleans it.
