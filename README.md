# Machine Learning & Deep Learning with Python - Training

## General information

### Contents in each folder

- [1st day](./1st%20day/)
  - Jupyter Notebooks
    - [Day1-Basics.ipynb](./1st%20day/Day1-Basics.ipynb)
      - Jupyter Notebook containing basic Python syntax as well as "introductory" information (such as variable declaration and arithmetic operations).
    - [Day1-Data Visualization.ipynb](./1st%20day/Day1-Data%20Visualization.ipynb)
      - Jupyter Notebook containing basic information to render plots using the _Matplotlib_ module.
    - [Day1-Numpy arrays.ipynb](./1st%20day/Day1-Numpy%20arrays.ipynb)
      - Jupyter Notebook containing introductory information as to handling Numpy Arrays.
    - [Day1-Pandas Dataframe.ipynb](./1st%20day/Day1-Pandas%20Dataframe.ipynb)
      - Jupyter Notebook containing basic information regarding data manipulation in a Pandas Dataframe.
  - Other files
    - [CleanedDF.csv](./1st%20day/CleanedDF.csv)
      - Output from the cleaned Pandas Dataframe to a CSV file.
    - [DFDescribe.csv](./1st%20day/DFDescribe.csv)
      - Output from the Pandas Dataframe after appending the _describe()_ function.
    - [Empexport.csv](./1st%20day/Empexport.csv)
      - CSV file containing a dataset information regarding employee data. It lacks headers.
- [2nd day](./2nd%20day/)
  - Jupyter Notebooks
    - [Day2-Demographics Data Analysis-StudentCopy.ipynb](./2nd%20day/Day2-Demographics%20Data%20Analysis-StudentCopy.ipynb)
      - Jupyter Notebook containing the example seen as to how to plot information given and how to start doing data analysis.
  - Other files
    - [DemographicData.csv](./2nd%20day/DemographicData.csv)
      - CSV file containing demographic data in each country. It's composed by the following columns:
        - Country Name
        - Country Code
        - Birth rate
        - Internet access
        - Income Group
  - [Unedited Notebooks](./2nd%20day/Unedited%20Notebooks/)
    - This folder is specifically being used to save the templates that were given during the lessons.

### Times in the course

|                | PST   | CST   |
| -------------- | ----- | ----- |
| START          | 7:00  | 9:00  |
| 15 min break   | 8:30  | 10:30 |
| Long 1hr break | 10:30 | 12:30 |
| 15 min break   | 13:15 | 15:15 |
| STOP           | 16:00 | 18:00 |

### Commands cheat sheet

#### On Windows

- Open Anaconda Prompt and run the following command:

```CMD
jupyter notebook --notebook-dir="path/to/directory"
```

#### On MacOS

- Open your Terminal and run the following command:

```bash
jupyter notebook --notebook-dir="path/to/directory"
```

## Machine Learning Libraries for Python

- Numpy
  - Numeric array operations
  - High performance arrays
  - Vectorization Algorithm
- Pandas
  - Name comes fromes Panel Data -> **Pan** + **Das**
  - Dataframe
    - May be taken from an Excel sheet
  - Data cleaning and exploration
  - More organized than Numpy
- Scipy
  - Complex statistical calculations
- Scikit
  - Algorithm implementation
- Matplotlib
  - Data visualization
- Seaborn
  - Data visualization
  - Based on matplotlib.
