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

  - MLData
    - Folder containing the following example datasets, these are located specifically in this file for the purpose of the "regular" script exercises.
    - [01HR_Data.csv](./2nd%20day/MLData/01HR_Data.csv)
      - This dataset is composed of two columns:
        - Years Of Experience (float)
        - Salary (integer)
    - [02Companies.csv](./2nd%20day/MLData/02Companies.csv)
      - This dataset is comprised by the following columns:
        - RNDSpend (float)
        - Administration (float)
        - Marketing Spend (float)
        - State (string)
        - Profit (float)
  - Python Scripts
    - [Day2-Correlation Calculation-StudentCopy.ipynb](./2nd%20day/Day2-Correlation%20Calculation-StudentCopy.ipynb)
      - Jupyter Notebook containing examples on how does data correlate between of the fields.
    - [Day2-Demographics Data Analysis-StudentCopy.ipynb](./2nd%20day/Day2-Demographics%20Data%20Analysis-StudentCopy.ipynb)
      - Jupyter Notebook containing the example seen as to how to plot information given and how to start doing data analysis.
    - [MultipleLinearRegression-Personal.py](./2nd%20day/MultipleLinearRegression-Personal.py)
      - Python script showcasing how to run a Multiple Linear regression example on a dataset.
        - The dataset used as an example is [02Companies.csv](./2nd%20day/MLData/02Companies.csv)
    - [SimpleLinearRegression-Personal.py](./2nd%20day/SimpleLinearRegression-Personal.py)
      - Python script containig information regarding how to run a Simple Linear regression on a dataset.
        - The dataset used as an example is [01HR_Data.csv](./2nd%20day/MLData/01HR_Data.csv)
  - Other files
    - [DemographicData.csv](./2nd%20day/DemographicData.csv)
      - CSV file containing demographic data in each country. It's composed by the following columns:
        - Country Name
        - Country Code
        - Birth rate
        - Internet access
        - Income Group
  - [Unedited Scripts](./2nd%20day/Unedited%20Notebooks/)
    - This folder is specifically being used to save the templates that were given during the lessons.
    - Some of the files located in this folder are Jupyter Notebooks and other are Python scripts that were used in Spyder.

- [3rd day](./3rd%20day/)
  - [MLData](./3rd%20day/MLData/)
    - Folder containing the following example datasets, these are located specifically in this file for the purpose of the "regular" script exercises.
    - NewPC.csv
      - Dataset composed by the following columns:
        - Average_income
        - Petrol_Consumption
    - Pressure.csv
      - Dataset composed by the following columns:
        - Temperature
        - Pressure
  - [Original files](./3rd%20day/Original%20files/)
  - Folder containing the original scripts before any alterations were made. - It also contains a folder named [Classification Algorithms](./3rd%20day/Original%20files/Classification%20Algorithms/). This one contains all of the original scripts located in the Classification Algorithms of the **3rd day** folder.
    \_ []()

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

#### To insert commands in Spyder

Use the following shortcut on your keyboard

```keyboard
CTRL + 4
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
