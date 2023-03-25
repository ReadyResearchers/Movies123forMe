This website aims to provide movie lovers with an easy-to-use platform to discover new movies and find information about their favorite films. This website is comprised of 1 main page and 5 subpages:

1. [Home Page](#home-page)
    - [Movie Search Subpage](#movie-search-subpage)
    - [Predict Success of a Movie Subpage](#predict-success-of-a-movie-subpage)
    - [What Movie Should You Watch Subpage](#what-movie-should-you-watch-subpage)
    - [Predict-Movie-Success-With-Text Subpage](#predict-movie-success-with-text-subpage)
2. [Analysis Page](#analysis-page)
3. [Analysis Upload Page](#analysis-upload-page)
4. [Clean Data Page](#clean-data-page)
5. [Machine Learning Page](#machine-learning-page)
6. [Regressions Page](#regressions-page)

Let's explore the different subpages available and how to use them!

---

### Home Page

The Home page serves as the gateway to our website, where users can find a brief introduction to the site and a list of the different pages available.
For the purposes of this website, please note that movie success is defined as the likelihood that a movie will generate a profit, given a certain proponent of a movie (such as actors, genre, directors, production budget, etc.).

---

#### Movie Search Subpage

This 'Movie Search' subpage is where users can search for movies by title -- make sure to type the exact title for the best results! Once a search is performed, our website displays the movie that closely matches that search query. Each movie in the list includes the movie title, movie poster, release year, genre, IMDB rating, and a brief description of the movie.

---

#### Predict Success of a Movie Subpage

This 'Predict Success of a Movie' subpage is where users can test out the different features that were found to have a significant impact on movie success. This includes the production budget, the type of rating, and the type of Genre. As this is meant to be an interacive way for users to test out the functionality of the machine learning prediction process, then the values can be reset at each submission. Once a user submits their 'features' for testing, the 6 built-in machine learning model will make their predictions on whether the user-chosen features will equate to:

- a successful movie OR
- a non-successful movie

Users can also see the varying confidence levels of their estimate for each machine learning model. *Confidence levels are based on the accuracy of the model.*

---

#### What Movie Should You Watch Subpage

This 'What Movie Should You Watch' Subpage features 5 different ways for users to get movie recommendations. The 5 ways are:

- plot
- director
- actors/actresses
- genre
- all of the above

For instance, the Plot grouping signifies that a user will want to find movies that have similar plots, the Director grouping signifies that a user will want to find movies that have similar directors, and so on and so forth.

Once a user has chosen the way that they want their movie to be recommended, then all that is left is to choose a movie to start with! Over 8,800 movies currently sit in our database, which are comprised of popular movies from Netflix, Hulu, Amazon Prime, and Disney +. There are also movies being added in that originate from the IMDB and The Number's database of movies.

After a user chooses their desired movie to base the recommendations off of, the application will deliver 5 movies that were closely related to the chosen movie based on the chosen grouping.

*For the most accurate recommendations, try out our 'All' grouping, as it will take all factors and features of a movie into account when making it's recommendations!*

---

#### Predict Movie Success With Text Subpage

This 'Predict Movie Success with Text' Subpage features 6 different ways to train and test our machine learning models using textual data. The 6 different ways available are:

- movie title
- director
- actors/actresses
- MPAA rating
- genre
- plot

Once a user chooses which feature that they would like to input as a way to test the machine learning model, then they are instructed to use the text box provided.

The text box can be filled in with any text needed to make the prediction. For example, a user can input 'Selena Gomez' for the 'Actors' feature to see if their chosen actress (Selena Gomez) can be linked to a successful movie. 

*Please note that movie success is determined by the analysis of the feature chosen agains the movie database's features. This does not necessarily reflect present statistics on the certain feature being chosen.*

---

### Analysis Page

The 'Analysis' Page features 3 different ways for users to visualize the preliminary analysis of the preset movie data, which is described below:

- this graph will take an X and a Y parameter (given by the user) to generate the relationship between the two variables via:
    - ***Bar Chart Graph***
    - ***Scatterplot Graph***
    - ***Funnel Graph***

This is meant to allow the users to see the different kinds of relationships that the variables in the preset data have with each other. To get started, simply [navigate to the analysis page](https://movies123forme/analysis)!

---

### Analysis Upload Page

Similar to the 'Analysis' Page, the 'Analysis Upload Page' has a multitude of different features that can be used to analyze your data (or the preset data)! Supported file formats or CSV or XLSX.

Once a user chooses the dataset that they would like to use, then they have the following visualization options that they can do with their dataframe:

- Info:
    - this visualization will describe the information present in the selected dataframe, such as the:
        - column
        - non-null count
        - data type
- NA Info:
    - this visualization will describe the following in regards to the dataframe's Null values present:
        - the column name
        - the number of null values present in the column
        - the percentage of null values present in comparison to the total number of data points listed
- Descriptive Analysis:
    - this visualization will return a table that contains all of the preliminary statistics for the chosen dataset, such as the:
        - count of rows for each column
        - mean number for each column
        - average standard deviation for each column
        - minimum value for each column
        - average value at the 25% confidence level for each column
        - average value at the 50% confidence level for each column
        - average value at the 75% confidence level for each column
        - maximum value for each column
- Target Analysis:
    - this visualization provides the count of unique values for the chosen target column
- Distribution of Numerical Columns:
    - this visualization shows the dsitribution of the user-chosen numerical columns in the chosen dataset
    - users can see the distribution of one column at a time or they can choose to see the distributions of all numerical columns
- Count Plots of Categorical Columns:
    - this visualization shows the count of the user-chosen categorical (textually-based data) columns in the chosen dataset
    - users can see the count of one column at a time or they can choose to see the count of all categorical columns
- Box Plots:
    - this visualization shows a box plot of the user-chosen numerically-based columns in the chosen dataset
    - users can see the boc plot of one column at a time or they can choose to see the box plots fpr all numerically-based columns
- Outlier Analysis:
    - this visualization shows a dataset of the count of outliers associated with a certain column in the chosen dataset
- Variance of Target Variable with Categorical Columns:
    - this visualization prompts the user for the target variable that is to be analyzed, which can be any of the columns listed in the chosen dataset
    - users then choose which columns they would like to generate plots for, which can be selected one at a time or all at once
    - from there, users choose to perform either of the following to the dataset columns:
        - regressions OR
        - classifications
    - this visualization tool leaves out columns that have multicolinearity by choice, but will plot the associated columns if the user chooses to

To get started, simply [navigate to the analysis upload page](https://movies123forme/analysis_upload)!

---

### Clean Data Page

For the 'Clean Data' Page, users can look at the different dataframes that were used to construct the analysis pages, as well as the machine learning page and Home page's 'What Movie Should I Watch' subpage.

Each of the dataframes had different techniques applied to it to get rid of the null values, change the data types to the correct format, and expand any significant categorical column values into the dummies version (via the `get_dummies` function in Python Pandas).

The cleaned versions of the dataframes are available for the users to download, explore, and maybe *plug into our analysis upload page*. To get started, simply [navigate to the clean data page](https://movies123forme/clean_data)!

---

### Machine Learning Page

For the 'Machine Learning' Page, this page provides users with a tool to apply different machine learning algorithms on a preloaded dataset of movie ratings.

To use this page, simply do the following:
- On the left-hand side of the page, users will find a section where they can choose from a variety of machine learning algorithms, including Random Forest, SVM, and KNN. Each algorithm is designed to analyze the preloaded dataset of movie ratings in a different way
- Users can select which columns from the dataset to include in the algorithm and which column to use as the target variable.
- Once users have selected an algorithm, the page run the algorithm with the chosen values in order to display difference between the predicted values and the actual values for the target variables via plot or confusion matrix.

---

### Regressions Page

For the 'Regressions' Page, this page is designed to provide users with a regression analysis experience by allowing them to explore the relationships between different movie variables.

On the sidebar of the application, there is the option for users to choose the X and Y variables being analyzed in the regression. From there, there are certain visuals that are generated that will show the relationship between the two movie variables chosen.

This regression analysis page shows a scatter plot of the selected movie variables, along with a regression line that represents the relationship between the variables. Users can interact with the plot by hovering over individual data points to see the corresponding movie information.

In addition to the scatter plot, the analysis page also displays summary statistics and the regression equation, allowing users to better understand the relationship between the movie variables.

By selecting the chosen variables and analyzing the scatter plot, users can gain insights into the relationship between the variables and better understand how different aspects of a movie are related.

---
