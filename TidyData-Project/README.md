# 🥇 Exploring the 2008 Summer Olympics Medalists 🥇

This project will clean and wrangle data from the 2008 Summer Olympics using tidy data principles to explore medal count distributions by event and gender.

<img src="https://github.com/user-attachments/assets/5e0680fe-e410-4b42-87cb-4e849620f472" width="200" height="200">

*"It's not about winning at the Olympic Games. It's about trying to win. The motto is faster, higher, stronger, not fastest, highest, strongest. Sometimes it's the trying that matters." - Bronte Barratt, Australian swimmer and 2008 gold medalist*

## 🗒️ Tidy Data Principles Overview

1. Each variable is in its own column.
2. Each observation forms its own row.
3. Each type of observational unit forms its own table.

Further Reading on the Importance of Tidy Data: [Tidy Data by Hadley Wickham](https://github.com/user-attachments/files/19293365/tidy-data.pdf)

## 💯 Data

- Name, gender, event, and medal data from a [csv file](https://github.com/atravlos/Data-Science-Portfolio/TidyData-Project/olympics_08_mdalists.csv) that was adapted from ["Global/All Medalists" Dataset by Giorgio Comai](https://edjnet.github.io/OlympicsGoNUTS/2008/). The original dataset contains additional information including country represented and place of birth.
>- Adaptations to get the csv file included removing all columns other than medalist_name, event_sport, sex_or_gender, and medal. Then, sex_or_gender and event_sport columns were merged. Dataframe was then pivoted so that the gender+event features were the columns and the medals won were the values.

*"Those Olympics in 2008 changed my life" - Usain Bolt*

## 🤓 Approach

- Method chaining to melt, rename, and sort data by medal awarded per event and gender.
- Aggregate and visualize data to evaluate medals awarded by event and gender.

*"There are a lot of guys who play in the NBA. There aren't a lot of guys who have a chance to win a gold medal, too." - James Harden*

## 📚 Libraries

- **Pandas** for data cleaning and wrangling
- **Matplotlib.pyplot** for visualization

## ✍️ Instructions

1. Download [olympics_08_medalists.csv](https://github.com/atravlos/Data-Science-Portfolio/TidyData-Project/olympics_08_mdalists.csv)
2. Run first Python cell with import statements to import Pandas and Matplotlib.pyplot libraries
    <img width="1101" alt="Screenshot 2025-03-17 at 2 40 08 PM" src="https://github.com/user-attachments/assets/94937c51-f485-48bf-adaf-18a8146dce6b" />
3. Run second python cell to load in .csv file as Pandas dataframe.
    <img width="1098" alt="Screenshot 2025-03-17 at 2 42 12 PM" src="https://github.com/user-attachments/assets/9cdf3d59-0e5d-40ff-a342-5f94ab9bcc87" />
4. From there, run the rest of the Python cells sequentially to appropriately clean the data to TidyData Principles and then visualize the data.

## 📈 Sample

Bar plot showing medal distribution by gender:

![output](https://github.com/user-attachments/assets/a4d75b4f-f53c-4505-b190-a13ad62cdba5)

Bar plot showing medal count by event:

![output1](https://github.com/user-attachments/assets/ae0e419d-e6f0-4a3d-a9a0-efe6fb8b35b1)

## 🗄️ Helpful Resources

- [Pandas Cheat Sheet](https://github.com/user-attachments/files/19293925/Pandas_Cheat_Sheet.pdf)


