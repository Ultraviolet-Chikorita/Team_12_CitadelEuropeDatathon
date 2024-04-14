import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import datetime
import numpy as np

from midas.mix import mix_freq, mix_freq2
from midas.adl import estimate, forecast, midas_adl, rmse, estimate2,forecast2, midas_adl2

obesity = pd.read_csv("Nutrition_Physical_Activity_and_Obesity_Data.csv")

# Drop the specified columns
obesity.drop(columns=["YearEnd", "LocationDesc", "Datasource", "Topic", "Data_Value_Unit", "Data_Value_Type", "Sample_Size", "Data_Value_Footnote_Symbol", "Data_Value_Footnote", "Low_Confidence_Limit", "High_Confidence_Limit", "Total", "Age(years)", "Education", "Gender", "Grade", "Income", "Race/Ethnicity", "GeoLocation", "ClassID", "TopicID", "QuestionID", "DataValueTypeID", "LocationID", "StratificationCategory1", "Stratification1", "StratificationCategoryId1", "Data_Value_Alt"], inplace=True)

# Drop rows where LocationAbbr is not equal to "US"
obesity = obesity[obesity["LocationAbbr"] == "US"]

obesity.drop("LocationAbbr", axis=1, inplace=True)

# Drop rows where Class is not equal to "Obesity / Weight Status"
obesity = obesity[obesity["Class"] == "Obesity / Weight Status"]

obesity.drop("Class", axis=1, inplace=True)

# Drop rows where "have obesity" is not in the Question column
obesity = obesity[obesity["Question"].str.contains("obesity")]

obesity.dropna(inplace=True)

obesity.sort_values('YearStart', inplace=True)

obesity_children = obesity[obesity["Question"].str.contains("students")]
obesity_adults = obesity[obesity["Question"].str.contains("adults")]

obesity_children.drop("Question", axis=1, inplace=True)
obesity_adults.drop("Question", axis=1, inplace=True)

obesity_children_overall = obesity_children[obesity_children["StratificationID1"] == "OVERALL"]
obesity_adults_overall = obesity_adults[obesity_adults["StratificationID1"] == "OVERALL"]

obesity_children_overall.drop("StratificationID1", axis=1, inplace=True)
obesity_adults_overall.drop("StratificationID1", axis=1, inplace=True)

obesity_children_overall.reset_index(inplace=True)
obesity_children_overall.drop("index", axis=1, inplace=True)

obesity_adults_overall.reset_index(inplace=True)
obesity_adults_overall.drop("index", axis=1, inplace=True)

"""

    Stock data processing

"""

stock = pd.read_csv("all_stock_and_etfs.csv")
stock.drop(columns=["High", "Low", "Close", "Volume"], inplace=True)

# Convert the date column to date format
stock["Date-Time"] = pd.to_datetime(stock["Date-Time"])

hrl = stock[stock["Ticker_Symbol"] == "HRL"]
cag = stock[stock["Ticker_Symbol"] == "CAG"]
ppc = stock[stock["Ticker_Symbol"] == "PPC"]
tsn = stock[stock["Ticker_Symbol"] == "TSN"]

hrl.drop("Ticker_Symbol", axis=1, inplace=True)
cag.drop("Ticker_Symbol", axis=1, inplace=True)
ppc.drop("Ticker_Symbol", axis=1, inplace=True)
tsn.drop("Ticker_Symbol", axis=1, inplace=True)

hrl.sort_values('Date-Time', inplace=True)
cag.sort_values('Date-Time', inplace=True)
ppc.sort_values('Date-Time', inplace=True)
tsn.sort_values('Date-Time', inplace=True)

hrl.reset_index(inplace=True)
hrl.drop("index", axis=1, inplace=True)

cag.reset_index(inplace=True)
cag.drop("index", axis=1, inplace=True)

ppc.reset_index(inplace=True)
ppc.drop("index", axis=1, inplace=True)

tsn.reset_index(inplace=True)
tsn.drop("index", axis=1, inplace=True)


""" 
    Tracking increases in values
"""

children_years = [2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]
adults_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

child_increase = pd.DataFrame(columns=["EndYear", "Increase"])
hrl_child_increase = pd.DataFrame(columns=["EndYear", "Increase"])
cag_child_increase = pd.DataFrame(columns=["EndYear", "Increase"])
ppc_child_increase = pd.DataFrame(columns=["EndYear", "Increase"])
tsn_child_increase = pd.DataFrame(columns=["EndYear", "Increase"])
for i in range(len(children_years)-1):
    curr_year = children_years[i]
    next_year = children_years[i + 1]
    child_curr = obesity_children_overall[obesity_children_overall["YearStart"] == curr_year]
    child_next = obesity_children_overall[obesity_children_overall["YearStart"] == next_year]
    child_curr.reset_index(inplace=True)
    child_curr.drop("index", axis=1, inplace=True)
    child_next.reset_index(inplace=True)
    child_next.drop("index", axis=1, inplace=True)
    child_increase.loc[len(child_increase.index)] = [next_year, child_next.loc[0, :].values.tolist()[1] - child_curr.loc[0, :].values.tolist()[1]]
    hrl_years = hrl[(hrl["Date-Time"].dt.year >= curr_year) & (hrl["Date-Time"].dt.year < next_year)]
    hrl_years.reset_index(inplace=True)
    hrl_years.drop("index", axis=1, inplace=True)
    firstAndLast = hrl_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    hrl_child_increase.loc[len(hrl_child_increase.index)] = [next_year, increase]
    cag_years = cag[(cag["Date-Time"].dt.year >= curr_year) & (cag["Date-Time"].dt.year < next_year)]
    cag_years.reset_index(inplace=True)
    cag_years.drop("index", axis=1, inplace=True)
    firstAndLast = cag_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    cag_child_increase.loc[len(cag_child_increase.index)] = [next_year, increase]
    ppc_years = ppc[(ppc["Date-Time"].dt.year >= curr_year) & (ppc["Date-Time"].dt.year < next_year)]
    ppc_years.reset_index(inplace=True)
    ppc_years.drop("index", axis=1, inplace=True)
    firstAndLast = ppc_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    ppc_child_increase.loc[len(ppc_child_increase.index)] = [next_year, increase]
    tsn_years = tsn[(tsn["Date-Time"].dt.year >= curr_year) & (tsn["Date-Time"].dt.year < next_year)]
    tsn_years.reset_index(inplace=True)
    tsn_years.drop("index", axis=1, inplace=True)
    firstAndLast = tsn_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    tsn_child_increase.loc[len(tsn_child_increase.index)] = [next_year, increase]


adult_increase = pd.DataFrame(columns=["EndYear", "Increase"])
hrl_adult_increase = pd.DataFrame(columns=["EndYear", "Increase"])
cag_adult_increase = pd.DataFrame(columns=["EndYear", "Increase"])
ppc_adult_increase = pd.DataFrame(columns=["EndYear", "Increase"])
tsn_adult_increase = pd.DataFrame(columns=["EndYear", "Increase"])
for i in range(len(adults_years)-1):
    curr_year = adults_years[i]
    next_year = adults_years[i + 1]
    adult_curr = obesity_adults_overall[obesity_adults_overall["YearStart"] == curr_year]
    adult_next = obesity_adults_overall[obesity_adults_overall["YearStart"] == next_year]
    adult_curr.reset_index(inplace=True)
    adult_curr.drop("index", axis=1, inplace=True)
    adult_next.reset_index(inplace=True)
    adult_next.drop("index", axis=1, inplace=True)
    adult_increase.loc[len(adult_increase.index)] = [next_year, adult_next.loc[0, :].values.tolist()[1] - adult_curr.loc[0, :].values.tolist()[1]]
    hrl_years = hrl[(hrl["Date-Time"].dt.year >= curr_year) & (hrl["Date-Time"].dt.year < next_year)]
    hrl_years.reset_index(inplace=True)
    hrl_years.drop("index", axis=1, inplace=True)
    firstAndLast = hrl_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    hrl_adult_increase.loc[len(hrl_adult_increase.index)] = [next_year, increase]
    cag_years = cag[(cag["Date-Time"].dt.year >= curr_year) & (cag["Date-Time"].dt.year < next_year)]
    cag_years.reset_index(inplace=True)
    cag_years.drop("index", axis=1, inplace=True)
    firstAndLast = cag_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    cag_adult_increase.loc[len(cag_adult_increase.index)] = [next_year, increase]
    ppc_years = ppc[(ppc["Date-Time"].dt.year >= curr_year) & (ppc["Date-Time"].dt.year < next_year)]
    ppc_years.reset_index(inplace=True)
    ppc_years.drop("index", axis=1, inplace=True)
    firstAndLast = ppc_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    ppc_adult_increase.loc[len(ppc_adult_increase.index)] = [next_year, increase]
    tsn_years = tsn[(tsn["Date-Time"].dt.year >= curr_year) & (tsn["Date-Time"].dt.year < next_year)]
    tsn_years.reset_index(inplace=True)
    tsn_years.drop("index", axis=1, inplace=True)
    firstAndLast = tsn_years.iloc[[0, -1]]
    firstAndLast.reset_index(inplace=True)
    firstAndLast.drop("index", axis=1, inplace=True)
    increase = firstAndLast.loc[1, :].values.tolist()[1] - firstAndLast.loc[0, :].values.tolist()[1]
    tsn_adult_increase.loc[len(tsn_adult_increase.index)] = [next_year, increase]



hrl_child_value = pd.DataFrame(columns=["Year", "Value"])
cag_child_value = pd.DataFrame(columns=["Year", "Value"])
ppc_child_value = pd.DataFrame(columns=["Year", "Value"])
tsn_child_value = pd.DataFrame(columns=["Year", "Value"])
for curr_year in children_years:
    hrl_year = hrl[hrl["Date-Time"].dt.year == curr_year]
    hrl_year.reset_index(inplace=True)
    hrl_year.drop("index", axis=1, inplace=True)
    hrl_value = hrl_year.loc[0, :].values.tolist()[1]
    hrl_child_value.loc[len(hrl_child_value.index)] = [curr_year, hrl_value]
    cag_year = cag[cag["Date-Time"].dt.year == curr_year]
    cag_year.reset_index(inplace=True)
    cag_year.drop("index", axis=1, inplace=True)
    cag_value = cag_year.loc[0, :].values.tolist()[1]
    cag_child_value.loc[len(cag_child_value.index)] = [curr_year, cag_value]
    ppc_year = ppc[ppc["Date-Time"].dt.year == curr_year]
    ppc_year.reset_index(inplace=True)
    ppc_year.drop("index", axis=1, inplace=True)
    ppc_value = ppc_year.loc[0, :].values.tolist()[1]
    ppc_child_value.loc[len(ppc_child_value.index)] = [curr_year, ppc_value]
    tsn_year = tsn[tsn["Date-Time"].dt.year == curr_year]
    tsn_year.reset_index(inplace=True)
    tsn_year.drop("index", axis=1, inplace=True)
    tsn_value = tsn_year.loc[0, :].values.tolist()[1]
    tsn_child_value.loc[len(tsn_child_value.index)] = [curr_year, tsn_value]

hrl_adult_value = pd.DataFrame(columns=["Year", "Value"])
cag_adult_value = pd.DataFrame(columns=["Year", "Value"])
ppc_adult_value = pd.DataFrame(columns=["Year", "Value"])
tsn_adult_value = pd.DataFrame(columns=["Year", "Value"])
for curr_year in adults_years:
    hrl_year = hrl[hrl["Date-Time"].dt.year == curr_year]
    hrl_year.reset_index(inplace=True)
    hrl_year.drop("index", axis=1, inplace=True)
    hrl_value = hrl_year.loc[0, :].values.tolist()[1]
    hrl_adult_value.loc[len(hrl_adult_value.index)] = [curr_year, hrl_value]
    cag_year = cag[cag["Date-Time"].dt.year == curr_year]
    cag_year.reset_index(inplace=True)
    cag_year.drop("index", axis=1, inplace=True)
    cag_value = cag_year.loc[0, :].values.tolist()[1]
    cag_adult_value.loc[len(cag_adult_value.index)] = [curr_year, cag_value]
    ppc_year = ppc[ppc["Date-Time"].dt.year == curr_year]
    ppc_year.reset_index(inplace=True)
    ppc_year.drop("index", axis=1, inplace=True)
    ppc_value = ppc_year.loc[0, :].values.tolist()[1]
    ppc_adult_value.loc[len(ppc_adult_value.index)] = [curr_year, ppc_value]
    tsn_year = tsn[tsn["Date-Time"].dt.year == curr_year]
    tsn_year.reset_index(inplace=True)
    tsn_year.drop("index", axis=1, inplace=True)
    tsn_value = tsn_year.loc[0, :].values.tolist()[1]
    tsn_adult_value.loc[len(tsn_adult_value.index)] = [curr_year, tsn_value]


# PRINTING

#print(tabulate(obesity_children_overall, headers='keys', tablefmt='psql'))
#print(tabulate(obesity_adults_overall, headers='keys', tablefmt='psql'))

#print(tabulate(child_increase, headers='keys', tablefmt='psql'))
#print(tabulate(hrl_child_increase, headers='keys', tablefmt='psql'))
#print(tabulate(cag_child_increase, headers='keys', tablefmt='psql'))
#print(tabulate(ppc_child_increase, headers='keys', tablefmt='psql'))
#print(tabulate(tsn_child_increase, headers='keys', tablefmt='psql'))

# plt.figure(figsize=(160, 80), dpi=150)
# hrl_child_increase['Increase'].plot(label='HRL')
# cag_child_increase['Increase'].plot(label='CAG')
# ppc_child_increase['Increase'].plot(label='PPC')
# tsn_child_increase['Increase'].plot(label='TSN') 
# plt.title('meat processing stock increase') 
# plt.xlabel('Year')
# plt.ylabel('increase over previous 2 years / $')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# child_increase['Increase'].plot(label='Obesity Increase / %', color='orange')
# plt.title('child Obesity increase') 
# plt.xlabel('Year')
# plt.ylabel('increase over previous 2 years / %')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# hrl_adult_increase['Increase'].plot(label='HRL')
# cag_adult_increase['Increase'].plot(label='CAG')
# ppc_adult_increase['Increase'].plot(label='PPC')
# tsn_adult_increase['Increase'].plot(label='TSN') 
# plt.title('meat processing stock increase') 
# plt.xlabel('Year')
# plt.ylabel('increase over previous 2 years / $')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# adult_increase['Increase'].plot(label='Obesity Increase / %', color='orange')
# plt.title('adult Obesity increase') 
# plt.xlabel('Year')
# plt.ylabel('increase over previous 2 years / %')
# plt.legend()
# plt.show()

"""
    ^ ^ ^ ^ ^ ^ ^ ^ ^
    | | | | | | | | |
    CONCLUSION FROM COMPARING RATES OF CHANGE: adult obesity increase is relatively constant, there is low correlation as changes in the increase of stock to the increase of obesity
    rate of increase of children's obesity level seems to be roughly constant, but fluctuating. The fluctuations seem to follow the changes of stocks 4 years prior. This could be erroneous due to the limited number of data points. Even if true, the movements can't be uniformly followed, as there's many other factors involved.
"""

# plt.figure(figsize=(160, 80), dpi=150)
# hrl_child_value["Value"].plot(label='HRL')
# cag_child_value["Value"].plot(label='CAG')
# ppc_child_value["Value"].plot(label='PPC')
# tsn_child_value["Value"].plot(label='TSN')
# plt.title('meat processing stock prices') 
# plt.xlabel('Year')
# plt.ylabel('Price per share / $')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# obesity_children_overall['Data_Value'].plot(label='Obesity / %', color='orange')
# plt.title('child Obesity') 
# plt.xlabel('Year')
# plt.ylabel('percentage obese / %')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# hrl_adult_value["Value"].plot(label='HRL')
# cag_adult_value["Value"].plot(label='CAG')
# ppc_adult_value["Value"].plot(label='PPC')
# tsn_adult_value["Value"].plot(label='TSN')
# plt.title('meat processing stock prices') 
# plt.xlabel('Year')
# plt.ylabel('Price per share / $')
# plt.legend()
# plt.show()

# plt.figure(figsize=(160, 80), dpi=150)
# obesity_adults_overall['Data_Value'].plot(label='Obesity / %', color='orange')
# plt.title('adult Obesity') 
# plt.xlabel('Year')
# plt.ylabel('percentage obese / %')
# plt.legend()
# plt.show()


#print(tabulate(hrl.head(), headers='keys', tablefmt='psql'))
#print(tabulate(hrl.tail(), headers='keys', tablefmt='psql'))
#print(tabulate(cag.head(), headers='keys', tablefmt='psql'))
#print(tabulate(cag.tail(), headers='keys', tablefmt='psql'))
#print(tabulate(ppc.head(), headers='keys', tablefmt='psql'))
#print(tabulate(ppc.tail(), headers='keys', tablefmt='psql'))
#print(tabulate(tsn.head(), headers='keys', tablefmt='psql'))
#print(tabulate(tsn.tail(), headers='keys', tablefmt='psql'))

"""
    Predictive model below - USE CANADA DATA AS IT EXTENDS FURTHER BACK IN TIME, AND HAS HIGH CORRELATION WITH EXISTING US DATA (RMSE)
    |
    V
"""

external_dataset = pd.read_csv("obesity-cleaned.csv", parse_dates=['Year'], index_col='Year')
external_dataset = external_dataset[external_dataset["Country"] == "Canada"]
external_dataset.reset_index(inplace=True)
external_dataset.drop("Country", axis=1, inplace=True)

canada_overall = external_dataset[external_dataset["Sex"] == "Both"]
canada_overall.set_index("Year", inplace=True)
print(canada_overall.head(30))

hrl.set_index("Date-Time", inplace=True)
cag.set_index("Date-Time", inplace=True)
ppc.set_index("Date-Time", inplace=True)
tsn.set_index("Date-Time", inplace=True)

rmse_hrl, hrl1 = midas_adl(canada_overall.Obesity, hrl.Open, 
                         start_date=datetime.datetime(1999,1,1),
                         end_date=datetime.datetime(2016,1,1),
                         xlag=1,
                         ylag=1,
                         horizon=1,
                         poly='expalmon',
                         method='rolling')

print(rmse_hrl)



"""
    https://library.ndsu.edu/ir/bitstream/handle/10365/28017/Quantifying%20Relationships%20Between%20Two%20Time%20Series%20Data%20Sets.pdf?sequence=1
    https://faculty.washington.edu/ezivot/econ584/notes/varModels.pdf
    https://www.jstor.org/stable/pdf/community.28111642.pdf?refreqid=fastly-default%3Ad8549d048c8fdc0eb18423e0e4bcb166&ab_segments=&origin=&initiator=&acceptTC=1
    https://www.mdpi.com/2227-7390/11/4/1054
"""