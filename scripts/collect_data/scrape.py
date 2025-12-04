"""
scrape.py

This script scrapes and saves our faculty dataset:

CS Faculty roster from Jeff Huang's "CS Professors" page:
   - Source: https://drafty.cs.brown.edu/csprofessors
   - Table includes one row per faculty member with multiple metadata columns.

   Output:
   - '../data/hiring/faculty.csv'
     Tabular file with the first 6 columns from the HTML table:
       ['Name', 'Institution', 'Rank', 'Year', 'PhD Institution', 'PhD Year']
     (Exact header names depend on the site; we keep them as-is.)

10/29/2025 — SD
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_faculty():
    """
    Scrape CS faculty roster from Jeff Huang's "CS Professors" page
    and save the first 6 static columns of the table to CSV.

    Source:
        https://drafty.cs.brown.edu/csprofessors

    Output:
        '../data/hiring/faculty.csv'

    Behavior:
        - Locates the HTML table with id='table'.
        - Extracts the <th> header texts (first 6 columns only).
        - Parses the table body from the <template id='table-data'> block.
        - Writes a CSV with those columns in the original order.
    """

    url = 'https://drafty.cs.brown.edu/csprofessors'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'lxml')
    table1 = soup.find('table', id='table')

    # Get column headers
    headers = []
    total_cols = 6
    counter = 0
    for i in table1.find_all('th'):
        if counter < total_cols:
            title = i.text
            headers.append(title)
            counter += 1

    # Create dataframe
    data = pd.DataFrame(columns=headers)
    table = soup.find('template', id='table-data')

    # Fill dataframe with table info
    for j in table.find_all('tr')[1:]:
        row_data = j.find_all('td')
        row = [i.find_next(string=True) for i in row_data]
        length = len(data)
        data.loc[length] = row

    data.to_csv('data/hiring/faculty.csv', index=False)


def main():

    scrape_faculty()


if __name__ == '__main__':
    main()