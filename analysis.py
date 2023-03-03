#!/usr/bin/env python3.9
# coding=utf-8
import zipfile

import numpy as np
import pandas as pd
import pandas.errors
import seaborn as sns
from matplotlib import pyplot as plt


# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str) -> pd.DataFrame:
    """
    Function opens zipped csv files with ';' separated values and loads file content into single pandas DataFrame
    :param filename: path to zipped file
    :return: pandas DataFrame
    """

    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27",
               "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t",
               "p5a"]

    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    # create inverse region dictionary and empty dataframe for concatenation
    my_regions = dict(zip(regions.values(), regions.keys()))
    li = []
    # complete_data = pd.DataFrame()

    # open zipped file in zipped file
    with zipfile.ZipFile(filename) as zfs:
        for name in zfs.namelist():
            with zfs.open(name) as zf:
                with zipfile.ZipFile(zf) as f:
                    # if file is in region dictionary, add its region name and then add it to dataframe
                    for name2 in f.namelist():
                        try:
                            reg = my_regions.get(name2[:2])
                            if not reg:
                                raise pandas.errors.EmptyDataError
                            # reg = [region for region in regions if regions[region] == name2[:2]]
                            raw_data = pd.read_csv(f.open(name2), sep=';', encoding='cp1250', names=headers,
                                                   low_memory=False)
                            raw_data['region'] = reg
                            li.append(raw_data)
                            # complete_data = pd.concat([complete_data, raw_data], ignore_index=True, axis=0)
                        except pandas.errors.EmptyDataError:
                            pass

    return pd.concat(li, ignore_index=True, axis=0)


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Function creates new DataFrame with only a few selected columns. Types of columns in new DataFrame may differ.
     Function does NOT change original DataFrame
    :param df: DataFrame to be parsed
    :param verbose: prints sizes of original and new Dataframes (default = False)
    :return: new DataFrame with parsed values
    """

    data_frame = pd.DataFrame()
    data_frame['crash_id'] = df['p1'].copy()
    data_frame['date'] = pd.to_datetime(df['p2a']).copy()
    data_frame['region'] = df['region'].copy()
    data_frame[['region', 'd', 'e', 'visibility', 'crash_type', 'dead_people', 'seriously_injured_people',
                'lightly_injured_people']] = df[['region', 'd', 'e', 'p19', 'p7', 'p13a', 'p13b', 'p13c']].copy()
    data_frame.drop_duplicates(subset=['crash_id'], inplace=True)

    data_frame['d'] = [str(x).replace(',', '.') for x in data_frame['d']]
    data_frame['e'] = [str(x).replace(',', '.') for x in data_frame['e']]

    data_frame['d'] = pd.to_numeric(data_frame['d'], downcast="float", errors='coerce')
    data_frame['e'] = pd.to_numeric(data_frame['e'], downcast="float", errors='coerce')
    data_frame['crash_id'] = data_frame['crash_id'].astype('category')
    data_frame['visibility'] = data_frame['visibility'].astype('category')
    data_frame['crash_type'] = data_frame['crash_type'].astype('category')
    data_frame['dead_people'] = data_frame['dead_people'].fillna(0).astype('int')
    data_frame['seriously_injured_people'] = data_frame['seriously_injured_people'].fillna(0).astype('int')
    data_frame['lightly_injured_people'] = data_frame['lightly_injured_people'].fillna(0).astype('int')
    data_frame.reset_index(drop=True, inplace=True)

    if verbose:
        print(f'orig_size=%.1f MB\nnew_size=%.1f MB' % (df.memory_usage(index=True, deep=True).sum() / 10 ** 6,
                                                        data_frame.memory_usage(index=True, deep=True).sum() / 10 ** 6))
    return data_frame


# Ukol 3: počty nehod v jednotlivých regionech podle viditelnosti
def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    Function plots number of accident in 4 regions, based on visibility. Function does NOT change original DataFrame
    :param df: DataFrame with values to be plotted, df is not modified
    :param fig_location: save location for figure (default: None)
    :param show_figure: shows plot if True (default: False)
    """

    data = df.copy()
    # [[D-G, D-B, N-G, N-B] [1, 2+3, 4, 5+6+7]
    #  [reg3] [reg4]]
    # To make 4 graphs with 4 types with visibility, new col will be created
    conditions = [data['visibility'] == 1, data['visibility'] == 2, data['visibility'] == 3, data['visibility'] == 4,
                  data['visibility'] == 5, data['visibility'] == 6, data['visibility'] == 7]
    outputs = [0, 1, 1, 2, 3, 3, 3]
    data['my_visibility'] = np.select(conditions, outputs)
    data = data.groupby(['region', 'my_visibility']).agg('count')

    # data.drop(columns=['date', 'd', 'e', 'crash_type', 'visibility'], inplace=True)

    # filter 4 enteries, for each of 4 regions
    data = data.iloc[:4 * 4]
    data.reset_index(inplace=True)

    # Select palette and style for plots; create 2x2 figure
    palette = sns.color_palette('Greens_d', 4)
    sns.set_style("darkgrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    # Plot graphs
    sns.barplot(data=data.loc[data['my_visibility'] == 0], x="region", y="crash_id", ax=ax1,
                palette=palette).set(title='Day, normal visibility')
    sns.barplot(data=data.loc[data['my_visibility'] == 1], x="region", y="crash_id", ax=ax2,
                palette=palette).set(title='Day, bad visibility')
    sns.barplot(data=data.loc[data['my_visibility'] == 2], x="region", y="crash_id", ax=ax3,
                palette=palette).set(title='Night, normal visibility')
    sns.barplot(data=data.loc[data['my_visibility'] == 3], x="region", y="crash_id", ax=ax4,
                palette=palette).set(title='Night, bad visibility')

    # Add title and labels
    fig.suptitle('Accidents in regions based on visibility', fontsize=18)
    ax1.bar_label(ax1.containers[0])
    ax1.set(xlabel='Region', ylabel='Number of accidents')
    ax2.bar_label(ax2.containers[0])
    ax2.set(xlabel='Region', ylabel='Number of accidents')
    ax3.bar_label(ax3.containers[0])
    ax3.set(xlabel='Region', ylabel='Number of accidents')
    ax4.bar_label(ax4.containers[0])
    ax4.set(xlabel='Region', ylabel='Number of accidents')

    fig.tight_layout()

    if fig_location:
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


# Ukol4: druh srážky jedoucích vozidel
def plot_direction(df: pd.DataFrame, fig_location: str = None,
                   show_figure: bool = False):
    """
    Function plots number of accidents during each month for different collision types in 4 different regions.
     Function does NOT change original DataFrame
    :param df: DataFrame with values to be plotted
    :param fig_location: save location for figure (default: None)
    :param show_figure: shows plot if True (default: False)
    """

    data = df.copy()

    # Create new column with types of collision
    data['month'] = pd.DatetimeIndex(data['date']).month
    conditions = [data['crash_type'] == 1, data['crash_type'] == 2, data['crash_type'] == 3,
                  data['crash_type'] == 4, data['crash_type'] == 0]
    outputs = ['frontal', 'side', 'side', 'rear', None]
    data['my_type'] = np.select(conditions, outputs)
    # Count accidents with types of collision from new column
    data = data.groupby(['region', 'month', 'my_type']).agg('count')
    data.reset_index(inplace=True)

    # Plot graphs into 2x2 figure
    sns.set_style("darkgrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
    sns.barplot(data=data.loc[data['region'] == 'HKK'], x="month", y="crash_id", ax=ax1,
                hue="my_type").set(title='Region: HKK')
    sns.barplot(data=data.loc[data['region'] == 'JHC'], x="month", y="crash_id", ax=ax2,
                hue="my_type").set(title='Region: JHC')
    sns.barplot(data=data.loc[data['region'] == 'JHM'], x="month", y="crash_id", ax=ax3,
                hue="my_type").set(title='Region: JHM')
    sns.barplot(data=data.loc[data['region'] == 'KVK'], x="month", y="crash_id", ax=ax4,
                hue="my_type").set(title='Region: KVK')

    # Add labels, title and remove 3 legends that are created by default
    # max_ylim = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1])
    fig.suptitle('Accidents during individual months based on region', fontsize=18)
    ax1.set(xlabel='Month', ylabel='Number of accidents')
    ax1.legend([], [], frameon=False)
    # ax1.set_ylim(top=max_ylim)
    ax2.set(xlabel='Month', ylabel='Number of accidents')
    ax2.legend([], [], frameon=False)
    # ax2.set_ylim(top=max_ylim)
    ax3.set(xlabel='Month', ylabel='Number of accidents')
    ax3.legend([], [], frameon=False)
    # ax3.set_ylim(top=max_ylim)
    ax4.set(xlabel='Month', ylabel='Number of accidents')
    ax4.legend(bbox_to_anchor=(1.25, 1.2), borderaxespad=0., title="Collision type")
    # ax4.set_ylim(top=max_ylim)

    if fig_location:
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


# Ukol 5: Následky v čase
def plot_consequences(df: pd.DataFrame, fig_location: str = None,
                      show_figure: bool = False):
    """
    Function plots number of people who suffered consequences of accidents, between January 1st 2016 and
     January 1st 2022 in 4 different regions. Function does NOT change original DataFrame
    :param df: DataFrame with values to be plotted
    :param fig_location: save location for figure (default: None)
    :param show_figure: shows plot if True (default: False)
    """

    data = df.copy()
    data = data.groupby(['region', pd.Grouper(key='date', freq="M")]).agg('sum')
    data.reset_index(inplace=True)

    # since Dates are from 1.1.2016 to 12.31.2021, drop is technically not necessary, but...
    index = data[(data['date'] < pd.to_datetime('2016-01-01 00:00:00')) |
                 (data['date'] >= pd.to_datetime('2022-01-01 00:00:00'))].index
    data.drop(index, inplace=True)

    sns.set_style("darkgrid")
    # The numbers in region KVK are barely readable with sharedy, therefore it has been omitted
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
    sns.lineplot(data=data.loc[data['region'] == 'HKK'], x="date", y='dead_people', ax=ax1,
                 label='Death').set(title='Region: HKK')
    sns.lineplot(data=data.loc[data['region'] == 'HKK'], x="date", y='seriously_injured_people', ax=ax1,
                 label='Serious')
    sns.lineplot(data=data.loc[data['region'] == 'HKK'], x="date", y='lightly_injured_people', ax=ax1,
                 label='Light')

    sns.lineplot(data=data.loc[data['region'] == 'JHC'], x="date", y='dead_people', ax=ax2).set(title='Region: JHC')
    sns.lineplot(data=data.loc[data['region'] == 'JHC'], x="date", y='seriously_injured_people', ax=ax2)
    sns.lineplot(data=data.loc[data['region'] == 'JHC'], x="date", y='lightly_injured_people', ax=ax2)

    sns.lineplot(data=data.loc[data['region'] == 'JHM'], x="date", y='dead_people', ax=ax3).set(title='Region: JHM')
    sns.lineplot(data=data.loc[data['region'] == 'JHM'], x="date", y='seriously_injured_people', ax=ax3)
    sns.lineplot(data=data.loc[data['region'] == 'JHM'], x="date", y='lightly_injured_people', ax=ax3)

    sns.lineplot(data=data.loc[data['region'] == 'KVK'], x="date", y='dead_people', ax=ax4).set(title='Region: KVK')
    sns.lineplot(data=data.loc[data['region'] == 'KVK'], x="date", y='seriously_injured_people', ax=ax4)
    sns.lineplot(data=data.loc[data['region'] == 'KVK'], x="date", y='lightly_injured_people', ax=ax4)

    # Add title, labels and legend
    fig.suptitle('Number of people suffering consequences based on region (years 2016-2022)', fontsize=18)
    ax1.set(xlabel='Date', ylabel='Number of people')
    ax2.set(xlabel='Date', ylabel='Number of people')
    ax3.set(xlabel='Date', ylabel='Number of people')
    ax4.set(xlabel='Date', ylabel='Number of people')

    ax1.legend([], [], frameon=False)
    fig.legend(loc='center right', title='Collision type')

    if fig_location:
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni 
    # funkce.
    df = load_data("data/data.zip")
    # df.to_csv(r'data\pandas.txt', header=True, index=True, sep=';', mode='w')
    # df = pd.read_csv('data/pandas.txt', delimiter=';', header=0, low_memory=False, index_col=0)
    df2 = parse_data(df, True)

    plot_visibility(df2, "01_visibility.png")
    plot_direction(df2, "02_direction.png", True)
    plot_consequences(df2, "03_consequences.png")

# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
