"""
Created on Mon Feburary 08 20:01:56 2021

@author: Jinghui Yuan @ UCF
"""
import pandas as pd
import os
import datetime


def read_data_detector(station_id, path, rename_dic):
    """
    read detector data based on StationId and folder path
    :param station_id: string, detector station id
    :param path: string, folder path
    :param rename_dic: dataframe, variable rename dictionary
    :return: data for specific detector station
    """
    tem_path = os.path.join(path, 'raw_data_{}.csv'.format(station_id))
    try:
        tem = pd.read_csv(tem_path, index_col=0)
        tem = tem.rename(columns=dict(zip(rename_dic["OriginalName"], rename_dic["StandardName"])))
        tem['Timestamp'] = pd.to_datetime(tem['Timestamp'])
    except:
        print('detector no data ', station_id)
        return None
    return tem


def data_cleaning(station_data):
    """
    initial data cleaning
    :param station_data: detector station data
    :return: cleaned station data
    """
    # delete abnormal data that have speed higher than 150 mph
    station_data = station_data.loc[station_data['Speed'] <= 150]

    # delete abnormal data that have volume equals to 0, while speed or occupancy is greater than 0
    station_data = station_data.loc[~((station_data['Volume'] == 0) & ((station_data['Speed'] > 0) | (station_data['Occupancy'] > 0)))]

    return station_data


def get_coefficient_of_variation(data):
    """
    get the coefficient of variation of speed and occupancy
    :param data: dataframe, input data
    :return: dataframe, output data
    """
    data['CoefOfVarSpeed'] = data['StdSpeed']/data['AvgSpeed']
    data['CoefOfVarOccupancy'] = data['StdOccupancy']/data['AvgOccupancy']

    return data


def data_aggregation(station_data, raw_data_resolution, start_year, end_year):
    """
    generate five aggregation datasets based on one detector station data
    :param station_data: dataframe, one detector station data
    :param raw_data_resolution: int, the resolution of the raw traffic data
    :param start_year: int, the start year of the analysis period
    :param end_year: int, the end year of the analysis period
    :return: five aggregated dataframes
    """
    # extract data based on the analysis period
    station_data = station_data[(station_data['Timestamp'] >= datetime.datetime.strptime('{}-01-01 00:00:00'.format(start_year), '%Y-%m-%d %H:%M:%S'))
                                & (station_data['Timestamp'] <= datetime.datetime.strptime('{}-12-31 23:59:59'.format(end_year), '%Y-%m-%d %H:%M:%S'))].reset_index(drop=True)

    # initial data cleaning
    station_data = data_cleaning(station_data)

    # if the resolution of the raw data is lower than 5 min (300 seconds), then we will have an initial aggregation at 5 min level to reduce fluctuation
    if raw_data_resolution < 300:
        station_data['minutes'] = station_data['Timestamp'].dt.minute
        station_data['hour'] = station_data['Timestamp'].dt.hour
        station_data['date'] = station_data['Timestamp'].dt.date
        station_data['5minutes'] = station_data['minutes']//5

        # aggregate detectors at every 5 minutes
        station_data = station_data.groupby(by=['date', 'hour', '5minutes'], as_index=False).agg({'Timestamp': 'last', 'Volume': 'sum', 'Speed': 'mean', 'Occupancy': 'mean'})

    # if the resolution of the raw data is higher than 5 min or the initial 5 min aggregation is done, then continue
    station_data['hour'] = station_data['Timestamp'].dt.hour
    station_data['day'] = station_data['Timestamp'].dt.dayofweek  # day of week, Monday=0, Sunday=6
    station_data['date'] = station_data['Timestamp'].dt.date  # date

    station_data['period'] = 0  # night time
    station_data.loc[station_data['hour'].isin([6, 7, 8]), 'period'] = 1  # AM peak
    station_data.loc[station_data['hour'].isin([16, 17, 18]), 'period'] = 2  # PM peak
    station_data.loc[station_data['hour'].isin([9, 10, 11, 12, 13, 14, 15]), 'period'] = 3  # Daytime Off-peak

    # aggregate at every hour
    hourly = station_data.groupby(by=['date', 'hour'], as_index=False).agg({'Volume': 'sum', 'Speed': ['mean', 'std'], 'Occupancy': ['mean', 'std'], 'day': 'first'})
    hourly.columns = ['date', 'hour', 'Volume', 'AvgSpeed', 'StdSpeed', 'AvgOccupancy', 'StdOccupancy', 'day']
    hourly = get_coefficient_of_variation(hourly)

    # aggregate at every period
    period = station_data.groupby(by=['date', 'period'], as_index=False).agg({'Volume': 'sum', 'Speed': ['mean', 'std'], 'Occupancy': ['mean', 'std'], 'day': 'first'})
    period.columns = ['date', 'period', 'Volume', 'AvgSpeed', 'StdSpeed', 'AvgOccupancy', 'StdOccupancy', 'day']
    period = get_coefficient_of_variation(period)

    # aggregate at every day
    daily = station_data.groupby(by=['date'], as_index=False).agg({'Volume': 'sum', 'Speed': ['mean', 'std'], 'Occupancy': ['mean', 'std'], 'day': 'first'})
    daily.columns = ['date', 'Volume', 'AvgSpeed', 'StdSpeed', 'AvgOccupancy', 'StdOccupancy', 'day']
    daily = get_coefficient_of_variation(daily)

    # generate five types of aggregation for given segment data
    weekday_hourly = hourly[hourly['day'].isin([0, 1, 2, 3, 4])].groupby(by='hour', as_index=False).agg({'Volume': 'mean',
                                                                                                        'AvgSpeed': 'mean', 'StdSpeed': 'mean', 'CoefOfVarSpeed': 'mean',
                                                                                                        'AvgOccupancy': 'mean', 'StdOccupancy': 'mean', 'CoefOfVarOccupancy': 'mean'})
    weekend_hourly = hourly[hourly['day'].isin([5, 6])].groupby(by='hour', as_index=False).agg({'Volume': 'mean',
                                                                                                        'AvgSpeed': 'mean', 'StdSpeed': 'mean', 'CoefOfVarSpeed': 'mean',
                                                                                                        'AvgOccupancy': 'mean', 'StdOccupancy': 'mean', 'CoefOfVarOccupancy': 'mean'})
    weekday_peak = period[period['day'].isin([0, 1, 2, 3, 4])].groupby(by='period', as_index=False).agg({'Volume': 'mean',
                                                                                                        'AvgSpeed': 'mean', 'StdSpeed': 'mean', 'CoefOfVarSpeed': 'mean',
                                                                                                        'AvgOccupancy': 'mean', 'StdOccupancy': 'mean', 'CoefOfVarOccupancy': 'mean'})
    day_of_week = daily.groupby(by='day', as_index=False).agg({'Volume': 'mean', 'AvgSpeed': 'mean', 'StdSpeed': 'mean', 'CoefOfVarSpeed': 'mean',
                                                                                                        'AvgOccupancy': 'mean', 'StdOccupancy': 'mean', 'CoefOfVarOccupancy': 'mean'})
    average_daily = daily.agg({'Volume': 'mean', 'AvgSpeed': 'mean', 'StdSpeed': 'mean', 'CoefOfVarSpeed': 'mean', 'AvgOccupancy': 'mean', 'StdOccupancy': 'mean',
                                                                                                        'CoefOfVarOccupancy': 'mean'}).to_frame().transpose()

    return weekday_hourly, weekend_hourly, weekday_peak, day_of_week, average_daily


def main(Final_segment_map, path, raw_data_resolution, start_year, end_year):
    """
    get the upstream and downstream aggregated traffic data for every segment
    :param Final_segment_map: dataframe, the final segment base map
    :param path: string, the folder path for detector station data
    :param raw_data_resolution: int, the resolution of the raw traffic data
    :param start_year: int, the start year of the analysis period
    :param end_year: int, the end year of the analysis period
    :return: five aggregated dataframes for all segments
    """
    # create empty data
    total_weekday_hourly = []
    total_weekend_hourly = []
    total_weekday_peak = []
    total_day_of_week = []
    total_average_daily = []

    # loop through all the segments in the final segment map
    for i in range(len(Final_segment_map)):
        # get the segment id
        seg_id = Final_segment_map.loc[i, 'SegmentId']

        # get the upstream and downstream detector station id
        up_detector_id = Final_segment_map.loc[i, 'UpStationId']
        down_detector_id = Final_segment_map.loc[i, 'DownStationId']

        # read the corresponding upstream and downstream detector data
        up_data = read_data_detector(up_detector_id, path, rename_dic)
        down_data = read_data_detector(down_detector_id, path, rename_dic)

        # if both upstream and downstream data are not empty, then
        if up_data is not None and down_data is not None:
            weekday_hourly_up, weekend_hourly_up, weekday_peak_up, day_of_week_up, average_daily_up = data_aggregation(up_data, raw_data_resolution, start_year, end_year)
            weekday_hourly_down, weekend_hourly_down, weekday_peak_down, day_of_week_down, average_daily_down = data_aggregation(down_data, raw_data_resolution, start_year, end_year)

            weekday_hourly = pd.merge(weekday_hourly_up, weekday_hourly_down, how='left', left_on='hour', right_on='hour', suffixes=('Up', 'Down'))
            weekend_hourly = pd.merge(weekend_hourly_up, weekend_hourly_down, how='left', left_on='hour', right_on='hour', suffixes=('Up', 'Down'))
            weekday_peak = pd.merge(weekday_peak_up, weekday_peak_down, how='left', left_on='period', right_on='period', suffixes=('Up', 'Down'))
            day_of_week = pd.merge(day_of_week_up, day_of_week_down, how='left', left_on='day', right_on='day', suffixes=('Up', 'Down'))
            average_daily = average_daily_up.merge(average_daily_down, how='outer', left_index=True, right_index=True, suffixes=('Up', 'Down'))

        # if downstream data is empty, while upstream data is not
        elif up_data is not None:
            weekday_hourly_up, weekend_hourly_up, weekday_peak_up, day_of_week_up, average_daily_up = data_aggregation(up_data, raw_data_resolution, start_year, end_year)

            weekday_hourly = weekday_hourly_up.add_suffix('Up')
            weekday_hourly.rename(columns={'hourUp': 'hour'}, inplace=True)

            weekend_hourly = weekend_hourly_up.add_suffix('Up')
            weekend_hourly.rename(columns={'hourUp': 'hour'}, inplace=True)

            weekday_peak = weekday_peak_up.add_suffix('Up')
            weekday_peak.rename(columns={'periodUp': 'period'}, inplace=True)

            day_of_week = day_of_week_up.add_suffix('Up')
            day_of_week.rename(columns={'dayUp': 'day'}, inplace=True)

            average_daily = average_daily_up.add_suffix('Up')

        # if upstream data is empty, while downstream data is not
        elif down_data is not None:
            weekday_hourly_down, weekend_hourly_down, weekday_peak_down, day_of_week_down, average_daily_down = data_aggregation(down_data, raw_data_resolution, start_year, end_year)

            weekday_hourly = weekday_hourly_down.add_suffix('Down')
            weekday_hourly.rename(columns={'hourDown': 'hour'}, inplace=True)

            weekend_hourly = weekend_hourly_down.add_suffix('Down')
            weekend_hourly.rename(columns={'hourDown': 'hour'}, inplace=True)

            weekday_peak = weekday_peak_down.add_suffix('Down')
            weekday_peak.rename(columns={'periodDown': 'period'}, inplace=True)

            day_of_week = day_of_week_down.add_suffix('Down')
            day_of_week.rename(columns={'dayDown': 'day'}, inplace=True)

            average_daily = average_daily_down.add_suffix('Down')

        else:
            print('{} have no detector data for both upstream and downstream'.format(seg_id))
            continue

        weekday_hourly['SegmentId'] = seg_id
        weekend_hourly['SegmentId'] = seg_id
        weekday_peak['SegmentId'] = seg_id
        day_of_week['SegmentId'] = seg_id
        average_daily['SegmentId'] = seg_id

        # append the aggregated data segment by segment
        total_weekday_hourly.append(weekday_hourly)
        total_weekend_hourly.append(weekend_hourly)
        total_weekday_peak.append(weekday_peak)
        total_day_of_week.append(day_of_week)
        total_average_daily.append(average_daily)

        print(i)

    # concatenate the aggregated data for all segments
    total_weekday_hourly = pd.concat(total_weekday_hourly, ignore_index=True).reset_index(drop=True)
    total_weekend_hourly = pd.concat(total_weekend_hourly, ignore_index=True).reset_index(drop=True)
    total_weekday_peak = pd.concat(total_weekday_peak, ignore_index=True).reset_index(drop=True)
    total_day_of_week = pd.concat(total_day_of_week, ignore_index=True).reset_index(drop=True)
    total_average_daily = pd.concat(total_average_daily, ignore_index=True).reset_index(drop=True)

    return total_weekday_hourly, total_weekend_hourly, total_weekday_peak, total_day_of_week, total_average_daily


# relative path of the folder that contains the detector data
detector_data_path = 'StationByStation'

# relative path for the final segment base map
base_map_path = 'Final_segment_map.csv'

# relative path for the rename dictionary
rename_dict_path = 'rename_dictionary.csv'

# relative path for the output aggregated traffic data
agg_detector_data_path = 'AggregatedDetectorData'

# given the resolution of the raw traffic data (seconds)
raw_data_resolution = 20

# specify the start and end year of the analysis period
start_year = 2018
end_year = 2019


if __name__ == '__main__':
    # read the final segment map
    Final_segment_map = pd.read_csv(base_map_path, index_col=0)

    # read the rename dictionary
    rename_dic = pd.read_csv(rename_dict_path)

    # rename variables
    Final_segment_map = Final_segment_map.rename(columns=dict(zip(rename_dic["OriginalName"], rename_dic["StandardName"])))

    # generate aggregated traffic data for every segment
    total_weekday_hourly, total_weekend_hourly, total_weekday_peak, total_day_of_week, total_average_daily = main(Final_segment_map, detector_data_path, raw_data_resolution, start_year, end_year)

    total_weekday_hourly.to_csv(os.path.join(agg_detector_data_path, 'total_weekday_hourly.csv'))
    total_weekend_hourly.to_csv(os.path.join(agg_detector_data_path, 'total_weekend_hourly.csv'))
    total_weekday_peak.to_csv(os.path.join(agg_detector_data_path, 'total_weekday_peak.csv'))
    total_day_of_week.to_csv(os.path.join(agg_detector_data_path, 'total_day_of_week.csv'))
    total_average_daily.to_csv(os.path.join(agg_detector_data_path, 'total_average_daily.csv'))
