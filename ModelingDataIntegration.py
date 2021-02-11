"""
Created on Mon Feburary 10 20:01:56 2021

@author: Jinghui Yuan @ UCF
"""
import pandas as pd
import os


def crash_severity_count(crash):
    """

    :param crash:
    :return:
    """
    crash['TotalCrashes'] = crash['CrashSeverity'].str.count('_') + 1
    crash['OCrashes'] = crash['CrashSeverity'].str.count('O')
    crash['KABCCrashes'] = crash['TotalCrashes'] - crash['OCrashes']
    crash = crash.drop(columns=['CrashSeverity'])
    return crash


def crash_aggregate(crash):
    """

    :param crash:
    :return:
    """
    # create different aggregation label
    crash['hour'] = crash['CrashDateTime'].dt.hour
    crash['day'] = crash['CrashDateTime'].dt.dayofweek  # Monday=0, Sunday=6
    crash['period'] = 0  # night time
    crash.loc[crash['hour'].isin([6, 7, 8]), 'period'] = 1  # AM peak
    crash.loc[crash['hour'].isin([16, 17, 18]), 'period'] = 2  # PM peak
    crash.loc[crash['hour'].isin([9, 10, 11, 12, 13, 14, 15]), 'period'] = 3  # Daytime Off-peak

    # encode crash severity to KABCO, depends on the original unique values, crash.CrashSeverity.unique()
    crash['CrashSeverity'] = crash['CrashSeverity'].replace({'Fatality': 'K', 'Injury': 'ABC', 'Property Damage Only': 'O'})

    # create different data sets for different types of aggregation
    weekday_hourly_total_crash = crash[crash['day'].isin([0, 1, 2, 3, 4])].groupby(by=['SegmentId', 'hour'], as_index=False)['CrashSeverity'].agg(lambda x: '_'.join(x))
    weekday_hourly_total_crash = crash_severity_count(weekday_hourly_total_crash)

    weekend_hourly_total_crash = crash[crash['day'].isin([5, 6])].groupby(by=['SegmentId', 'hour'], as_index=False)['CrashSeverity'].agg(lambda x: '_'.join(x))
    weekend_hourly_total_crash = crash_severity_count(weekend_hourly_total_crash)

    weekday_peak_crash = crash[crash['day'].isin([0, 1, 2, 3, 4])].groupby(by=['SegmentId', 'period'], as_index=False)['CrashSeverity'].agg(lambda x: '_'.join(x))
    weekday_peak_crash = crash_severity_count(weekday_peak_crash)

    day_of_week_crash = crash.groupby(by=['SegmentId', 'day'], as_index=False)['CrashSeverity'].agg(lambda x: '_'.join(x))
    day_of_week_crash = crash_severity_count(day_of_week_crash)

    average_daily_crash = crash.groupby(by=['SegmentId'], as_index=True)['CrashSeverity'].apply('_'.join).reset_index(drop=False)
    average_daily_crash = crash_severity_count(average_daily_crash)

    return weekday_hourly_total_crash, weekend_hourly_total_crash, weekday_peak_crash, day_of_week_crash, average_daily_crash


def integrate_crash_traffic_geometric(Final_segment_map, agg_detector_data_path, integrated_model_data_path):
    """

    :param Final_segment_map:
    :param agg_detector_data_path:
    :param integrated_model_data_path:
    :return:
    """
    # generate different crash datasets for five aggregation levels
    weekday_hourly_total_crash, weekend_hourly_total_crash, weekday_peak_crash, day_of_week_crash, average_daily_crash = crash_aggregate(crash)

    aggregate_crash = dict()
    aggregate_crash['weekday_hourly'] = weekday_hourly_total_crash
    aggregate_crash['weekend_hourly'] = weekend_hourly_total_crash
    aggregate_crash['weekday_peak'] = weekday_peak_crash
    aggregate_crash['day_of_week'] = day_of_week_crash
    aggregate_crash['average_daily'] = average_daily_crash

    for aggregation_level in list(['weekday_hourly', 'weekend_hourly', 'weekday_peak', 'day_of_week', 'average_daily']):

        # read the aggregated traffic data given the aggregation level
        aggregate_traffic_data = pd.read_csv(os.path.join(agg_detector_data_path, 'total_{}.csv'.format(aggregation_level)), index_col=0)

        # read aggregated crash data given the aggregation level
        tem_aggregate_crash = aggregate_crash[aggregation_level]

        if aggregation_level in list(['weekday_hourly', 'weekend_hourly']):
            # merge traffic and crash data
            traffic_crash = pd.merge(aggregate_traffic_data, tem_aggregate_crash, how='left', left_on=['SegmentId', 'hour'], right_on=['SegmentId', 'hour'])

        elif aggregation_level == 'weekday_peak':
            traffic_crash = pd.merge(aggregate_traffic_data, tem_aggregate_crash, how='left', left_on=['SegmentId', 'period'], right_on=['SegmentId', 'period'])

        elif aggregation_level == 'day_of_week':
            traffic_crash = pd.merge(aggregate_traffic_data, tem_aggregate_crash, how='left', left_on=['SegmentId', 'day'], right_on=['SegmentId', 'day'])

        else:
            traffic_crash = pd.merge(aggregate_traffic_data, tem_aggregate_crash, how='left', left_on=['SegmentId'], right_on=['SegmentId'])

        # merge traffic, crash, and geometric
        traffic_crash_geometric = pd.merge(traffic_crash, Final_segment_map, how='left', left_on='SegmentId', right_on='SegmentId')

        # fill the empty crash count with 0
        traffic_crash_geometric[['TotalCrashes', 'KABCCrashes', 'OCrashes']] = traffic_crash_geometric[['TotalCrashes', 'KABCCrashes', 'OCrashes']].fillna(0)

        traffic_crash_geometric = traffic_crash_geometric[~((traffic_crash_geometric['VolumeUp'].isnull()) & (traffic_crash_geometric['VolumeDown'].isnull()))].reset_index(drop=True)

        traffic_crash_geometric.to_csv(os.path.join(integrated_model_data_path, '{}_model_data.csv'.format(aggregation_level)))

        print(aggregation_level)


# relative path for the final segment base map
base_map_path = 'Final_segment_map.csv'

# relative path for the rename dictionary
rename_dict_path = 'rename_dictionary.csv'

# relative path for the aggregated traffic data
agg_detector_data_path = 'AggregatedDetectorData'

# relative path for the output integrated model data
integrated_model_data_path = 'Model'

# relative path for the crash data (after spatial join)
crash_data_path = 'crash_data.csv'


if __name__ == '__main__':

    # read the final segment map
    Final_segment_map = pd.read_csv(base_map_path, index_col=0)

    # read the rename dictionary
    rename_dic = pd.read_csv(rename_dict_path)

    # rename variables in Final_segment_map
    Final_segment_map = Final_segment_map.rename(columns=dict(zip(rename_dic["OriginalName"], rename_dic["StandardName"])))

    # read crash data (after spatial join)
    crash = pd.read_csv(crash_data_path, index_col=0)

    # crash datetime format (This should be customized based on specific state)
    crash['CrashDateTime'] = crash['Crash_Date'].str[0:-8] + ' ' + crash['Crash_Time']
    crash['CrashDateTime'] = pd.to_datetime(crash['CrashDateTime'], format='%m/%d/%Y %I:%M %p')

    # rename variables in crash data
    crash = crash.rename(columns=dict(zip(rename_dic["OriginalName"], rename_dic["StandardName"])))

    # call the function to integrate the traffic, crash, and geometric data for model development
    integrate_crash_traffic_geometric(Final_segment_map, agg_detector_data_path, integrated_model_data_path)

