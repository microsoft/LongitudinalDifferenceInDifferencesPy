# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from dailyTimeSeriesDataManager import DailyTimeSeriesDataManager
from didDesigner import DiDDesigner
import json

if __name__ == "__main__":
    identifierColName = ""
    firstTimestampColName = ""
    kpiColName = ""
    kpiFileName = ""
    kpiTimestampColName = ""
    treatmentFileName = ""
    kpiAxisLabel = ""
    differenceInDifferencesTitle = ""
    parallelLinesTitle = ""
    filters = None
    kpiScalingFactor = 1

    with open("config.json", "r") as f:
        configData = json.load(f)
        identifierColName = configData["identifierColName"]
        firstTimestampColName = configData["firstTimestampColName"]
        kpiColName = configData["kpiColName"]
        kpiFileName = configData["kpiFileName"]
        kpiTimestampColName = configData["kpiTimestampColName"]
        treatmentFileName = configData["treatmentFileName"]
        filters = configData["filters"]
        kpiAxisLabel = configData["kpiAxisLabel"]
        differenceInDifferencesPlotTitle = configData["differenceInDifferencesPlotTitle"]
        parallelLinesPlotTitle = configData["parallelLinesPlotTitle"]
        kpiScalingFactor = configData["kpiScalingFactor"]

    dataManager = DailyTimeSeriesDataManager(
        kpiFileName,
        treatmentFileName,
        identifierColName,
        kpiColName,
        firstTimestampColName,
        kpiTimestampColName,
        filters
    )

    preWindowSize = 28
    bufferWindowSize = 7
    postWindowSize = 28

    obj = DiDDesigner(
        dataManager,
        preWindowSize,
        bufferWindowSize,
        postWindowSize
    )

    df = obj.dataFrame
    df["kpi"] = df["kpi"] * kpiScalingFactor
    obj.visualizeDifferenceInDifferences(df, kpiAxisLabel, differenceInDifferencesPlotTitle)
    obj.visualizeParallelLinesAssumption(df, kpiAxisLabel, parallelLinesPlotTitle)
    obj.estimateSummmary(df)
