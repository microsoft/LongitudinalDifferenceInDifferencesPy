# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from dailyTimeSeriesDataManager import DailyTimeSeriesDataManager
from didDesigner import DiDDesigner

if __name__ == "__main__":
    identifierColName = "EmailName"
    firstTimestampColName = "timestamp"
    kpiColName = "CodeAuthoringTimeApproximation"
    kpiFileName = "wavekpis.csv"
    kpiTimestampColName = "DateId_UTC"
    treatmentFileName = "WAVE_Start_Date_Analysis.csv"
    filters = {
        "RoleType": "IC"
    }

    try:
        with open(treatmentFileName, "r") as f:
            pass
    except:
        msg = treatmentFileName + "is not available on your computer. Please try running exampleDIDAnalysis.py if you're looking for a simple demonstration of the tool."
        print(msg)

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

    obj = DiDDesigner(dataManager, preWindowSize,
                      bufferWindowSize, postWindowSize)
    df = obj.dataFrame
    df["kpi"] = df["kpi"] / (3600*1000)
    obj.visualizeDifferenceInDifferences(
        df, "Code Authoring Time (hours)", "Code Authoring Time Difference In Differences")
    obj.visualizeParallelLinesAssumption(
        df, "Code Authoring Time (hours)", "Code Authoring Time - Check Parallel Lines Assumption")
    obj.estimateSummmary(df)
