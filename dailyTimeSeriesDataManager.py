# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from datetime import datetime
import pandas as pd
import numpy as np


class DailyTimeSeriesDataManager():
    def __init__(self, kpiFileName, treatmentFileName, identiferColName, kpiColName, firstTimestampColName, kpiTimestampColName, filters):
        self.mapTimeNormalizedToDateTime = dict()
        self.mapDateTimeToTimeNormalized = dict()
        self.state,self.minTimePointForIdentifierDict = self.Read(kpiFileName,
        treatmentFileName,
        identiferColName,
        kpiColName,
        firstTimestampColName,
        kpiTimestampColName,
        filters)

    def generateAndModifyOneToOneAndOntoDateTimeMaps(self, normalizedTimes, dateTimes):
        '''
        Modifies:  self.mapTimeNormalizedToDateTime,self.mapDateTimeToTimeNormalized 
        Assumes: arguments are bijections
        '''
        for x, y in zip(normalizedTimes, dateTimes):
            self.mapTimeNormalizedToDateTime[x] = y
            self.mapDateTimeToTimeNormalized[y] = x

    def generateTimePoints(self, minDayInStudy, dateTimes, modifyDateTimeAndNormalizedTimeMappings=False):
        '''
        Modifies: nothing    
        Returns: a list of time normalized points for the study in the same order as the provided dateTimes
        
        2. Find the minDayInStudy with linear search. If it's not in the study, throw an error.
        3. Go through each item in dateTimes and find the day distance between each item and the MinDayInStudy
        '''
        def isSimilarDate(diff):
            return diff.days == 0

        found = False
        for dt in dateTimes:
            if isSimilarDate(dt-minDayInStudy):
                found = True
        
        if not found:
            raise ValueError("Invalid list of dateTimes because given minDayInStudy is not present.")

        normalizedTimes = [(dt-minDayInStudy).days for dt in dateTimes]
        if modifyDateTimeAndNormalizedTimeMappings:
            self.generateAndModifyOneToOneAndOntoDateTimeMaps(normalizedTimes, dateTimes)
        return normalizedTimes

    def DiffInDiffProcessedDataFrame(self):
        return self.DiffInDiffProcessedDataFrameUnrolledState(**self.state)

    def DiffInDiffProcessedDataFrameUnrolledState(self, minDayInStudy, dateTimes, identifiers, kpis, isTreatmentGroups):
        '''
        Returns: A data frame that is usable by the didDesigner

        The returned data frame will contain:
            1. timePoint
            2. kpi
            3. treatment
            4. identifier
        '''
        df_dict = dict()
        df_dict["dateTimes"] = dateTimes
        df_dict["timePoint"] = self.generateTimePoints(
            minDayInStudy,
            dateTimes,
            modifyDateTimeAndNormalizedTimeMappings=True
        ) 

        df_dict["identifier"] = identifiers
        df_dict["kpi"] = kpis
        df_dict["treat"] = isTreatmentGroups 
        
        return pd.DataFrame(df_dict)

    def generateTreatmentGroups(self, identifierObservations, validIdentifiers):
        '''
        Modifies: nothing
        Note: aliasObservations can contain repeats.
        '''
        s = set(validIdentifiers)

        treatmentGroups = []
        for identifier in identifierObservations:
            if identifier in s:
                treatmentGroups.append(1)
            else:
                treatmentGroups.append(0)
        return treatmentGroups

    def _process_timestamps(self, timestamps, method="ISO"):
        f = None
        if method == "ISO":
            f = lambda x: datetime(int(x.split("T")[0].split("-")[0]),int(x.split("T")[0].split("-")[1]),int(x.split("T")[0].split("-")[2]))
        elif method == "DATEID_UTC":
            f = lambda x : datetime(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:]))
        else:
            raise ValueError("Unsupported method for splitting processed timestamps.")
        processedTimestamps = list(map(f, timestamps))
        return processedTimestamps

    def generateStartDatetimesInStudy(self, df, firstTimestampColName):
        timestamps = np.array(df[firstTimestampColName])
        return self._process_timestamps(timestamps)

    def generateDatetimesInStudy(self, df, timestampName):
        timestamps = np.array(df[timestampName])
        return self._process_timestamps(timestamps, method ="DATEID_UTC")

    def generateMinDayInStudy(self, df, firstTimestampColName):
        datetimes = self.generateStartDatetimesInStudy(df, firstTimestampColName)
        return min(datetimes)

    def generateMinDateInStudyByIdentifier(self, df, minDayInStudy, identiferColName, firstTimestampColName):
        identifiers = np.array(df[identiferColName])
        datetimes = self.generateStartDatetimesInStudy(df, firstTimestampColName)

        dayDiffs = list(map(
            lambda x: (x-minDayInStudy).days,
            datetimes
        ))

        d = dict()
        for identifier, dayDiff in zip(identifiers, dayDiffs):
            if d.get(identifier) == None:
                d[identifier] = dayDiff
            else:
                d[identifier] = min([dayDiff, d[identifier]])

        return d

    def Read(self, kpiFileName, treatmentFileName, identiferColName, kpiColName, firstTimestampColName, kpiTimeStampColName, filters):
        '''
        Returns: a tuple of (arguments for DiffInDiffProcessedDataFrameUnrolledState, dictionary mapping identifiers to their adjusted normalized units after the min point in study)
        Creates: mapping from identifier to minTimePointByEmployee
        '''
        KPI_DF = pd.read_csv(kpiFileName)
        
        self.KPI_DF = KPI_DF
        kpis = self.KPI_DF[kpiColName]
        identifiers = self.KPI_DF[identiferColName]
        START_DATE_DF = pd.read_csv(treatmentFileName)
        for eachFilter in filters:
            START_DATE_DF = START_DATE_DF[(START_DATE_DF[eachFilter] == filters[eachFilter])]

        isTreatmentGroups = self.generateTreatmentGroups(identifiers, START_DATE_DF[identiferColName])
        minDayInStudy = self.generateMinDayInStudy(START_DATE_DF, firstTimestampColName)
        minDateInStudyByIdentifier = self.generateMinDateInStudyByIdentifier(START_DATE_DF, minDayInStudy, identiferColName, firstTimestampColName)
        dateTimes = self.generateDatetimesInStudy(self.KPI_DF, kpiTimeStampColName)
        
        return ({
            "identifiers": identifiers,
            "kpis": kpis,
            "isTreatmentGroups": isTreatmentGroups,
            "minDayInStudy": minDayInStudy,
            "dateTimes": dateTimes
        }, minDateInStudyByIdentifier)


if __name__ == "__main__":
    
    identifierColName = "identifier"
    firstTimestampColName = "timestamp"
    kpiColName = "My KPI"
    kpiFileName = "Example_Kpis.csv"
    kpiTimestampColName = "DateId_UTC"
    treatmentFileName = "Example_Start_Dates.csv"
    filters = {
        "RoleType": "IC"
    }

    dm = DailyTimeSeriesDataManager(
        kpiFileName,
        treatmentFileName,
        identifierColName,
        kpiColName,
        firstTimestampColName,
        kpiTimestampColName,
        filters
    )

    # ---------------------------------------------------------------------------------------
    # Test #1- Test generate Timepoint normalization.
    # ---------------------------------------------------------------------------------------
    datetimes = [datetime(2019, 5, 20), datetime(2019, 5, 29), datetime(2018, 4, 15)]
    minDateTimeInStudy = datetime(2019, 5, 20)
    
    solution = [0, 9, -1*400]
    for t in [t for t in zip(solution,dm.generateTimePoints(minDateTimeInStudy, datetimes)) ]:
        x, y = t
        assert(x == y)

    # ---------------------------------------------------------------------------------------
    # Test #2 - test for generateminDateInStudyByIdentifier
    # ---------------------------------------------------------------------------------------
    
    identifiers = ["chbuja","otherperson"]
    timestamps = ["2019-02-09T21:33:31.608Z", "2019-05-10T04:38:06.9669405Z"]
    df = pd.DataFrame({"EmailName": identifiers, "timestamp":timestamps})
    minDayInStudy = datetime(2019, 2, 9)
    sol = {
        "chbuja": 0,
        "otherperson": (datetime(2019, 5, 10)-datetime(2019, 2, 9)).days
    }
    computedResults = dm.generateMinDateInStudyByIdentifier(df, minDayInStudy, "EmailName", "timestamp")
    assert(sol["chbuja"] == computedResults["chbuja"])
    assert(sol["otherperson"] == computedResults["otherperson"])


    # ---------------------------------------------------------------------------------------
    # Test #3 - Test generateMinDayInStudy
    # ---------------------------------------------------------------------------------------
    
    assert(minDayInStudy == dm.generateMinDayInStudy(df, "timestamp"))

    # ---------------------------------------------------------------------------------------
    # Test #4  - Test min day in study and min date by identifier with real data
    # --------------------------------------------------------------------------------------- 
    minDate = datetime(2019, 2, 9)
    testMinByIdentifier = {
        "be0b3c59-fb6e-452e-a865-8102bad36d70" : 0,
        "95690136-ff30-4f05-bd18-e1d174ce0473": 1,
        "b3600a7c-5b0b-4710-84ac-1db5505269db": 23,
        "9f869dc4-4c1b-4f2f-b24b-47bd621a569b":4
    }

    for key in testMinByIdentifier:
        assert(testMinByIdentifier[key] == dm.minTimePointForIdentifierDict[key])
    for key in dm.minTimePointForIdentifierDict:
        assert(dm.minTimePointForIdentifierDict[key] >= 0)
    df = dm.DiffInDiffProcessedDataFrame()

    assert(minDate == dm.state["minDayInStudy"])
    # --------------------------------------------------------------------------------------- 
    # Test #5 - two individuals should not be in the data frame when reading from the wava data because they are not ICs due to filter settings
    # --------------------------------------------------------------------------------------- 
    testIdentifier1 = "bf64ce0b-f939-4624-9517-ea1cb51a96a4"
    testIdentifier2 = "9d44954f-b5a1-4e7e-9c30-dc30eda51720"
    assert(testIdentifier1 not in dm.minTimePointForIdentifierDict)
    assert(testIdentifier2 not in dm.minTimePointForIdentifierDict)
