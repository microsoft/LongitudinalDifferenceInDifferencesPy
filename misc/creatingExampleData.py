# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import uuid
import numpy as np


def getUUIDs(fileNames):
    '''
    input: a list of filenames with EmailName as column
    output: a dictionary mapping EmailName to a UUID
    '''
    uuidDictionary = dict()
    for fileName in fileNames:
        df = pd.read_csv(fileName)
        for email in df["EmailName"]:
            if uuidDictionary.get(email) is None:
                uuidDictionary[email] = str(uuid.uuid4())
    return uuidDictionary


def makeExampleKPIs(inFile, outFile, uuidDictionary):
    dataFrame = pd.read_csv(inFile)
    resultConstructor = {
        "identifier": [uuidDictionary[email] for email in dataFrame["EmailName"]],
        "DateId_UTC": dataFrame["DateId_UTC"]
    }

    result = pd.DataFrame(resultConstructor)
    result["My KPI"] = np.arange(len(result["identifier"]))*10e7  # Generate arbitrary kpi data.
    result.to_csv(outFile, index=False)


def makeExampleStartDate(inFile, outFile, uuidDictionary):
    dataFrame = pd.read_csv(inFile)
    resultConstructor = {
        "identifier": [uuidDictionary[email] for email in dataFrame["EmailName"]],
        "timestamp": dataFrame["timestamp"],
        "RoleType": dataFrame["RoleType"]
    }

    result = pd.DataFrame(resultConstructor)
    result.to_csv(outFile, index=False)


if __name__ == "__main__":
    kpiFileName = "wavekpis.csv"
    treatmentFileName = "WAVE_Start_Date_Analysis.csv"
    uuidDictionary = getUUIDs([kpiFileName, treatmentFileName])
    makeExampleKPIs(kpiFileName, "Example_Kpis.csv", uuidDictionary)
    makeExampleStartDate(
        treatmentFileName,
        "Example_Start_Dates.csv",
        uuidDictionary
    )
