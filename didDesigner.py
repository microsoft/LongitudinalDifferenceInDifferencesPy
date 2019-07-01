# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind as ttest
import matplotlib.dates as matplotlibDates
from datetime import datetime

plt.style.use("classic")


class DiDDesigner():
    def __init__(self, timeSeriesDataManager, preWindowSize, bufferWindowSize, postWindowSize):
        self.timeSeriesDataManager = timeSeriesDataManager

        if self.timeSeriesDataManager != None:
            #Use window sizes to compute values of post.
            self.dataFrame = self.timeSeriesDataManager.DiffInDiffProcessedDataFrame()
            postIndicators = self.calculatePostValues( 
                self.dataFrame["timePoint"]
                ,preWindowSize
                ,bufferWindowSize
                ,postWindowSize
                ,self.dataFrame["identifier"]
                ,self.minTimePointForIdentifierDict)
            self.dataFrame["post"] = postIndicators

    def printCohortNumbers(self):
        print("Number of identities marked as treated:")
        print(len(self.timeSeriesDataManager.minTimePointForIdentifierDict))
        print("")
        print("Number of identities marked as treated and have KPIs:")
        print(len(self.dataFrame[self.dataFrame.treat == 1].groupby(["identifier"]).mean()))
        print("")
        print("Number of identities marked as untreated and have KPIs:")
        print(len(self.dataFrame[self.dataFrame.treat == 0].groupby(["identifier"]).mean()))

    def calculatePostValues(self, timePoints, preWindowSize, bufferWindowSize, postWindowSize, identifiers, minTimePointForIdentifierDict):
        isPostValues = []
        zeroIsFound = False

        def isPostTreatment(timeVal, _preWindowSize, _bufferWindowSize, _postWindowSize, adjustmentForIdentifierStartingTime):
            adjustedTimeVal = timeVal - adjustmentForIdentifierStartingTime
            if adjustedTimeVal >= -1 * _preWindowSize and adjustedTimeVal <= (-1):
                return 0
            if adjustedTimeVal >= _bufferWindowSize and adjustedTimeVal <= _bufferWindowSize + _postWindowSize - 1:
                return 1
            return np.nan
        
        for timePoint, identifier in zip(timePoints, identifiers):
            if timePoint == 0:
                zeroIsFound = True
            isPostValues.append(
                isPostTreatment(timePoint, preWindowSize, bufferWindowSize, postWindowSize,
                minTimePointForIdentifierDict.get(identifier, 0) #if treat is zero, then generate post based on the timepoint of zero
                )
            )

        if not zeroIsFound:
            raise Exception("No minumum timepoint found in study.")
        
        # We need to make sure that there are no identifiers have just pre or post values.
        '''
        Check that each identifier has it has both a pre and post value.

        Create a hash table hasPreHasPost for an identfier mapping it to a list of length two of boolean values initialized to False.

        New data structure removeFromStudyBecauseDoesNotHavePreAndPost for mapping identifier if we should go back write. This criteria is the xor of the value of the hasPreHasPost table.

        For each identifier, mark everything as nan if removeFromStudyBecauseDoesNotHavePreAndPost is true.
        '''
        hasPreHasPost = dict()
        for identifier in identifiers:
            if hasPreHasPost.get(identifier) is None:
                hasPreHasPost[identifier] = [False, False]

        for isPostValue, identifier in zip(isPostValues, identifiers):
            if isPostValue == 0:
                hasPreHasPost[identifier][0] = True
            if isPostValue == 1:
                hasPreHasPost[identifier][1] = True
            
        def xor(v1, v2):
            return bool(v1) ^ bool(v2)
        
        removeFromStudyBecauseDoesNotHavePreAndPost = dict()

        for identifier in hasPreHasPost:
            removeFromStudyBecauseDoesNotHavePreAndPost[identifier] = xor(*hasPreHasPost[identifier])
        del hasPreHasPost

        isPostValuesGuaranteedToHavePreAndPostTreatmentByIdentifier = list()
        
        for isPostValue, identifier in zip(isPostValues, identifiers):
            if removeFromStudyBecauseDoesNotHavePreAndPost[identifier]:
                isPostValuesGuaranteedToHavePreAndPostTreatmentByIdentifier.append(np.nan)
            else:
                isPostValuesGuaranteedToHavePreAndPostTreatmentByIdentifier.append(isPostValue)
        return isPostValuesGuaranteedToHavePreAndPostTreatmentByIdentifier
    
    def _ttest_with_identifier_aggregation_setup(self, df, isTreatmentGroup=1):
        if isTreatmentGroup != 0 and isTreatmentGroup != 1:
            raise ValueError("Provide a valid isTreatmentGroup, instead of: " + str(isTreatmentGroup))

        return (df[(df.treat == isTreatmentGroup) & (df.post == 1)],
        df[(df.treat == isTreatmentGroup) & (df.post == 0)])

    def postPreTTest(self,df,isTreatmentGroup=1,equal_var=True,groupbyidentifier=True):
        """
        Run a T-test on the pre group versus the post group, grouped by the identifier and taking the mean.

        We are comparing the means of each identifier before and after the test.
        """
        postTreatmentDataFrame, preTreatmentDataFrame = self._ttest_with_identifier_aggregation_setup(df, isTreatmentGroup)

        if groupbyidentifier:
            postTreatmentDataFrame = postTreatmentDataFrame.groupby(["identifier"]).mean()
            preTreatmentDataFrame = preTreatmentDataFrame.groupby(["identifier"]).mean()
        testStatistic, pValue = ttest(
            np.array(postTreatmentDataFrame["kpi"]),
            np.array(preTreatmentDataFrame["kpi"]),
            equal_var=equal_var
        )
        
        estimate = np.mean(np.array(postTreatmentDataFrame["kpi"]))-np.mean(np.array(preTreatmentDataFrame["kpi"]))
        return {"test statistic": testStatistic
        ,"p-value": pValue
        ,"estimate": estimate}

    def AreaUnderCurveEstimate(self, df, isTreatmentGroup=1, equal_var=True, groupbyidentifier=True):
        """
        Run a T-test on the pre group versus the post group, grouped by the identifier and taking the sum.

        We are comparing the means of each identifier before and after the test.
        """
        postTreatmentDataFrame, preTreatmentDataFrame = self._ttest_with_identifier_aggregation_setup(df,isTreatmentGroup)

        if groupbyidentifier:
            postTreatmentDataFrame = postTreatmentDataFrame.groupby(["identifier"]).sum()
            preTreatmentDataFrame = preTreatmentDataFrame.groupby(["identifier"]).sum()
        testStatistic, pValue = ttest(
            np.array(postTreatmentDataFrame["kpi"]),
            np.array(preTreatmentDataFrame["kpi"]),
            equal_var=equal_var
        )
        
        estimate = np.mean(np.array(postTreatmentDataFrame["kpi"]))-np.mean(np.array(preTreatmentDataFrame["kpi"]))
        return {
            "test statistic": testStatistic,
            "p-value": pValue,
            "estimate": estimate
        }

    # Returns: dictionary with keys "effect" and "confidence interval"
    def calculateDifferenceInDifferences(self, df):
        #dropna is necessary to remove non-pre and post values.
        reg = smf.ols('kpi ~ post + treat + treat*post', data = df.dropna()).fit()
        differenceInDifferencesEstimateIndex = 3
        return {
            "estimate": reg.params[differenceInDifferencesEstimateIndex],
            "confidence interval": list((reg._results.conf_int(alpha=0.05, cols=[differenceInDifferencesEstimateIndex]))[0])
        }
    
    def estimateSummmary(self,df):
        print("Difference in Differences Estimate:")
        print(self.calculateDifferenceInDifferences(df))

        print("")
        print("Treatment T-test:")
        print(self.postPreTTest(df))

        print("")
        print("Area under the Curve Estimate:")
        print(self.AreaUnderCurveEstimate(df))

    @property
    def minTimePointForIdentifierDict(self):
        if self.timeSeriesDataManager == None:
            raise TypeError("Time series data manager must be defined in order to adjust for an identifier's starting time.")
        return self.timeSeriesDataManager.minTimePointForIdentifierDict

    @property
    def mapTimeNormalizedToDateTime(self):
        if self.timeSeriesDataManager == None:
            raise TypeError("Time series data manager must be defined in order to map normalized times back to datetimes.")
        return self.timeSeriesDataManager.mapTimeNormalizedToDateTime

    def attemptTransformToDateTimeWithTSManager(self, arr):
        return [(self.mapTimeNormalizedToDateTime[x] if len(self.mapTimeNormalizedToDateTime) != 0 else x) for x in arr]

    def visualizeParallelLinesAssumption(self,df, yLabel, title):
        dfTreatment = df[(df.treat == 1)].groupby(["timePoint"],as_index=False).mean().sort_values(by=["timePoint"],ascending=True)
        dfComparison = df[(df.treat == 0)].groupby(["timePoint"],as_index=False).mean().sort_values(by=["timePoint"],ascending=True)
        
        def getX(self, df):
            x = df["timePoint"]
            # Fail back to using normalized times.
            if self.timeSeriesDataManager is not None:
                x = self.attemptTransformToDateTimeWithTSManager(df["timePoint"])
            return x

        x1 = getX(self, dfTreatment)
        plt.plot(x1, dfTreatment["kpi"], label="Treatment")
        x2 = getX(self, dfComparison)
        plt.plot(x2, dfComparison["kpi"], label="Comparison")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        if isinstance(x1[0], datetime) and isinstance(x2[0], datetime):
            plt.gca().xaxis.set_major_formatter(matplotlibDates.DateFormatter('%m/%d/%Y'))
            plt.gcf().autofmt_xdate()
        plt.ylabel(yLabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def _calculate_aggregate_treatment_matrix(self, df, aggregate=np.mean):
        theAggregates = [
            aggregate(df[(df.treat == 0) & (df.post == 0)]["kpi"]),
            aggregate(df[(df.treat == 0) & (df.post == 1)]["kpi"]),
            aggregate(df[(df.treat == 1) & (df.post == 0)]["kpi"]),
            aggregate(df[(df.treat == 1) & (df.post == 1)]["kpi"]) 
        ]

        if any([np.isnan(x) for x in theAggregates]):
            raise Exception("There must be at least a single value assigned to each combination of treated/untreated and pre/post.")

        return theAggregates

    def visualizeDifferenceInDifferences(self,df,yLabel,title):
        '''
        1. Set the values of the truth table for treat and pre/post
        2. Plot them in a line chart
        '''
        numSamplesForInterpolation = 2 # num samples for interpolation
        treatmentMatrix = self._calculate_aggregate_treatment_matrix(df, aggregate=np.mean)
        kpiTreat0Post0_mean, kpiTreat0Post1_mean, kpiTreat1Post0_mean, kpiTreat1Post1_mean = treatmentMatrix

        x = np.linspace(0,numSamplesForInterpolation-1,numSamplesForInterpolation)
        treated = np.linspace(kpiTreat1Post0_mean, kpiTreat1Post1_mean,numSamplesForInterpolation )
        comparison = np.linspace(kpiTreat0Post0_mean,kpiTreat0Post1_mean,numSamplesForInterpolation)
        plt.plot(x, treated, label="Treatment")
        plt.plot(x, comparison, label="Comparison")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        
        tickLabels = ["." for x in range(numSamplesForInterpolation)]
        tickLabels[0] = "Pre"
        tickLabels[numSamplesForInterpolation-1] = "Post"
        plt.xticks([x for x in range(numSamplesForInterpolation)],tuple(tickLabels) )

        plt.ylabel(yLabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    SEED_VALUE = 1
    np.random.seed(SEED_VALUE)
    N = 100
    kpi = np.arange(N) * np.random.random(N)
    post = np.round(np.random.random(N)).astype(int)
    treatment = np.round(np.random.random(N)).astype(int)
    timePoint = np.array([x for x in range(N//2)] + [x for x in range (N - N//2)])

    df= pd.DataFrame({"kpi":kpi,"post":post,"treatment":treatment,"timePoint":timePoint})

    identifierColName = "EmailName"
    kpiColName = "CodeAuthoringTimeApproximation"
    #---------------------------------------------------------------------------------------
    #Test 1
    #I generated a working example. Then, I prepended 7 irrelevant values to all data vectors
    #---------------------------------------------------------------------------------------
    
    preWindowSize = 5
    bufferWindowSize = 5
    postWindowSize=5

    obj = DiDDesigner(None,preWindowSize,bufferWindowSize,postWindowSize)
    
    treat1KPITestValues = 7*[50]+[i+1 for i in range(15)]
    treat0KPITestValues = 7*[-10]+[4]*6 + [5]*5 + [10]*4
    kpis = np.array(treat1KPITestValues + treat0KPITestValues)
    treat = [1]*len(treat1KPITestValues) + [0]*len(treat0KPITestValues)
    timePoints = np.array([-32,-30,-28,26,-24,-22,-20]+ list(np.arange(15)-5)+[-32,-30,-28,26,-24,-22,-20]+ list(np.arange(15)-5))
    identifiers = ["satyan"]*7 + ["chbuja"]*3 + ["satyan"]*2 + ["chbuja"]*3 + ["satyan"]*2 +  ["chbuja"]*3 + ["satyan"]*2 +["chbuja"]*11+["satyan"]*11
    minTimePointForIdentifierDict = {"chbuja":0,"satyan":0}
    expectedPostValues = np.array(([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]+[0]*5+[np.nan]*5+[1]*5)*2)
    observedPostValues = np.array(obj.calculatePostValues(timePoints,preWindowSize,bufferWindowSize,postWindowSize,identifiers,minTimePointForIdentifierDict))
    #assert statement was complicated because np.nan!=np.nan
    assert(all([truth1 or truth2 for truth1,truth2 in zip( (expectedPostValues==observedPostValues), [np.isnan(x) and np.isnan(y) for x,y in zip(expectedPostValues,observedPostValues)] )]))
    names=["post0treat1","post0treat0","post1treat1","post1treat0"]
    meanKPIs=[3.0,4.0,13.0,9.0]
    post = observedPostValues

    dfBuilder = {
        "treat":treat,
        "post":post,
        "kpi":kpis,
        "identifier":identifiers,
        "timePoint":timePoints
    }

    df = pd.DataFrame(dfBuilder)
    obj.visualizeDifferenceInDifferences(df,"My KPI","Improvement in KPI Accounting for Inflation")
    obj.visualizeParallelLinesAssumption(df,"My KPI", "KPI - Check Parallel Lines Assumption")

    didOutput = obj.calculateDifferenceInDifferences(df)
    estimatedEffect = didOutput["estimate"]
    confidenceInterval = didOutput["confidence interval"]
    trueEffect = 5.0 # 10-5
    assert(trueEffect==((meanKPIs[2]-meanKPIs[0])-(meanKPIs[3]-meanKPIs[1])))
    assert(np.isclose(estimatedEffect,trueEffect))
    ci = confidenceInterval
    assert(trueEffect > ci[0] and trueEffect < ci[1])

    #---------------------------------------------------------------------------------------
    #Test 2 - Before and After - T-test
    # group by identifier only in the treatment group
    #x is treatment_post, y is treatment_pre
    #---------------------------------------------------------------------------------------
    y= [2.0,4.5]
    x = [12.0,14.5]
    x_bar = np.mean(x)
    n=2
    m=2
    s_x   = np.std(x,ddof=1) #unbiased estimate
    y_bar = np.mean(y)
    s_y   = np.std(y,ddof=1)
    s_p = np.sqrt(((n - 1) * s_x ** 2 + (m - 1) * s_y ** 2) / (n + m - 2.0))
    t = ((x_bar - y_bar) - 0) / (s_p * np.sqrt(1 / n + 1 / m))
    assert(t==obj.postPreTTest(df)["test statistic"])

    #---------------------------------------------------------------------------------------
    #Test 3 - Area Under the Curve Estimation
    #---------------------------------------------------------------------------------------
    y = [1+2+3,4+5]
    x=[11+12+13,14+15]
    x_bar = np.mean(x)
    n=2
    m=2
    s_x   = np.std(x,ddof=1) #unbiased estimate
    y_bar = np.mean(y)
    s_y   = np.std(y,ddof=1)
    s_p = np.sqrt(((n - 1) * s_x ** 2 + (m - 1) * s_y ** 2) / (n + m - 2.0))
    t = ((x_bar - y_bar) - 0) / (s_p * np.sqrt(1 / n + 1 / m))
    assert(t==obj.AreaUnderCurveEstimate(df)["test statistic"])
