#Calculate monthly and annual building performance metrics
import random
import sys
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Define file names

fileName1 = "Output/Perf-Metrics-Case-8-Jan.txt"
fileName2 = "Output/Perf-Metrics-Case-8-Feb.txt"
fileName3 = "Output/Perf-Metrics-Case-8-Mar.txt"
fileName4 = "Output/Perf-Metrics-Case-8-Apr.txt"
fileName5 = "Output/Perf-Metrics-Case-8-May.txt"
fileName6 = "Output/Perf-Metrics-Case-8-Jun.txt"
fileName7 = "Output/Perf-Metrics-Case-8-Jul.txt"
fileName8 = "Output/Perf-Metrics-Case-8-Aug.txt"
fileName9 = "Output/Perf-Metrics-Case-8-Sep.txt"
fileName10 = "Output/Perf-Metrics-Case-8-Oct.txt"
fileName11 = "Output/Perf-Metrics-Case-8-Nov.txt"
fileName12 = "Output/Perf-Metrics-Case-8-Dec.txt"

#Load all data in a matrix
data1 = numpy.loadtxt(fileName1)

TotalSensHeatDemand1 = data1[0]
TotalGasConsumpHeat1 = data1[1]
TotalElecHeatDemand1 = data1[2]
TotalSensCoolDemand1 = data1[3]
TotalElecCoolDemand1 = data1[4]
TotalSensWaterHeatDemand1 = data1[5]
TotalGasConsumpWaterHeat1 = data1[6]
TotalElecDomesticDemand1 = data1[7]
TotalElecProducedPV1 = data1[8]
TotalElecProducedWT1 = data1[9]

data2 = numpy.loadtxt(fileName2)

TotalSensHeatDemand2 = data2[0]
TotalGasConsumpHeat2 = data2[1]
TotalElecHeatDemand2 = data2[2]
TotalSensCoolDemand2 = data2[3]
TotalElecCoolDemand2 = data2[4]
TotalSensWaterHeatDemand2 = data2[5]
TotalGasConsumpWaterHeat2 = data2[6]
TotalElecDomesticDemand2 = data2[7]
TotalElecProducedPV2 = data2[8]
TotalElecProducedWT2 = data2[9]

data3 = numpy.loadtxt(fileName3)

TotalSensHeatDemand3 = data3[0]
TotalGasConsumpHeat3 = data3[1]
TotalElecHeatDemand3 = data3[2]
TotalSensCoolDemand3 = data3[3]
TotalElecCoolDemand3 = data3[4]
TotalSensWaterHeatDemand3 = data3[5]
TotalGasConsumpWaterHeat3 = data3[6]
TotalElecDomesticDemand3 = data3[7]
TotalElecProducedPV3 = data3[8]
TotalElecProducedWT3 = data3[9]

data4 = numpy.loadtxt(fileName4)

TotalSensHeatDemand4 = data4[0]
TotalGasConsumpHeat4 = data4[1]
TotalElecHeatDemand4 = data4[2]
TotalSensCoolDemand4 = data4[3]
TotalElecCoolDemand4 = data4[4]
TotalSensWaterHeatDemand4 = data4[5]
TotalGasConsumpWaterHeat4 = data4[6]
TotalElecDomesticDemand4 = data4[7]
TotalElecProducedPV4 = data4[8]
TotalElecProducedWT4 = data4[9]

data5 = numpy.loadtxt(fileName5)

TotalSensHeatDemand5 = data5[0]
TotalGasConsumpHeat5 = data5[1]
TotalElecHeatDemand5 = data5[2]
TotalSensCoolDemand5 = data5[3]
TotalElecCoolDemand5 = data5[4]
TotalSensWaterHeatDemand5 = data5[5]
TotalGasConsumpWaterHeat5 = data5[6]
TotalElecDomesticDemand5 = data5[7]
TotalElecProducedPV5 = data5[8]
TotalElecProducedWT5 = data5[9]

data6 = numpy.loadtxt(fileName6)

TotalSensHeatDemand6 = data6[0]
TotalGasConsumpHeat6 = data6[1]
TotalElecHeatDemand6 = data6[2]
TotalSensCoolDemand6 = data6[3]
TotalElecCoolDemand6 = data6[4]
TotalSensWaterHeatDemand6 = data6[5]
TotalGasConsumpWaterHeat6 = data6[6]
TotalElecDomesticDemand6 = data6[7]
TotalElecProducedPV6 = data6[8]
TotalElecProducedWT6 = data6[9]

data7 = numpy.loadtxt(fileName7)

TotalSensHeatDemand7 = data7[0]
TotalGasConsumpHeat7 = data7[1]
TotalElecHeatDemand7 = data7[2]
TotalSensCoolDemand7 = data7[3]
TotalElecCoolDemand7 = data7[4]
TotalSensWaterHeatDemand7 = data7[5]
TotalGasConsumpWaterHeat7 = data7[6]
TotalElecDomesticDemand7 = data7[7]
TotalElecProducedPV7 = data7[8]
TotalElecProducedWT7 = data7[9]

data8 = numpy.loadtxt(fileName8)

TotalSensHeatDemand8 = data8[0]
TotalGasConsumpHeat8 = data8[1]
TotalElecHeatDemand8 = data8[2]
TotalSensCoolDemand8 = data8[3]
TotalElecCoolDemand8 = data8[4]
TotalSensWaterHeatDemand8 = data8[5]
TotalGasConsumpWaterHeat8 = data8[6]
TotalElecDomesticDemand8 = data8[7]
TotalElecProducedPV8 = data8[8]
TotalElecProducedWT8 = data8[9]

data9 = numpy.loadtxt(fileName9)

TotalSensHeatDemand9 = data9[0]
TotalGasConsumpHeat9 = data9[1]
TotalElecHeatDemand9 = data9[2]
TotalSensCoolDemand9 = data9[3]
TotalElecCoolDemand9 = data9[4]
TotalSensWaterHeatDemand9 = data9[5]
TotalGasConsumpWaterHeat9 = data9[6]
TotalElecDomesticDemand9 = data9[7]
TotalElecProducedPV9 = data9[8]
TotalElecProducedWT9 = data9[9]

data10 = numpy.loadtxt(fileName10)

TotalSensHeatDemand10 = data10[0]
TotalGasConsumpHeat10 = data10[1]
TotalElecHeatDemand10 = data10[2]
TotalSensCoolDemand10 = data10[3]
TotalElecCoolDemand10 = data10[4]
TotalSensWaterHeatDemand10 = data10[5]
TotalGasConsumpWaterHeat10 = data10[6]
TotalElecDomesticDemand10 = data10[7]
TotalElecProducedPV10 = data10[8]
TotalElecProducedWT10 = data10[9]

data11 = numpy.loadtxt(fileName11)

TotalSensHeatDemand11 = data11[0]
TotalGasConsumpHeat11 = data11[1]
TotalElecHeatDemand11 = data11[2]
TotalSensCoolDemand11 = data11[3]
TotalElecCoolDemand11 = data11[4]
TotalSensWaterHeatDemand11 = data11[5]
TotalGasConsumpWaterHeat11 = data11[6]
TotalElecDomesticDemand11 = data11[7]
TotalElecProducedPV11 = data11[8]
TotalElecProducedWT11 = data11[9]

data12 = numpy.loadtxt(fileName12)

TotalSensHeatDemand12 = data12[0]
TotalGasConsumpHeat12 = data12[1]
TotalElecHeatDemand12 = data12[2]
TotalSensCoolDemand12 = data12[3]
TotalElecCoolDemand12 = data12[4]
TotalSensWaterHeatDemand12 = data12[5]
TotalGasConsumpWaterHeat12 = data12[6]
TotalElecDomesticDemand12 = data12[7]
TotalElecProducedPV12 = data12[8]
TotalElecProducedWT12 = data12[9]

AnnualTotalSensHeatDemand = TotalSensHeatDemand1+TotalSensHeatDemand2+TotalSensHeatDemand3+TotalSensHeatDemand4+TotalSensHeatDemand5+ \
               TotalSensHeatDemand6+TotalSensHeatDemand7+TotalSensHeatDemand8+TotalSensHeatDemand9+TotalSensHeatDemand10+ \
               TotalSensHeatDemand11+TotalSensHeatDemand12

AnnualTotalGasConsumpHeat = TotalGasConsumpHeat1+TotalGasConsumpHeat2+TotalGasConsumpHeat3+TotalGasConsumpHeat4+TotalGasConsumpHeat5+ \
               TotalGasConsumpHeat6+TotalGasConsumpHeat7+TotalGasConsumpHeat8+TotalGasConsumpHeat9+TotalGasConsumpHeat10+ \
               TotalGasConsumpHeat11+TotalGasConsumpHeat12

AnnualTotalElecHeatDemand = TotalElecHeatDemand1+TotalElecHeatDemand2+TotalElecHeatDemand3+TotalElecHeatDemand4+TotalElecHeatDemand5+ \
               TotalElecHeatDemand6+TotalElecHeatDemand7+TotalElecHeatDemand8+TotalElecHeatDemand9+TotalElecHeatDemand10+ \
               TotalElecHeatDemand11+TotalElecHeatDemand12

AnnualTotalSensCoolDemand = TotalSensCoolDemand1+TotalSensCoolDemand2+TotalSensCoolDemand3+TotalSensCoolDemand4+TotalSensCoolDemand5+ \
               TotalSensCoolDemand6+TotalSensCoolDemand7+TotalSensCoolDemand8+TotalSensCoolDemand9+TotalSensCoolDemand10+ \
               TotalSensCoolDemand11+TotalSensCoolDemand12

AnnualTotalElecCoolDemand = TotalElecCoolDemand1+TotalElecCoolDemand2+TotalElecCoolDemand3+TotalElecCoolDemand4+TotalElecCoolDemand5+ \
               TotalElecCoolDemand6+TotalElecCoolDemand7+TotalElecCoolDemand8+TotalElecCoolDemand9+TotalElecCoolDemand10+ \
               TotalElecCoolDemand11+TotalElecCoolDemand12

AnnualTotalSensWaterHeatDemand = TotalSensWaterHeatDemand1+TotalSensWaterHeatDemand2+TotalSensWaterHeatDemand3+TotalSensWaterHeatDemand4+TotalSensWaterHeatDemand5+ \
               TotalSensWaterHeatDemand6+TotalSensWaterHeatDemand7+TotalSensWaterHeatDemand8+TotalSensWaterHeatDemand9+TotalSensWaterHeatDemand10+ \
               TotalSensWaterHeatDemand11+TotalSensWaterHeatDemand12

AnnualTotalGasConsumpWaterHeat = TotalGasConsumpWaterHeat1+TotalGasConsumpWaterHeat2+TotalGasConsumpWaterHeat3+TotalGasConsumpWaterHeat4+TotalGasConsumpWaterHeat5+ \
               TotalGasConsumpWaterHeat6+TotalGasConsumpWaterHeat7+TotalGasConsumpWaterHeat8+TotalGasConsumpWaterHeat9+TotalGasConsumpWaterHeat10+ \
               TotalGasConsumpWaterHeat11+TotalGasConsumpWaterHeat12

AnnualTotalElecDomesticDemand = TotalElecDomesticDemand1+TotalElecDomesticDemand2+TotalElecDomesticDemand3+TotalElecDomesticDemand4+TotalElecDomesticDemand5+ \
               TotalElecDomesticDemand6+TotalElecDomesticDemand7+TotalElecDomesticDemand8+TotalElecDomesticDemand9+TotalElecDomesticDemand10+ \
               TotalElecDomesticDemand11+TotalElecDomesticDemand12

AnnualTotalElecProducedPV = TotalElecProducedPV1+TotalElecProducedPV2+TotalElecProducedPV3+TotalElecProducedPV4+TotalElecProducedPV5+ \
               TotalElecProducedPV6+TotalElecProducedPV7+TotalElecProducedPV8+TotalElecProducedPV9+TotalElecProducedPV10+ \
               TotalElecProducedPV11+TotalElecProducedPV12

AnnualTotalElecProducedWT = TotalElecProducedWT1+TotalElecProducedWT2+TotalElecProducedWT3+TotalElecProducedWT4+TotalElecProducedWT5+ \
               TotalElecProducedWT6+TotalElecProducedWT7+TotalElecProducedWT8+TotalElecProducedWT9+TotalElecProducedWT10+ \
               TotalElecProducedWT11+TotalElecProducedWT12

AnnualTotalGasConsump = AnnualTotalGasConsumpHeat + AnnualTotalGasConsumpWaterHeat
AnnualTotalNetElec = AnnualTotalElecHeatDemand + AnnualTotalElecCoolDemand + AnnualTotalElecDomesticDemand - \
    AnnualTotalElecProducedPV - AnnualTotalElecProducedWT

#Print results to screen "LaTeX Ready"
print("TotalSensHeatDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalSensHeatDemand1, TotalSensHeatDemand2, TotalSensHeatDemand3, TotalSensHeatDemand4, TotalSensHeatDemand5,
               TotalSensHeatDemand6, TotalSensHeatDemand7, TotalSensHeatDemand8, TotalSensHeatDemand9, TotalSensHeatDemand10,
               TotalSensHeatDemand11, TotalSensHeatDemand12, AnnualTotalSensHeatDemand))

print("TotalGasConsumpHeat: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalGasConsumpHeat1, TotalGasConsumpHeat2, TotalGasConsumpHeat3, TotalGasConsumpHeat4, TotalGasConsumpHeat5,
               TotalGasConsumpHeat6, TotalGasConsumpHeat7, TotalGasConsumpHeat8, TotalGasConsumpHeat9, TotalGasConsumpHeat10,
               TotalGasConsumpHeat11, TotalGasConsumpHeat12, AnnualTotalGasConsumpHeat))

print("TotalElecHeatDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalElecHeatDemand1, TotalElecHeatDemand2, TotalElecHeatDemand3, TotalElecHeatDemand4, TotalElecHeatDemand5,
               TotalElecHeatDemand6, TotalElecHeatDemand7, TotalElecHeatDemand8, TotalElecHeatDemand9, TotalElecHeatDemand10,
               TotalElecHeatDemand11, TotalElecHeatDemand12, AnnualTotalElecHeatDemand))

print("TotalSensCoolDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalSensCoolDemand1, TotalSensCoolDemand2, TotalSensCoolDemand3, TotalSensCoolDemand4, TotalSensCoolDemand5,
               TotalSensCoolDemand6, TotalSensCoolDemand7, TotalSensCoolDemand8, TotalSensCoolDemand9, TotalSensCoolDemand10,
               TotalSensCoolDemand11, TotalSensCoolDemand12, AnnualTotalSensCoolDemand))

print("TotalElecCoolDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalElecCoolDemand1, TotalElecCoolDemand2, TotalElecCoolDemand3, TotalElecCoolDemand4, TotalElecCoolDemand5,
               TotalElecCoolDemand6, TotalElecCoolDemand7, TotalElecCoolDemand8, TotalElecCoolDemand9, TotalElecCoolDemand10,
               TotalElecCoolDemand11, TotalElecCoolDemand12, AnnualTotalElecCoolDemand))

print("TotalSensWaterHeatDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalSensWaterHeatDemand1, TotalSensWaterHeatDemand2, TotalSensWaterHeatDemand3, TotalSensWaterHeatDemand4, TotalSensWaterHeatDemand5,
               TotalSensWaterHeatDemand6, TotalSensWaterHeatDemand7, TotalSensWaterHeatDemand8, TotalSensWaterHeatDemand9, TotalSensWaterHeatDemand10,
               TotalSensWaterHeatDemand11, TotalSensWaterHeatDemand12, AnnualTotalSensWaterHeatDemand))

print("TotalGasConsumpWaterHeat: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalGasConsumpWaterHeat1, TotalGasConsumpWaterHeat2, TotalGasConsumpWaterHeat3, TotalGasConsumpWaterHeat4, TotalGasConsumpWaterHeat5,
               TotalGasConsumpWaterHeat6, TotalGasConsumpWaterHeat7, TotalGasConsumpWaterHeat8, TotalGasConsumpWaterHeat9, TotalGasConsumpWaterHeat10,
               TotalGasConsumpWaterHeat11, TotalGasConsumpWaterHeat12, AnnualTotalGasConsumpWaterHeat))

print("TotalElecDomesticDemand: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalElecDomesticDemand1, TotalElecDomesticDemand2, TotalElecDomesticDemand3, TotalElecDomesticDemand4, TotalElecDomesticDemand5,
               TotalElecDomesticDemand6, TotalElecDomesticDemand7, TotalElecDomesticDemand8, TotalElecDomesticDemand9, TotalElecDomesticDemand10,
               TotalElecDomesticDemand11, TotalElecDomesticDemand12, AnnualTotalElecDomesticDemand))

print("TotalElecProducedPV: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalElecProducedPV1, TotalElecProducedPV2, TotalElecProducedPV3, TotalElecProducedPV4, TotalElecProducedPV5,
               TotalElecProducedPV6, TotalElecProducedPV7, TotalElecProducedPV8, TotalElecProducedPV9, TotalElecProducedPV10,
               TotalElecProducedPV11, TotalElecProducedPV12, AnnualTotalElecProducedPV))

print("TotalElecProducedWT: %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f" \
            % (TotalElecProducedWT1, TotalElecProducedWT2, TotalElecProducedWT3, TotalElecProducedWT4, TotalElecProducedWT5,
               TotalElecProducedWT6, TotalElecProducedWT7, TotalElecProducedWT8, TotalElecProducedWT9, TotalElecProducedWT10,
               TotalElecProducedWT11, TotalElecProducedWT12, AnnualTotalElecProducedWT))

print("AnnualTotalGasConsump and AnnualTotalNetElec: %5.2f & %5.2f " \
            % (AnnualTotalGasConsump, AnnualTotalNetElec))