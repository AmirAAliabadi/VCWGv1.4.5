#Calculate monthly building performance metrics and save to file
import random
import sys
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def PostProcess2Serial(Adv_ene_heat_mode, outputFileName):

    #Define file names
    fileName = "Output/BEM_hourly.txt"
    #Define building energy system characteristics
    COP_cooling = 3.13              # COP under cooling mode (typical air conditioning)
    heatEff = 0.95                  # Thermal efficiency of a natural gas furnace or hot water heater
    HeatingValue = 37000            # Energy in a cubic meter of natural gas [kJ m^-3]
    SpinUpDays = 3                  # Number of days to ignore data
    SpinUpHours = SpinUpDays * 24   # Number of hours to ignore data

    #Load all data in a matrix
    data = numpy.loadtxt(fileName)

    Hour = data[:,0]
    sensWaste = data[:,1]
    dehumDemand = data[:,2]
    QWater = data[:,3]
    QGas = data[:,4]
    sensCoolDemand = data[:,5]
    coolConsump = data[:,6]
    sensHeatDemand = data[:,7]
    heatConsump = data[:,8]
    Q_st = data[:,9]
    Q_he_st = data[:,10]
    Q_bites = data[:,11]
    Q_hp = data[:,12]
    Q_recovery = data[:,13]
    W_hp = data[:,14]
    W_pv = data[:,15]
    COP_hp = data[:,16]
    indoorTemp = data[:,17]
    T_st_f_i = data[:,18]
    T_st_f_o = data[:,19]
    T_he_st_i = data[:,20]
    T_he_st_o = data[:,21]
    T_bites = data[:,22]
    W_wt = data[:,23]
    f_pcm = data[:,24]
    Q_waterSaved = data[:,25]
    sensWaterHeatDemand = data[:,26]
    Q_ground = data[:,27]
    elecDomesticDemand = data[:,28]
    Q_waterRecovery = data[:,29]

    # Ne renewable energy system
    if Adv_ene_heat_mode == 2:

        # heating demand is partitioned into Q_hp or sensible heating demand
        # so to calculate total building sensible heating demand they must be added
        TotalSensHeatDemand = (numpy.sum(sensHeatDemand[SpinUpHours:])) / 1000
        TotalGasConsumpHeat = ((numpy.sum(sensHeatDemand[SpinUpHours:])) / 1000) * 3600 / heatEff / HeatingValue
        TotalElecHeatDemand = 0

        TotalSensCoolDemand = numpy.sum(sensCoolDemand[SpinUpHours:])/1000
        TotalElecCoolDemand = TotalSensCoolDemand/COP_cooling

        # sensible water heating demand is partitioned into Q_waterSaved or sensible water heating demand
        # so to calculate total building sensible water heating demand they must be added
        TotalSensWaterHeatDemand = (numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000
        TotalGasConsumpWaterHeat = ((numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000) * 3600 / heatEff / HeatingValue

        TotalElecDomesticDemand = numpy.sum(elecDomesticDemand[SpinUpHours:]) / 1000
        TotalElecProducedPV = 0
        TotalElecProducedWT = 0

    # Renewable energy system under heating mode
    if Adv_ene_heat_mode == 1:

        # heating demand is partitioned into Q_hp or sensible heating demand
        # so to calculate total building sensible heating demand they must be added
        TotalSensHeatDemand = (numpy.sum(sensHeatDemand[SpinUpHours:])) / 1000 + (numpy.sum(Q_hp[SpinUpHours:])) / 1000
        TotalGasConsumpHeat = ((numpy.sum(sensHeatDemand[SpinUpHours:])) / 1000) * 3600 / heatEff / HeatingValue
        TotalElecHeatDemand = numpy.sum(W_hp[SpinUpHours:]) / 1000

        TotalSensCoolDemand = numpy.sum(sensCoolDemand[SpinUpHours:])/1000
        TotalElecCoolDemand = TotalSensCoolDemand/COP_cooling

        # sensible water heating demand is partitioned into Q_waterSaved or sensible water heating demand
        # so to calculate total building sensible water heating demand they must be added
        TotalSensWaterHeatDemand = (numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000 + (numpy.sum(Q_waterSaved[SpinUpHours:])) / 1000
        TotalGasConsumpWaterHeat = ((numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000) * 3600 / heatEff / HeatingValue

        TotalElecDomesticDemand = numpy.sum(elecDomesticDemand[SpinUpHours:]) / 1000
        TotalElecProducedPV = numpy.sum(W_pv[SpinUpHours:]) / 1000
        TotalElecProducedWT = numpy.sum(W_wt[SpinUpHours:]) / 1000

    # Renewable energy system under cooling mode
    elif Adv_ene_heat_mode == 0:

        TotalSensHeatDemand = numpy.sum(sensHeatDemand[SpinUpHours:]) / 1000
        TotalGasConsumpHeat = (TotalSensHeatDemand * 3600 / heatEff) / HeatingValue
        TotalElecHeatDemand = 0

        # cooling demand is entirely partitioned into Q_hp or sensible cooling demand
        # so to calculate total building sensible cooling demand they must be added
        TotalSensCoolDemand = (numpy.sum(sensCoolDemand[SpinUpHours:])) / 1000 + (numpy.sum(Q_hp[SpinUpHours:])) / 1000
        TotalElecCoolDemand = numpy.sum(coolConsump[SpinUpHours:]) / 1000 + numpy.sum(W_hp[SpinUpHours:]) / 1000

        # sensible water heating demand is partitioned into Q_waterSaved or sensible water heating demand
        # so to calculate total building sensible water heating demand they must be added
        TotalSensWaterHeatDemand = (numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000 + (numpy.sum(Q_waterSaved[SpinUpHours:])) / 1000
        TotalGasConsumpWaterHeat = ((numpy.sum(sensWaterHeatDemand[SpinUpHours:])) / 1000) * 3600 / heatEff / HeatingValue

        TotalElecDomesticDemand = numpy.sum(elecDomesticDemand[SpinUpHours:]) / 1000
        TotalElecProducedPV = numpy.sum(W_pv[SpinUpHours:]) / 1000
        TotalElecProducedWT = numpy.sum(W_wt[SpinUpHours:]) / 1000

    outputFile = open(outputFileName, "w")
    outputFile.write("#0: TotalSensHeatDemand [kW hr] \t 1: TotalGasConsumpHeat [m3] \t 2: TotalElecHeatDemand [kW hr] \t \
                            3: TotalSensCoolDemand [kW hr] \t 4: TotalElecCoolDemand [kW hr] \t \
                            5: TotalSensWaterHeatDemand [kW hr] \t 6: TotalGasConsumpWaterHeat [m3] \t \
                            7: TotalElecDomesticDemand [kW hr] \t 8: TotalElecProducedPV [kW hr] \t 9: TotalElecProducedWT [kW hr] \n")

    outputFile.write("%f \t %f \t %f \t \
                        %f \t %f \t \
                        %f \t %f \t \
                        %f \t %f \t %f \n"
        % (TotalSensHeatDemand, TotalGasConsumpHeat, TotalElecHeatDemand, \
           TotalSensCoolDemand, TotalElecCoolDemand, \
           TotalSensWaterHeatDemand, TotalGasConsumpWaterHeat, \
           TotalElecDomesticDemand, TotalElecProducedPV, TotalElecProducedWT))

    outputFile.close()