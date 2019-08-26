# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:02:53 2016

@author: henry

pip install git+git://github.com/jradavenport/cubehelix.git
"""

'''
except Exception as ex:
        print(str(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys
import scipy.interpolate
import string
import random
from collections import Counter
import datetime
import scipy.constants
from scipy.integrate import odeint
import numbers
from matplotlib import colors as mplcol
from matplotlib.pyplot import colorbar
import matplotlib
import copy
import math

if os.environ["PATH"][0] == "C" or os.environ["PATH"][0] == "D":
    pathh = "D:/hesten-bec/"
    sys.path.insert(0, pathh)
    import code_.cubehelix as cubehelix
else:
    import cubehelix

    pathh = "/home/henry/storage/hesten-bec/"
    sys.path.insert(0, pathh)


# isinstance(x, numbers.Number)

def defaultPath():
    try:
        return os.environ["myPythonPath"]
    except:
        return pathh


# in T rad/s
def getRealParams():
    cavity_mode_decay = 1e-3
    G_Down = 0.25e-3
    omega_zpl = 3458.
    omega_gs = 3195.
    Omega_HO = 314e-3
    photon_mass = 8e-36  # Kg

    length_HO = (scipy.constants.hbar / photon_mass / Omega_HO) ** 0.5  # m

    return locals()


'''
# drop is the fractional change in gradient which gives the width
def thresholdFromPop(gup, pop, minCurveWarning=1, drop=1/2., applyLogs=True, plots=False):
    ret = noneClass()
    try:
        if(applyLogs):
            gup = np.log10(gup)
            pop = np.log10(pop)
            
        gradX,gradd = myGradient(gup, pop)
        ggX,gradOfGrad = myGradient(gradX, gradd)
        
        if(plots):
            plt.plot(gup,pop/max(pop), label="pop, max: {}".format(max(pop)))
            plt.plot(gradX,gradd/max(gradd), label="grad, max: {}".format(max(gradd)))
            plt.plot(ggX,gradOfGrad/max(abs(np.array(gradOfGrad))), label="grad of grad, max: {}".format(max(abs(np.array(gradOfGrad)))))
            plt.legend(loc="best")
        
        maxInd,maxVal = max(enumerate(gradOfGrad), key=lambda x:x[1])
        if(maxVal<minCurveWarning):
            print("Curvature, {} is smaller than cutoff: {}".format(maxVal,minCurveWarning))
            return noneClass("Curvature, {} is smaller than cutoff: {}".format(maxVal,minCurveWarning))
        if(plots):
            print("maxInd: {}  maxVal: {}".format(maxInd,maxVal))
        
        gradientThresholdGup = noneClass()
        threshIIsh = None
        for i in range(maxInd,len(gradOfGrad)):
            if(gradOfGrad[i]<0 and gradientThresholdGup==None):
                if(i>0):
                    threshIIsh = i
                    gradientThresholdGup = ggX[i-1] + (ggX[i]-ggX[i-1])*(gradOfGrad[i-1])/(gradOfGrad[i-1]-gradOfGrad[i])
                break;
        if(plots):
            print("gradientThresholdGup: {}".format(gradientThresholdGup))
                
        halfWidth=noneClass()
        if(gradientThresholdGup!=None):
            threshGrad = scipy.interpolate.interp1d(gradX,gradd)(gradientThresholdGup)
            halfGup = scipy.interpolate.interp1d(gradd[:threshIIsh],gradX[:threshIIsh])(threshGrad*drop)
            halfWidth = gradientThresholdGup-halfGup        
        ret = [gradientThresholdGup, halfWidth]
    except:
        print("threshold exception")
        ret = noneClass()
    return ret
'''


# drop is the fractional change in gradient which gives the width
def thresholdFromPop(gup, pop, minAllowedGradRatio=2, drop=1 / 2., applyLogs=True, plots=False, tracePrint=False,
                     errors=True, minAllowedGrad=None, minAllowedP2PDiff=None):
    if (tracePrint): print("thresholdFromPop a"); sys.stdout.flush()
    ret = noneClass()
    try:
        if (tracePrint): print("thresholdFromPop b"); sys.stdout.flush()
        if None in set(pop):
            print("interpolating over {} None values".format(uniqC(pop)[None]))
        pop = interpOverNone(pop)
        if (applyLogs):
            lgup = np.log10(gup)
            lpop = np.log10(pop)
        else:
            lgup = gup
            lpop = pop

        if (tracePrint): print("thresholdFromPop c"); sys.stdout.flush()
        gradX, gradd = myGradient(lgup, lpop)
        ggX, gradOfGrad = myGradient(gradX, gradd)
        if (tracePrint): print("thresholdFromPop d"); sys.stdout.flush()

        ### Using maximum gradient
        maxInd, maxGrad = max(enumerate(gradd), key=lambda x: x[1])
        try:
            if (maxGrad / np.mean(gradd[:maxInd]) < minAllowedGradRatio):
                err = "Gradient Ratio, {} is smaller than cutoff: {}".format(maxGrad / np.mean(gradd[:maxInd]),
                                                                             minAllowedGradRatio)
                if (errors): print(err)
                maxInd = noneClass(err)
        except:
            print("Threshold: It is likely that the average gradient is 0")
        if (minAllowedGrad != None):
            if (maxGrad < minAllowedGrad):
                err = "Gradient {} smaller than minAllowedGrad {}".format(maxGrad, minAllowedGrad)
                if (errors): print(err)
                maxInd = noneClass(err)
        if (minAllowedP2PDiff != None):
            # for sh in [-2,-1,0,1,2]:
            #    print(sh, lpop[maxInd+1+sh]-lpop[maxInd+sh]
            if (maxInd + 1 == len(lpop) or maxInd == None):
                maxInd = noneClass()
            else:
                if (lpop[maxInd + 1] - lpop[maxInd] < minAllowedP2PDiff):
                    err = "Ratio between the start and end of threshold {} is less than minAllowedP2PDiff {}".format(
                        lpop[maxInd + 1] - lpop[maxInd], minAllowedP2PDiff)
                    if (errors): print(err)
                    maxInd = noneClass(err)

        if (plots and errors):
            print("maxInd: {}  maxGrad: {}".format(maxInd, maxGrad))
        if (tracePrint): print("thresholdFromPop e")

        try:
            if (tracePrint): print("thresholdFromPop f"); sys.stdout.flush()
            w1 = abs(1 / (gradd[maxInd] - gradd[maxInd + 1]))
            w2 = abs(1 / (gradd[maxInd] - gradd[maxInd - 1]))
            thresholdGupA = ((gradX[maxInd] + gradX[maxInd + 1]) * w1 / 2. + (
                        gradX[maxInd] + gradX[maxInd - 1]) * w2 / 2.) / (w1 + w2)
            if (tracePrint): print("thresholdFromPop g"); sys.stdout.flush()
        except:
            if (tracePrint): print("thresholdFromPop h"); sys.stdout.flush()
            thresholdGupA = noneClass()

        if (tracePrint): print("thresholdFromPop i"); sys.stdout.flush()
        halfWidthA = noneClass("thresholdGupA=None")
        if (thresholdGupA != None):
            try:
                if (tracePrint): print("thresholdFromPop ib"); sys.stdout.flush()
                threshGrad = myInterpolate(gradX, gradd)(thresholdGupA)
                if (tracePrint): print("thresholdFromPop ic"); sys.stdout.flush()
                halfGup = myInterpolate(gradd[maxInd:], gradX[maxInd:])(threshGrad * drop)
                if (tracePrint): print("thresholdFromPop id"); sys.stdout.flush()
                halfWidthA = halfGup - thresholdGupA
            except:
                print("ThreshWidthA calculation failed")
                print("threshGrad*drop: {}".format(threshGrad * drop))
                print("maxInd:{} len:{} lenX:{}".format(maxInd, len(gradd), len(gradX)))
                print("gradd[maxInd:]\n")
                print(gradd[maxInd:])
                print("\ngradX[maxInd:]\n")
                print(gradX[maxInd:])
                print("\n\n")
                halfWidthA = noneClass("ThreshWidthA calculation failed")
        if (tracePrint): print("thresholdFromPop j"); sys.stdout.flush()

        ### Using maxima for the largest Gup
        maxInd2 = noneClass()
        for i, gg in list(enumerate(gradOfGrad))[::-1]:
            if (tracePrint): print("thresholdFromPop k {} {}".format(i, gg)); sys.stdout.flush()
            if (minAllowedGrad != None):
                if (gradOfGrad[i] < minAllowedGrad):
                    continue
            if (minAllowedP2PDiff != None):
                # print("{}  {}  {}  {}".format(gg,lpop[i+1]-lpop[i],lpop[i]-lpop[i-1],lpop[i+2]-lpop[i+1])
                if (lpop[i + 2] - lpop[i + 1] < minAllowedP2PDiff):
                    continue
            try:
                if (gg > 0 and gradd[i] / np.mean(gradd[:i]) > minAllowedGradRatio):
                    maxInd2 = i
                    break
            except:
                pass

        if (errors): print("maxInd2", maxInd2); sys.stdout.flush()
        try:
            if (tracePrint): print("thresholdFromPop l"); sys.stdout.flush()
            thresholdGupB = ggX[maxInd2] + (ggX[maxInd2 + 1] - ggX[maxInd2]) * (gradOfGrad[maxInd2]) / (
                        gradOfGrad[maxInd2] - gradOfGrad[maxInd2 + 1])
        except:
            thresholdGupB = noneClass()
        if (tracePrint): print("thresholdFromPop m"); sys.stdout.flush()

        halfWidthB = noneClass()
        if (thresholdGupB != None):
            try:
                if (tracePrint): print("thresholdFromPop n"); sys.stdout.flush()
                threshGrad = myInterpolate(gradX, gradd)(thresholdGupB)
                if (tracePrint): print("thresholdFromPop n1"); sys.stdout.flush()
                halfGup = myInterpolate(gradd[maxInd2:], gradX[maxInd2:])(threshGrad * drop)
                if (tracePrint): print("thresholdFromPop n2"); sys.stdout.flush()
                halfWidthB = halfGup - thresholdGupB
                if (tracePrint): print("thresholdFromPop n3"); sys.stdout.flush()
            except:
                print("ThreshWidthB calculation failed")
                print("threshGrad*drop: {}".format(threshGrad * drop))
                print("maxInd2:{} len:{} lenX:{}".format(maxInd2, len(gradd), len(gradX)))
                print("gradd[maxInd2:]\n")
                print(gradd[maxInd2:])
                print("\ngradX[maxInd2:]\n")
                print(gradX[maxInd2:])
                print("\n\n")
                halfWidthB = noneClass("ThreshWidthB calculation failed")
        if (tracePrint): print("thresholdFromPop o"); sys.stdout.flush()

        meanInd = maxInd
        if (meanInd == None):
            meanInd = len(gradd) - 1

        threshStep = None
        '''
        cutEst = None
        if(  (lpop[-1]-lpop[-2])/(lgup[-1]-lgup[-2]) < 0.1* (lpop[-1] - lpop[0])/(lgup[-1]-lgup[0]) ):
            cutEst = lgup[-2]
        if(thresholdGupB!=None):
            threshEst=thresholdGupB
        elif(thresholdGupA!=None):
            threshEst=thresholdGupA
        else:
            threshEst=None
            
        #cutEst = -4 #################
        
        if(tracePrint): print("thresholdFromPop p"); sys.stdout.flush()
        threshStep = thresholdFromPopStepFunc(lgup,lpop,applyLogs=False,cutEst=cutEst, threshEst=threshEst, tracePrint=False)
        '''

        if (tracePrint): print("thresholdFromPop q"); sys.stdout.flush()
        if (plots):
            plt.plot(lgup, lpop / max(abs(lpop)), label="lpop, max: {}".format(max(lpop)))
            plt.plot(gradX, gradd / max(abs(array(gradd))), label="grad, max: {}".format(max(gradd)))
            plt.plot(ggX, gradOfGrad / max(abs(np.array(gradOfGrad))),
                     label="grad of grad, max: {}".format(max(abs(np.array(gradOfGrad)))))
            # plt.plot(lgup,threshStep["fit_points"]/max(lpop),label="Fit of step function")
            plt.plot(gradX, [np.mean(gradd[:meanInd]) / max(abs(array(gradd)))] * len(gradX),
                     label="Mean Grad: {}".format(np.mean(gradd)))
            plt.legend(loc="best")

            # plt.ylim([-1,1])

        ret = [threshStep, thresholdGupA, thresholdGupB, halfWidthA, halfWidthB, maxGrad / np.mean(gradd[:meanInd]),
               max(abs(np.array(gradOfGrad)))]
    except Exception as ex:
        print("threshold exception")
        print(str(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        ret = noneClass(ex)
    return ret


def thresholdFromPopStepFunc(gup, pop, threshEst=-4, cutEst=None, applyLogs=True, plots=False, tracePrint=False):
    if (tracePrint): print("thresholdFromPopStepFunc a"); sys.stdout.flush()

    def lin2flat(x, a1, b1):
        if (isListOrArray(x)):
            return np.array(map(lambda y: lin2flat(y, a1, b1), x))
        ret = (x - (np.log(np.cosh(a1 * (x - b1))) + np.log(2)) / a1) / 2.
        if (ret == np.inf or ret == -np.inf):
            if (x > b1):
                ret = 0
            else:
                ret = x
        return ret

    def fitFunc(x, c1, m, a2, b2, c2, a3, b3):
        return c1 + c2 / (np.exp(-a2 * (x - b2)) + 1) + m * lin2flat(x, a3, b3)

    def fitFuncThresh(x, c, m, a2, b2, c2):
        return c + m * x + c2 / (np.exp(-a2 * (x - b2)) + 1)

    def fitFuncSat(x, c, m, a3, b3):
        return c + m * lin2flat(x, a3, b3)

    if (tracePrint): print("thresholdFromPopStepFunc b"); sys.stdout.flush()

    ret = noneClass()
    try:
        if (applyLogs):
            lgup = np.log10(gup)
            lpop = np.log10(pop)
        else:
            lgup = gup
            lpop = pop

        if (tracePrint): print("thresholdFromPopStepFunc c1"); sys.stdout.flush()
        if (cutEst == None):
            cutEst = lgup[-1]
        if (threshEst == None):
            threshEst = -4
        print("threshEst", threshEst)
        print("cutEst", cutEst)
        if (tracePrint): print("thresholdFromPopStepFunc c2"); sys.stdout.flush()
        im = (lpop[-1] - lpop[0]) / (lgup[-1] - lgup[0])
        if (tracePrint): print("thresholdFromPopStepFunc c3"); sys.stdout.flush()
        ic = im * lgup[0]
        if (tracePrint): print("thresholdFromPopStepFunc c4"); sys.stdout.flush()
        p0 = (ic, im, 10, threshEst, 1, 1, cutEst)
        if (tracePrint): print("thresholdFromPopStepFunc c5"); sys.stdout.flush()
        print(p0)
        # p0 = (3.3867347,1.12435869,10.04,-3.672,0,3.27,-4)
        # p0=(4.5,1,10,-4,1,100,10)

        if (tracePrint): print("thresholdFromPopStepFunc d"); sys.stdout.flush()
        lbound = [ic - abs(ic) * 0.1, 0, -np.inf, -np.inf, 0, -np.inf, -np.inf]
        ubound = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        popt, pcov = scipy.optimize.curve_fit(fitFunc, lgup, lpop, p0=p0, bounds=(lbound, ubound))
        perr = np.sqrt(np.diag(pcov))

        if (tracePrint): print("thresholdFromPopStepFunc e"); sys.stdout.flush()
        keys = ["constant", "gradient", "threshold_width", "threshold_log_power", "threshold_height",
                "saturation_width", "saturation_log_power"]
        ret = {k: v for k, v in zip(keys, popt)}
        errs = {k: v for k, v in zip(map(lambda x: x + "_sd", keys), perr)}
        ret.update(errs)

        fitPoints = fitFunc(lgup, *popt)
        ret["fit_points"] = fitPoints
        if (tracePrint): print("thresholdFromPopStepFunc f"); sys.stdout.flush()
        if (plots):
            if (tracePrint): print("thresholdFromPopStepFunc f1"); sys.stdout.flush()
            plt.figure()
            if (tracePrint): print("thresholdFromPopStepFunc f2"); sys.stdout.flush()
            plt.plot(lgup, lpop, label="Data")
            if (tracePrint): print("thresholdFromPopStepFunc f3"); sys.stdout.flush()
            plt.plot(lgup, fitPoints, label="Fit")
            if (tracePrint): print("thresholdFromPopStepFunc f4"); sys.stdout.flush()
            plt.legend()
            if (tracePrint): print("thresholdFromPopStepFunc f5"); sys.stdout.flush()
        if (tracePrint): print("thresholdFromPopStepFunc g"); sys.stdout.flush()

    except Exception as ex:
        if (tracePrint): print("thresholdFromPopStepFunc h"); sys.stdout.flush()
        print("step threshold exception")
        print(ex)
        ret = noneClass(ex)
        if (tracePrint): print("thresholdFromPopStepFunc i"); sys.stdout.flush()

    if (tracePrint): print("thresholdFromPopStepFunc j"); sys.stdout.flush()
    return ret


## Populations and gup are NOT log scale
def thresholdFromPopsRelative(gup, pops, logGup=True, plots=False, tracePrint=False, minAllowedMax=1e-2):
    if (tracePrint): print("thresholdFromPopRel a"); sys.stdout.flush()
    ret = noneClass()
    try:
        if (tracePrint): print("thresholdFromPopRel b"); sys.stdout.flush()
        popsT = np.array(pops).transpose()
        normsT = np.array(map(lambda x: x / sum(x), popsT))
        norms = normsT.transpose()

        if (tracePrint): print("thresholdFromPopRel c"); sys.stdout.flush()
        ret = map(lambda x: thresholdFromPopRelative(gup, norms[x], logGup=logGup, plots=plots, tracePrint=tracePrint,
                                                     minAllowedMax=minAllowedMax, labell=x), range(len(norms)))

    except Exception as ex:
        print("thresholdFromPopsRelative exception")
        print(ex)
        ret = noneClass(ex)
    return ret


def thresholdFromPopRelative(gup, norm, logGup=True, plots=False, tracePrint=False, minAllowedMax=1e-2, labell=None):
    # print(labell)
    if (tracePrint): print("thresholdFromPopRel a"); sys.stdout.flush()
    ret = noneClass()
    try:
        if (logGup):
            gup = np.log10(gup)

        if (tracePrint): print("thresholdFromPopRel b"); sys.stdout.flush()
        gradX, gradd = myGradient(gup, norm)
        ggX, gradOfGrad = myGradient(gradX, gradd)

        if (tracePrint): print("thresholdFromPopRel c"); sys.stdout.flush()
        maxInd, maxGrad = max(enumerate(gradd), key=lambda x: x[1])
        if (tracePrint): print("maxGrad: {}".format(maxGrad))
        if (maxGrad > minAllowedMax):
            ret = gradX[maxInd]
        else:
            maxInd = len(gradd) - 1
            ret = noneClass("Maximum gradient is too small {}<{}".format(maxGrad, minAllowedMax))

        if (tracePrint): print("thresholdFromPopRel d"); sys.stdout.flush()

        if (plots):
            if (labell == None):
                labell = randomWord()
            plt.figure(labell)
            plt.plot(gup, norm, label="norm, max: {}".format(max(norm)))
            plt.plot(gradX, gradd / max(abs(np.array(gradd))), label="grad, max: {}".format(max(abs(np.array(gradd)))))
            plt.plot(ggX, gradOfGrad / max(abs(np.array(gradOfGrad))),
                     label="grad of grad, max: {}".format(max(abs(np.array(gradOfGrad)))))
            plt.plot(gradX[:maxInd + 1] + gradX[maxInd:], [-1] * (maxInd + 1) + [1] * (len(gradX) - maxInd))

            # plt.plot(lgup,threshStep["fit_points"]/max(lpop),label="Fit of step function")
            # plt.plot(gradX,[np.mean(gradd[:meanInd])/max(gradd)]*len(gradX), label="Mean Grad: {}".format(np.mean(gradd)))
            plt.legend(loc="best")


    except Exception as ex:
        print("thresholdFromPopRelative exception")
        print(ex)
        ret = noneClass(ex)
    return ret


def phaseFromPop(modePop):
    modePopT = np.array(modePop).transpose()

    arrOfDict = map(phaseFromPopSingle, modePopT)

    return transposeDictionary(arrOfDict)


'''
def phaseFromPopSingle(singlePop):
    singlePop = np.array(singlePop)
    NN = sum(singlePop)
    
    aboveTh = singlePop > np.sqrt(NN)
    
    if(aboveTh[0]):
        phase = sum(aboveTh)
    else:
        if(sum(aboveTh) == 0):
            phase = 0
        else:
            phase = -1
            
    if(NN==0):
        participation = 1
    else:
        p_i = np.array(singlePop/NN)
        invParticipation = sum(p_i**2)  # range from 1 to 1/d
        participation = ( 1/invParticipation -1 )/( len(singlePop) -1 )  # range from 0 to 1
    
    return {"phase":phase, "participation":participation, "modesAboveThreshold":aboveTh, "totalPhotons":NN}
    '''


def thermalParticipation2D(tFreq, modeSpacing, trunc=100, mu=None):
    if mu == None:
        mu = -tFreq * 10
    pop = []
    for lvl in range(trunc):
        pop += [BEDist(lvl * modeSpacing, mu, tFreq)] * (lvl + 1)
    return calcParticipation(pop)


def calcParticipation(modePopT):
    if modePopT == None:
        return None
    sh = np.shape(modePopT)
    if len(sh) > 1:
        return map(calcParticipation, modePopT)

    modePopT = np.array(modePopT)
    NN = sum(modePopT)

    p = sum((modePopT / NN) ** 2)

    return 1 / p


def phaseFromThreshold(thresholds, useMethod=None):
    if (useMethod == None):
        thresh = thresholds
    else:
        thresh = map(lambda x: x[useMethod], thresholds)
    if (thresh[0] == None):
        for t in thresh[1:]:
            if (t != None):
                return "non_gs_condensate"
        return "no_thresholds"
    else:
        for t in thresh[1:]:
            if (t != None):
                return "multimode"
        return "single_mode"


## -1 == laser
## 0 == no condensates
## 1 == BEC
## n>=2 == n modes above threshold including GS
def phaseFromThresholdAndPump(thresholds, pump, logPump=False, useMethod=None, tracePrint=False):
    if (useMethod == None):
        thresh = thresholds
    else:
        thresh = map(lambda x: x[useMethod], thresholds)

    if (logPump):
        pump = np.log10(pump)

    if (tracePrint): print(pump, thresh[0], thresh); sys.stdout.flush()
    if (pump < thresh[0]):
        if (tracePrint): print("if"); sys.stdout.flush()
        for th in thresh:
            if (pump >= th):
                return -1
        return 0
    else:
        if (tracePrint): print("else"); sys.stdout.flush()
        ret = 0
        for th in thresh:
            if (pump >= th):
                ret += 1
        return ret


####################################################

def uniqC(lst, recure=True, cnt=None):
    if (cnt == None):
        cnt = Counter()
    for elem in lst:
        if (isListOrArray(elem) and recure):
            uniqC(elem, recure, cnt)
        else:
            try:
                cnt[elem] += 1
            except:
                cnt["unhashable"] += 1
    return cnt


def unique(iterable):
    seen = set()
    for x in iterable:
        if str(x) in seen:
            continue
        seen.add(str(x))
        yield x


def addDict(q, w):
    e = {}
    e.update(q)
    e.update(w)
    for k in q:
        if (k in w):
            e[k] = q[k] + w[k]
    return e


# single map has level 1
def myMap(func, obj, level=np.inf, skipNone=False, returnArray=False):
    if (level != 0):
        if isListOrArray(obj):
            tmp = map(lambda x: myMap(func, x, level - 1, skipNone=skipNone, returnArray=returnArray), obj)
            if returnArray:
                return np.array(tmp)
            else:
                return tmp

        if isinstance(obj, dict):
            return {k: myMap(func, v, level - 1, skipNone=skipNone, returnArray=returnArray) for k, v in obj.items()}

    if (skipNone and obj == None):
        return None
    return func(obj)


def myMapWithNone(func, arr, returnArray=False):
    ret = []
    for elem in arr:
        if elem == None:
            ret.append(None)
        else:
            ret.append(func(elem))

    if returnArray:
        return array(ret)
    else:
        return ret


def myConvertNotNumberToVal(obj, val, returnArray=False):
    def mapable(x):
        if (not isinstance(x, numbers.Number)) or np.isnan(x):
            return val
        else:
            return x

    return myMap(mapable, obj, returnArray=returnArray)


def myMax2(arr, lvl):
    if lvl == 1:
        return max(arr)
    else:
        return max(map(lambda x: myMax2(x, lvl - 1), arr))


def myMin2(arr, lvl):
    if lvl == 1:
        return min(arr)
    else:
        return min(map(lambda x: myMin2(x, lvl - 1), arr))


def myMax(obj, func=np.max, useAbs=False):
    return myReduce(obj, func=func, useAbs=False, ignoreNone=True)


def myMin(obj, useAbs=False):
    return myReduce(obj, func=np.min, useAbs=False, ignoreNone=True)


def myReduce(obj, func, useAbs=False, ignoreNone=False):
    arr = None
    if isListOrArray(obj):
        arr = map(lambda x: myReduce(x, func=func, useAbs=useAbs, ignoreNone=ignoreNone), obj)
    if isinstance(obj, dict):
        arr = map(lambda x: myReduce(x, func=func, useAbs=useAbs, ignoreNone=ignoreNone), obj.values())

    if (arr != None):
        if (ignoreNone):
            arr = [x for x in arr if (x is not None and not np.isnan(x))]
        return func(arr)

    if (useAbs):
        return np.abs(obj)
    else:
        return obj


def myMaxInd(arr, returnMax=False):
    return mySearchListND(arr, max(np.ravel(arr)), returnVals=returnMax)


def discrete_cmap(N, base_cmap=None):
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    if base_cmap == None:
        base_cmap = cubehelix.cmap(start=0.5, rot=-1.5, minSat=1.2, maxSat=1.2, minLight=0.1, maxLight=0.9, gamma=1)

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# Saves current figure as path/name.pdf and appends all the codeFiles
# path must be from / not ~
# no not escape spaces in path name
def mySaveFig(name, path, figg=None, appendd=True, dataDict=None, tabs=20, extraTypes=[], *codeFiles):
    if not os.path.exists(path):
        os.makedirs(path)

    namm = path + "/" + name
    nam = namm + ".pdf"
    if (figg == None):
        plt.savefig(nam)
        for typ in extraTypes:
            plt.savefig(namm + "." + typ)
    else:
        figg.savefig(nam)
        for typ in extraTypes:
            figg.savefig(namm + "." + typ)

    if (not appendd):
        return True

    print("written pdf, now appending")
    f = open(nam, 'a')

    f.write("\n\n" + time.strftime("%d/%m/%Y %H:%M:%S") + "\n\n")
    for code in codeFiles:
        f.write("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        f.write("\nCode File " + code + "\n")
        f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f2 = open(code)
        f.write(f2.read())
        f2.close()
    f.close()

    def myLen(q):
        if (isinstance(q, np.ndarray) or isinstance(q, list)):
            return len(q) * myLen(q[0])
        else:
            return 1

    if (dataDict != None):
        print("Writing data")
        mySaveDicc(dataDict, namm, diccPathh=defaultPath(), timeStamp=False, wordStamp=False, filterNoneClass=True,
                   reduceUniformArrays=True, filterFunctionss=True, saveSmallLength=0)

        '''
        dataDict = filterFunctions(dataDict)
        
        nam2 = path+"/"+name+".data"
        np.save(nam2,dataDict)
        sortedKeys = sorted(dataDict.keys(),key=lambda k: myLen(dataDict[k]))
        nam3 = path+"/"+name+".out"
        f3 = open(nam3,'w')
        for key in sortedKeys:
            if(isinstance(dataDict[key],np.ndarray) or isinstance(dataDict[key],list)):
                f3.write("################\n")
                f3.write("### "+str(key)+" ###\n")
                f3.write("################\n")
                f3.write(str(dataDict[key]))
                f3.write("\n\n\n")
            else:
                strr = str(key)+":"
                strr += " "*(tabs-len(str(key)))
                strr += str(dataDict[key])+"\n\n"
                f3.write(strr)
        f3.close()
        '''


def mySaveDicc(dicc, nam, diccPathh=defaultPath(), timeStamp=False, wordStamp=False, filterNoneClass=True,
               reduceUniformArrays=True, filterFunctionss=True, saveSmallLength=0):
    if (filterFunctionss):
        dicc = filterFunctions(dicc)

    dirr = os.path.dirname(nam)
    if not os.path.exists(dirr):
        try:
            os.makedirs(dirr)
        except Exception as ex:
            time.sleep(1)
            if not os.path.exists(dirr):
                print(str(ex))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                raise Exception(["My Save Dicc, cannot save file", sys.exc_info()])

    if (timeStamp):
        nam += datetime.datetime.now().strftime("_%Y-%m-%d_%H_%M_%s")
    if (wordStamp == True):
        nam += "_" + randomWord(diccPathh)
    elif (wordStamp != False):
        nam += "_" + randomWord(wordStamp)

    np.save(nam + ".dat", dicc)

    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    def myLen(q):
        if (isListOrArray(q)):
            if (len(q) == 0):
                return 0
            return len(q) * myLen(q[0])
        elif isinstance(q, dict):
            if (len(q) == 0):
                return 0
            return sum(map(myLen, q.values()))
        else:
            return 1

    sortedKeys = sorted(dicc.keys(), key=lambda k: myLen(dicc[k]))
    nam3 = nam + ".out"
    f3 = open(nam3, 'w')
    for key in sortedKeys:
        val = dicc[key]
        if (filterNoneClass and (isListOrArray(val) or val == noneClass())):
            val = noneClass2None(val)
        if (isinstance(val, np.ndarray) or isinstance(val, list)):
            if (reduceUniformArrays):
                try:
                    if (len(set(val)) == 1):
                        val = "[{}]*{}".format(set(val).pop(), len(val))
                except:
                    pass
            hashNum = 30
            f3.write("#" * hashNum + "\n")
            strr = "#" * int(np.floor((hashNum - len(str(key)) - 2) / 2.)) + " "
            strr += str(key)
            strr += " " + "#" * int(np.ceil((hashNum - len(str(key)) - 2) / 2.)) + "\n"
            f3.write(strr)
            f3.write("#" * hashNum + "\n")
            f3.write(str(val))
            f3.write("\n\n\n")
        else:
            strr = str(key) + ":"
            strr += " " * (20 - len(str(key)))
            strr += str(val) + "\n\n"
            f3.write(strr)
    f3.close()

    if (saveSmallLength > 0):
        smallD = {}
        for k in dicc.keys():
            if myLen(dicc[k]) <= saveSmallLength:
                smallD[k] = dicc[k]
        mySaveDicc(smallD, nam + "_small", filterNoneClass=filterNoneClass, reduceUniformArrays=reduceUniformArrays,
                   filterFunctionss=False, saveSmallLength=0)


## recursively remove functions
def filterFunctions(obj):
    if hasattr(obj, '__call__'):
        return str(obj)

    if isinstance(obj, dict):
        return {k: filterFunctions(v) for k, v in obj.items()}
    if isListOrArray(obj):
        return map(filterFunctions, obj)

    return obj


def myBool(v, op, val):
    if (isListOrArray(v)):
        return np.product(map(lambda x: myBool(x, op, val), v))
    if (op == "=="):
        if (v == val):
            return True
        else:
            return False
    if (op == ">"):
        if (v > val):
            return True
        else:
            return False
    if (op == "<"):
        if (v < val):
            return True
        else:
            return False
    if (op == ">="):
        if (v >= val):
            return True
        else:
            return False
    if (op == "<="):
        if (v <= val):
            return True
        else:
            return False
    raise Exception("Unknown operator")


def myBoolDict(dicc, condition):
    key, op, val = condition
    v = dicc[key]
    return myBool(v, op, val)


def myGradient(xx, yy):
    if (len(xx) != len(yy)):
        raise Exception("Unequal x and y. in gradient")
    grad = []
    newX = []

    for i in np.arange(0, len(xx) - 1, 1):
        dy1 = yy[i + 1] - yy[i]
        dx1 = xx[i + 1] - xx[i]
        grad.append(dy1 / dx1)
        newX.append(xx[i] + dx1 / 2.)

    return [newX, grad]


## Allows numerical operations to be done on None, without raising errors
class noneClass(object):
    strr = None
    iterr = 1

    def __init__(self, strr=None):
        self.strr = strr

    def __add__(self, other):
        return noneClass(self.strr)

    def __radd__(self, other):
        return noneClass(self.strr)

    def __sub__(self, other):
        return noneClass(self.strr)

    def __rsub__(self, other):
        return noneClass(self.strr)

    def __mul__(self, other):
        return noneClass(self.strr)

    def __rmul__(self, other):
        return noneClass(self.strr)

    def __div__(self, other):
        return noneClass(self.strr)

    def __rdiv__(self, other):
        return noneClass(self.strr)

    def __str__(self):
        return "NoneClass"

    '''
    def __gt__(self,other):
        return False
    def __lt__(self,other):
        return False
    def __rgt__(self,other):
        return True
    def __rlt__(self,other):
        return True
    def __le__(self,other):
        if(other == None):
            return True
        return False
    def __ge__(self,other):
        if(other == None):
            return True
        return False
    def __rge__(self,other):
        return True
    def __rle__(self,other):
        return True
        '''

    def __eq__(self, other):
        if (str(self) == str(other) or other == None):
            return True
        else:
            return False

    def __ne__(self, other):
        if (str(self) == str(other) or other == None):
            return False
        else:
            return True

    def __req__(self, other):
        if (str(self) == str(other) or other == None):
            return True
        else:
            return False

    def __getitem__(self, i):
        return noneClass(self.strr)

    def __floordiv__(self, other):
        return noneClass(self.strr)

    def __rfloordiv__(self, other):
        return noneClass(self.strr)

    def __truediv(self, other):
        return noneClass(self.strr)

    def __rtruediv(self, other):
        return noneClass(self.strr)

    def __iter__(self):
        return self

    def next(self):
        self.iterr *= -1
        if (self.iterr == 1):
            raise StopIteration
        else:
            return self

    def log10(self):
        return noneClass(self.strr)


def noneClass2None(arr, insteadOfNone=None):
    if (isListOrArray(arr)):
        return map(noneClass2None, arr)
    else:
        if (arr == noneClass()):
            return insteadOfNone
        else:
            return arr


def none2noneClass(arr, insteadOfNoneClass=None):
    if (isListOrArray(arr)):
        return map(lambda x: none2noneClass(x, insteadOfNoneClass), arr)
    else:
        if (arr == None):
            if (insteadOfNoneClass != None):
                return insteadOfNoneClass
            else:
                return noneClass()
        else:
            return arr


def interpOverNone(arr, typ="linear"):
    arr = np.array(arr)
    if (typ == "linear"):
        l = len(arr)
        for i in range(l):
            if (arr[i] == None):
                if (i == 0):
                    j = 0
                    while (arr[j] == None):
                        j += 1
                        if (j == l):
                            raise Exception("interpOverNone: No non-none values")
                    arr[0:j] = arr[j]
                else:
                    j = i
                    while (arr[j] == None):
                        j += 1
                        if (j == l):
                            arr[i:j] = arr[i - 1]
                            break
                    if (j != l):
                        tmp = np.linspace(arr[i - 1], arr[j], j - i + 2)[1:-1]
                        arr[i:j] = tmp
    return np.array(arr, dtype="float64")


## func should have args x,t
def odeToDone(func, y0, stepSize, finishedCutoff=1e-6, maxIters=4, keepAll=False, plotAsCalculate=False,
              useGradPercChange=True, printt=True):
    info = ["empty"]
    try:
        useY0 = y0

        if (plotAsCalculate):
            plt.figure("odeToDone")
            plt.clf()

        keepA = [[y0, None, None, None, 0]]
        for i in range(maxIters):
            odeOut = odeint(func, useY0, [0, stepSize])
            newY = odeOut[-1]
            newGrad = func(newY, 0)
            if useGradPercChange:
                maxPercentChangeGrad = max(abs(np.array(newGrad) * stepSize / np.array(newY)))
            else:
                maxPercentChangeGrad = 0
            maxPercentChangeDiff = max(abs((np.array(newY) - np.array(useY0)) / np.array(newY)))
            maxPercentChange = max(maxPercentChangeGrad, maxPercentChangeDiff)

            if printt:
                print("Iteration {},  fraction change {}".format(i, maxPercentChange))

            if (plotAsCalculate != False):
                try:
                    if plotAsCalculate == "log":
                        for j, yi in enumerate(newY):
                            plt.semilogy(i, yi, 'o', color=matplotlib.cm.rainbow(j / (len(newY) - 0.999)))
                    else:
                        for j, yi in enumerate(newY):
                            plt.plot(i, yi, 'o', color=matplotlib.cm.rainbow(j / (len(newY) - 0.999)))
                    plt.draw()
                    plt.show(block=False)
                except:
                    print("Error plotting in OdeToDone")

            useY0 = newY

            info = [newY, newGrad, maxPercentChange, i + 1, stepSize * (i + 1)]
            if (keepAll):
                keepA.append(info)

            if (maxPercentChange < finishedCutoff):
                if (keepAll):
                    return keepA
                else:
                    return info

    except Exception as ex:
        print(str(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        if (keepAll):
            return keepA
        else:
            return info

    if keepAll:
        return {"FAILED########": "OdeToDone: cutoff not reached {} > {}".format(maxPercentChange, finishedCutoff),
                "keepA": keepA}
    raise Exception("OdeToDone: cutoff not reached {} > {}".format(maxPercentChange, finishedCutoff))


def isListOrArray(aa):
    if isinstance(aa, np.ndarray):
        if len(aa.shape) == 0:
            return False
        else:
            return True
    return isinstance(aa, list) or isinstance(aa, tuple) or isinstance(aa, np.matrix)


def isDict(aa):
    if isinstance(aa, dict):
        return True
    else:
        return False


def noneFromShape(shape):
    if (len(shape) == 0):
        return None

    shape = list(shape)
    ll = shape.pop(0)

    ret = []
    for i in range(ll):
        ret.append(noneFromShape(shape))

    return ret


## takes an list of dictionaries and returns a dictionary of lists
## if given a list of lists of dictionaries it returns a dictionary of lists of lists
## This only makes sense if all the input dictionaries are similar
def transposeDictionary(lst, propagateNone=False, printt=False):
    # if(lst==None):
    #    return {}
    if (len(lst) == 0):
        return {}
    if (isListOrArray(lst[0])):
        return transposeDictionary(list(map(transposeDictionary, lst)))

    ret = {}
    if lst[0] != None and "innerShape" in lst[0]:
        innerShape = lst[0]["innerShape"]
    else:
        innerShape = []

    currLen = 0
    for dicc in lst:
        if dicc == None:
            if propagateNone:
                dicc = {}
            else:
                if printt:
                    print("### One of the dictionaries is None")
                continue
        elif not isinstance(dicc, dict):
            if (printt):
                try:
                    print("### " + str(dicc) + " is not a dictionary!")
                except:
                    print("### is not a dictionary!")
            continue

        for key in dicc:
            if (not key in ret):
                ret[key] = []
            while (len(ret[key]) < currLen):
                ret[key].append(noneFromShape(innerShape))
            ret[key].append(dicc[key])
        currLen += 1

    for key in ret.keys():
        while (len(ret[key]) < currLen):
            ret[key].append(noneFromShape(innerShape))

    ret["innerShape"] = [len(lst)] + innerShape

    return ret


def unTransposeDictionary(dic):
    lenn = None
    for k in dic.keys():
        if k == "innerShape":
            continue
        arr = dic[k]
        if not isListOrArray(arr):
            continue
        if lenn == None:
            lenn = len(arr)
        else:
            if lenn != len(arr):
                raise Exception("Key {} has length {} not {}".format(k, len(arr), lenn))

    dicA = []
    for i in range(lenn):
        tmp = {}
        for k in dic.keys():
            if k == "innerShape":
                continue
            arr = dic[k]
            if isListOrArray(arr):
                tmp[k] = arr[i]
            else:
                tmp[k] = arr
        dicA.append(tmp)
    return dicA


# for when nunmpy doesn't recongise nested lists
# if lst contains None then allowUnequal might work
def myTranspose(lst, allowUnequal=False):
    ilen = len(lst)
    jlen = None
    for i in range(ilen):
        try:
            l = len(lst[i])
        except Exception as ex:
            if (allowUnequal):
                continue
            else:
                print("myTranspose: Cannot calculate length for index i = {}".format(i))
                raise Exception("myTranspose: Cannot calculate length for index i = {} \n\n {}".format(i, str(ex)))

        if (jlen == None):
            jlen = l
        if (jlen != l):
            if (allowUnequal):
                jlen = max(jlen, l)
            else:
                raise Exception("myTranspose: sub lists have unequal lengths")

    ret = noneFromShape([jlen, ilen])

    for i in range(ilen):
        try:
            l = len(lst[i])
        except:
            l = 0
        for j in range(l):
            ret[j][i] = lst[i][j]

    return ret


def randomString(lenn):
    np.random.seed()

    def randomChar():
        rnd = int(np.random.random() * 26)
        return list(string.ascii_lowercase)[rnd]

    return "".join([randomChar() for _ in range(lenn)])


def randomWord(pathh=defaultPath()):
    dictionaryPath = pathh + "code_/dictionary"
    total_bytes = os.stat(dictionaryPath).st_size
    random_point = random.randint(0, total_bytes)
    file = open(dictionaryPath)
    file.seek(random_point)
    file.readline()  # skip this line to clear the partial line
    return "" + file.readline().rstrip()

    # adds latex packages for use in titles etc.


def setupPlot():  # test
    plt.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"] = "\\usepackage{braket}"


# x = setIfUndef("x",3)
def setIfUndef(var, val=None, glb=globals()):
    if var in glb:
        return glb[var]
    else:
        return val


## find the index i for which arr[i] is closest t0 val
def closestIndex(arr, val):
    return min(enumerate(abs(np.array(arr) - val)), key=lambda x: x[1])[0]


def tryNumConversion(strr):
    try:
        return int(strr)
    except:
        try:
            return float(strr)
        except:
            return strr


def myReadCsv(fil, useHeader=False):
    lines = [line.rstrip('\r\n').split(",") for line in open(fil)]
    if (useHeader):
        data = np.array(lines[1:]).transpose()
    else:
        data = np.array(lines).transpose()

    data = map(lambda x: map(tryNumConversion, x), data)

    if (useHeader):
        return dict(zip(lines[0], data))
    else:
        return data


def BEDist(freq, mu, tFreq):
    return 1 / (np.exp((freq - mu) / tFreq) - 1)


def myInterpolate(x, y):
    xx, yy = np.array(sorted(zip(x, y))).transpose()
    return scipy.interpolate.interp1d(xx, yy)


## takes in mode pop as a function of pump power. pop = modePopA[mode][gup]
def thermalFits(delt, modePopA, thresholds, gup):
    if (len(delt) != len(modePopA)):
        raise Exception("Thermal Fits: len(delt) != len(modePopA); {} != {}".format(len(delt), len(modePopA)))

    ret = []
    for i in range(len(modePopA[0])):
        nPop = []
        nDelt = []
        for j in range(len(modePopA)):
            if (gup[i] < thresholds[j]):
                nPop.append(modePopA[j][i])
                nDelt.append(delt[j])

        if (len(nPop) < 2):
            ret.append(noneClass("Too few points below threshold"))
            continue;

        [[dataMu, dataTemp], varMatrix] = scipy.optimize.curve_fit(BEDist, nDelt, np.array(nPop).real,
                                                                   [nDelt[0] - 0.01, 1])
        tmp = {"chemical_potemtial": dataMu, "temperature": dataTemp}

        if (len(nPop > 2)):
            fitModePop = np.array(map(lambda d: BEDist(d, dataMu, dataTemp), delt))
            tmp["residual_squares"] = np.sqrt(np.mean((np.nPop - fitModePop) ** 2))
            tmp["relative_residual_squares"] = tmp["residual_squares"] / np.sqrt(np.mean(np.array(nPop) ** 2))
            sdMat = np.sqrt(np.diag(varMatrix))
            tmp["chemical_potential_sd"] = sdMat[0]
            tmp["temperature_sd"] = sdMat[1]
        else:
            tmp["residual_squares"] = noneClass("Too few points")
            tmp["chemical_potential_sd"] = noneClass("Too few points")
            tmp["temperature_sd"] = noneClass("Too few points")

    return ret


def getDigitsVarBase(num, baseArr, offsets=None):
    if isListOrArray(num):
        return map(lambda x: getDigitsVarBase(x, baseArr, offsets=offsets), num)
    # print(num,baseArr)
    if isinstance(baseArr, int):
        num2 = num
        if (num2 == 0):
            num2 = 1
        baseArr = [baseArr] * int(np.log(num2) / np.log(baseArr) + 1)
    if (offsets == None):
        offsets = [0] * len(baseArr)
    ret = [0] * len(baseArr)
    for i, b in list(enumerate(baseArr))[::-1]:
        tmp = int(num / b)
        ret[i] = int(num - tmp * b + offsets[i])
        num = tmp
    if num > 0:
        return None;
    return ret


def logMean(x, y):
    return 10 ** (np.mean((np.log10(x), np.log10(y))))


def signedLog10(x, logZero=None):
    if isListOrArray(x):
        return map(lambda y: signedLog10(y, logZero), x)
    if x == 0:
        return logZero
    return np.sign(x) * np.log10(abs(x))


def myLog10(x):
    if isListOrArray(x):
        return map(myLog10, x)
    elif x == None:
        return None
    else:
        return np.log10(x)


def myBiLin(x, m1, m2, x0, s):
    return myBiLinDiff(x, m1, m2 - m1, x0, s)


def myBiLinDiff(x, m, dm, x0, s):
    return (m + dm / 2.) * x + np.sqrt(s ** 2 + (x - x0) ** 2) * dm / 2.


# increases array length by 1 and shifts it
def pcolorArray(arr):
    dx = arr[1] - arr[0]
    arr = np.array(arr) - dx / 2.
    arr = np.append(arr, arr[-1] + dx)
    return arr


# colA = [ [12, [0,0.5,1]] ,  [102,[1,1,0]]  ]
def discretePcolorPrep(data, colA, extraCols=True):
    # print("q"
    colours = []
    keys = []
    keyInd = {}
    dataSet = uniqC(data).keys()
    labels = []
    for arr in colA:
        key = arr[0]
        col = arr[1]
        if (len(arr) > 2):
            lab = arr[2]
        else:
            lab = key
        if not extraCols:
            if key not in dataSet:
                continue
        keyInd[key] = len(colours)
        colours += [col]
        keys += [key]
        labels += [lab]

    newData = []
    for line in data:
        tmp = []
        for elem in line:
            if isListOrArray(elem):
                print("discretePcolorPrep: wrong dimensions\n{}".format(elem))
            if not elem in keyInd.keys():
                raise Exception("discretePcolorPrep  data value not in colour array  {}".format(elem))
            tmp += [keyInd[elem]]
        newData.append(tmp)

    cmap = mplcol.ListedColormap(colours)
    # print("w"
    return [newData, cmap, keys, labels]


def myDiscreteColorPlot(data, colA, fig=None, ax=None, name=None, extraCols=True, xData=[], yData=[], **extras):
    data = copy.deepcopy(data)
    if fig == None:
        fig = plt.figure(name)
        plt.clf()
    if ax == None:
        ax = fig.add_subplot(111)

    nd, cmap, keys, labels = discretePcolorPrep(data, colA, extraCols=extraCols)

    if (len(xData) == 0 and len(yData) == 0):
        im = ax.pcolor(nd, cmap=cmap, vmin=-0.5, vmax=len(keys) - 0.5, **extras)
    else:
        im = ax.pcolor(xData, yData, nd, cmap=cmap, vmin=-0.5, vmax=len(keys) - 0.5, **extras)

    cb = colorbar(im)
    cb.set_ticks(range(len(keys)))
    cb.set_ticklabels(labels)

    return {"fig": fig, "ax": ax, "im": im, "cb": cb}


def myPlotFunc(func, x0=0, x1=1, num_points=100, squared=False):
    xx = np.linspace(x0, x1, num_points)
    yy = map(func, xx)
    if squared:
        yy = np.array(yy) ** 2
    plt.plot(xx, yy)


def myFlatten2D(arr):
    return [item for sublist in arr for item in sublist]


def myGaus(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def myVariance(xx, yy):
    if len(xx) != len(yy):
        raise Exception("VARIAANCE: len(xx) != len(yy)")

    meann = 0
    varr = 0
    norm = 0
    for x, y in zip(xx, yy):
        meann += x * y
        norm += y

    meann /= norm

    for x, y in zip(xx, yy):
        varr += y * (x - meann) ** 2
    varr /= norm

    return {"mean": meann, "variance": varr, "sd": np.sqrt(varr)}


def mySearchListND(lst, val, absError=None, relError=None, returnVals=False):
    if isListOrArray(lst[0]):
        ret = []
        for ind in range(len(lst)):
            tmp = mySearchListND(lst[ind], val, absError=absError, relError=relError)
            for tt in tmp:
                ret.append([ind] + list(tt))
        return ret
    else:
        if absError != None:
            ret = filter(lambda x: abs(x[1] - val) < absError, enumerate(lst))
        elif relError != None:
            ret = filter(lambda x: abs(x[1] / val - 1) < relError, enumerate(lst))
        else:
            ret = filter(lambda x: x[1] == val, enumerate(lst))

        if returnVals:
            return ret
        else:
            return map(lambda x: [x[0]], ret)


def mySearchList(lst, val, absError=None, relError=None, returnVals=False):
    if absError != None:
        ret = list(filter(lambda x: abs(x[1] - val) < absError, enumerate(lst)))
    elif relError != None:
        ret = list(filter(lambda x: abs(x[1] / val - 1) < relError, enumerate(lst)))
    else:
        ret = list(filter(lambda x: x[1] == val, enumerate(lst)))

    if returnVals:
        return ret
    else:
        return list(map(lambda x: x[0], ret))


def myNpLoad(nam, maxTries=100, waitTime=10):
    for i in range(maxTries):
        try:
            return np.load(nam)
        except:
            print("Failed to load {}, trying again".format(nam))
            sys.stdout.flush()
            time.sleep(waitTime)


def myLinspaceArray(startA, stopA, num, doLogspace=False):
    if (not isListOrArray(startA)) and (not isListOrArray(stopA)):
        if doLogspace:
            return logspace(startA, stopA, num)
        else:
            return linspace(startA, stopA, num)

    if len(startA) != len(stopA):
        raise Exception("linspaceArray lengths unequal")
    tmp = []
    for i in range(len(startA)):
        if doLogspace:
            tmp.append(np.logspace(startA[i], stopA[i], num))
        else:
            tmp.append(np.linspace(startA[i], stopA[i], num))

    return np.transpose(tmp)


def myInterpArray(arr, insertFactor, doLogspace=False):
    tmp = [arr[0]]
    for i in range(len(arr) - 1):
        nw = myLinspaceArray(arr[i], arr[i + 1], insertFactor + 1)[1:]
        tmp += list(nw)
    return array(tmp)


## returns a string of letters e.g. am so that files can be ordered in a folder
def myAlphabetNumber(num, tot=None):
    alpha = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
             "v", "w", "x", "y", "z"]

    if tot == None:
        tot = num + 1

    letterNum = int(ceil(log(tot) / log(26)))

    if (letterNum == 0):
        letterNum = 1

    numA = getDigitsVarBase(num, [26] * letterNum)

    return "".join(map(lambda x: alpha[x], numA))


def myPolyLog(s, z, dicc=None, returnDicc=False):
    sA = [1.5, 2.5, 3.5, 4.5, 5.5]
    if s not in sA:
        raise Exception("myPolyLog. Only have s in {}".format(sA))

    strr = "polyLog{:.1f}".format(s)
    if dicc == None:
        dicc = np.load(pathh + "code_/models/{}.npy".format(strr)).item(0)

    pl = np.interp(z, dicc["zz"], dicc[strr])

    if returnDicc:
        return [pl, dicc]
    else:
        return pl


def myOdeintComplex(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z


def myPrintMat(mat, singleLines=False):
    if singleLines:
        for v in mat:
            print("\t".join(map(str, v)))
    else:
        for a in mat:
            print(a)


def freq2Trads(freq):
    return 2 * scipy.pi * freq * 1e-12


## switches to stirling for large nums
'''
def myLogEFactorial(n,relErr=1e-3):
    if isListOrArray(n):
        return map(lambda x: myLogEFactorial(x,relErr),n)
    if 1./n >= relErr:
        return math.log(math.factorial(n))
    else:
        return n*log(n) - n + 0.5*log(2*np.pi*n)
def myLog10Factorial(n,relErr=1e-3):
    return myLogEFactorial(n,relErr=1e-3)/log(10)
'''


def myLogFactorial(n):
    if isListOrArray(n):
        return map(myLogFactorial, n)
    return math.lgamma(n + 1)


# input to eigV is eigSys[1]
# This is for non-orthogonal eigenvectors and is therefore slow
# eigSys1.dot(ret) = vec
def myDecomposeAsEigV(vec, eigSys1, xtol=1.49012e-8, useComplex=False):
    vec = array(vec)
    eigV = transpose(eigSys1)  # eigV[0] is an eigenvector

    def optFunc(decomp):
        if isListOrArray(decomp[0]):
            return np.array(map(optFunc, decomp))
        return vec - array(decomp).dot(eigV)

    if useComplex:
        return myFsolveComplex(optFunc, [1] * len(eigV), xtol=xtol)
    return scipy.optimize.fsolve(optFunc, [1] * len(eigV), xtol=xtol)


def myMakeReal(arr, angTol=1e-3):
    if isListOrArray(arr):
        return map(lambda elem: myMakeReal(elem, angTol), arr)

    ang = abs(numpy.angle(arr))
    if ang > pi / 2.:
        ang -= pi
    if abs(ang) > angTol:
        raise Exception("Element is not sufficiently real: {}".format(arr))

    return real(arr)


# turns complex numbes into [real,imag]
def myComplex2Vec(comp):
    if isListOrArray(comp):
        return map(myComplex2Vec, comp)
    return [comp.real, comp.imag]


def myVec2Complex(vec):
    if isListOrArray(vec[0]):
        return map(myVec2Complex, vec)
    return vec[0] + 1j * vec[1]


def myFsolveComplex(func, z0, *args, **kw):
    x0 = myComplex2Vec(z0)

    def nFunc(xx):
        zz = myVec2Complex(xx)
        out = func(zz)
        return myComplex2Vec(out)

    sol = myFsolveND(nFunc, x0, *args, **kw)
    return myVec2Complex(sol)


def myFsolveND(func, x0, *args, **kw):
    sp = shape(x0)
    n0 = np.ravel(x0)

    def nFunc(nn):
        xx = nn.reshape(sp)
        out = func(xx)
        return np.ravel(out)

    sol = scipy.optimize.fsolve(nFunc, n0, *args, **kw)
    return sol.reshape(sp)


def myPlotMatrix(mat, fig=None, logScale=False):
    if logScale:
        tmp = np.log10(copy.deepcopy(mat))
        if (tmp == -np.inf).any() or np.isnan(tmp).any():
            mn = None
            for el in myFlatten2D(tmp):
                if el != -np.inf and not np.isnan(el):
                    if mn == None:
                        mn = el
                    if el < mn:
                        mn = el
            for i in range(len(tmp)):
                for j in range(len(tmp[0])):
                    if tmp[i][j] == -np.inf or np.isnan(tmp[i][j]):
                        tmp[i][j] = mn
        mat = tmp
    if fig == None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ys, xs = np.shape(mat)
    yd = np.arange(ys + 1) - 0.5
    xd = (np.arange(xs + 1) - 0.5)
    im = ax.pcolor(xd, yd, np.array(mat), cmap=cubehelix.cmap())
    plt.colorbar(im)

    if (ax.get_xticks() % 1 != 0).any():
        ax.set_xticks(range(xs))
    if (ax.get_yticks() % 1 != 0).any():
        ax.set_yticks(range(ys))

    ax.set_xlim(xd[[0, -1]])
    ax.set_ylim(yd[[0, -1]])

    ax.invert_yaxis()

    return fig, ax


# identifies the indices in the array where the array flip from <val to >val
def myFlipIndices(arr, val=None):
    aa = arr > val
    inds = []
    prev = aa[0]

    for i, vv in enumerate(aa):
        if prev != vv:
            inds.append(i - 1)
        prev = vv
    return inds


def myRgbToHex(r, g, b):
    str = "#"
    for v in [r, g, b]:
        ss = hex(int(v))[2:]
        if len(ss) == 1:
            ss = "0" + ss
        str += ss
    return str


def myFlatten(lst):
    ret = []
    for el in lst:
        if isListOrArray(el):
            ret += myFlatten(el)
        else:
            ret.append(el)
    return ret


def myGetRainbowColor(ind, outOf):
    return matplotlib.cm.rainbow(ind / (outOf - 1))


def myHistogramLine(data, fig=None, kwargs={}, kwargsPlot={}):
    if fig == None:
        fig = figure()
    ax = fig.add_subplot(111)
    histData, bin_edges = np.histogram(data, **kwargs)
    binMid = (bin_edges[1:] + bin_edges[:-1]) / 2.
    ax.plot(binMid, histData, **kwargsPlot)
    return fig, ax


def transformXForHist(xx):
    ret = [xx[0]]
    for x in xx[1:-1]:
        ret += [x] * 2
    ret.append(xx[-1])
    return ret


def transformYForHist(yy):
    ret = []
    for y in yy:
        ret += [y] * 2
    return ret


def myLoad2DataAs3(dirr):
    q = np.load(dirr, encoding="bytes")
    if (len(np.shape(q)) == 0):
        q = q.item(0)

    def tmp(w):
        if isListOrArray(w):
            return list(map(tmp, w))
        elif isinstance(w, dict):
            e = {}
            for k in w.keys():
                nk = "".join(str(k).split("'")[1:-1])
                # print(k,nk)
                e[nk] = tmp(w[k])
            return e
        else:
            return w

    return tmp(q)


def myOuter(xA, yA, fnc):
    ret = []
    for x in xA:
        tmp = []
        for y in yA:
            tmp.append(fnc(x, y))
        ret.append(tmp)
    return ret


def myFitBED(x, y):
    if max(y[1:]) > 100:
        raise Exception("myFitBED function optimised for one singular point, max(y[1:])>100")

    x2 = np.array(x) - x[0]
    ly0 = np.log10(y[0])
    ly1 = np.log10(y[1:])

    def BED(d, mu, T):
        if T <= 0:
            if not isListOrArray(d):
                d = [d]
            return [np.inf] * len(d)
        ret = 1 / (np.exp((d - mu) / T) - 1)
        return ret

    lmu = -6.1150058520870516
    T = 300
    lmuStep = 1
    lmuStepChange = 0.5
    muDir = -1
    TStep = 100
    TStepChange = 0.5
    TStepDir = 1

    iterations = 0
    while (TStep > 1e-8 or lmuStep > 1e-8):
        iterations += 1
        if iterations > 1000:
            break
        nlmu = np.array([lmu - lmuStep, lmu, lmu + lmuStep])
        nlbed = np.log10(BED(0, -10. ** nlmu, T))
        dif = abs(nlbed - ly0)
        minI = mySearchList(dif, np.min(dif))[0]
        lmu = nlmu[minI]
        minI2 = np.array(minI) - 1
        if muDir != minI2:
            lmuStep *= lmuStepChange
            if minI2 != 0:
                muDir = minI2
        lmuDif = dif[minI]
        # print("mu",minI2,muDir,lmuStep, lmu, nlbed[minI], nlbed[minI]-ly0)

        nT = np.array([T - TStep, T, T + TStep])
        nlbed = list(map(lambda t: np.log10(BED(x2[1:], -10. ** lmu, t)), nT))
        dif = np.sum((nlbed - ly1) ** 2, 1)
        minI = mySearchList(dif, np.min(dif))[0]
        # print("TDif",dif,np.min(dif),minI)
        T = nT[minI]
        minI2 = np.array(minI) - 1
        if muDir != minI2:
            TStep *= TStepChange
            if minI2 != 0:
                TStepDir = minI2

        # print("T",minI2,muDir,TStep, T, dif[minI], nlbed[minI], nlbed[minI]-y[1:])
        # print(iterations, lmuStep, TStep, lmuDif,dif[minI])
        # print("")
    '''    
    figure("tmp")
    clf()
    x3 = linspace(0,13,1000)
    semilogy(x3,BED(x3,-10.**lmu,T))
    semilogy(x2,y,'o')
    '''

    return [x[0] - 10 ** lmu, T, lmu]


def mySet(lst, fnc=None):
    if fnc == None:
        fnc = lambda x: x
    dd = {}
    for el in lst:
        dd[fnc(el)] = el
    return list(dd.values())


## intellginently splits a latex equation over multiple lines
def mySplitLatex(strr, maxChar=100):
    orig = strr
    strr = copy.deepcopy(strr)
    lineA = []
    i = 0
    chars = 0
    lastAM = 0
    lastAMChars = 0
    while (i < len(strr)):
        # print(strr[i:i+20], chars, len(lineA))
        if chars > maxChar:
            lineA.append(strr[:lastAM])
            chars -= lastAMChars
            strr = strr[lastAM:]
            i -= lastAM
            lastAM = 0
            lastAMChars = 0

        if strr[i] == "&":
            chars = 0
            lastAMChars = 0
            i += 1
        elif strr[i].isalnum():
            chars += 1
            i += 1
        elif strr[i] == "_" or strr[i] == "^":
            if strr[i + 1] != "{":
                i += 2
            else:
                # raise Exception("subscript must be in {{ }}")
                bracnum = 1
                i += 1
                while bracnum > 0:
                    if strr[i] == "}":
                        bracnum += 1
                    if strr[i] == "{":
                        bracnum -= 1
                    i += 1
        elif strr[i] == "\\":
            chars += 1
            i += 1
            if i == len(strr) - 1:
                break
            while strr[i].isalnum():
                i += 1
                if i == len(strr) - 1:
                    break
        elif strr[i] == "+" or strr[i] == "-":
            lastAMChars = chars
            chars += 1
            lastAM = i
            i += 1
        else:
            i += 1
    lineA.append(strr)
    ln = sum(list(map(len, lineA)))
    if ln != len(orig):
        raise Exception("different lengths {} != {}".format(ln, len(orig)))
    return "\\\\ \n&".join(lineA)
