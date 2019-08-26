# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:42:03 2017

@author: fredh
"""

import numpy
from numpy import *
from myCommon import *
import copy


commonPath = "d:/hesten-bec/code_/models/"
import sys

sys.path.insert(0, commonPath)


pathh = "D:/programming/chronochess/"


def samplePiece():
    return [0, "k", 1, 3, 4, 10]  # white, king, alive, x=3, y=4, index in st = 10


def startSt():
    ret = [
        [0, "r", 1, 0, 0],
        [0, "n", 1, 1, 0],
        [0, "b", 1, 2, 0],
        [0, "q", 1, 3, 0],
        [0, "k", 1, 4, 0],
        [0, "b", 1, 5, 0],
        [0, "n", 1, 6, 0],
        [0, "r", 1, 7, 0],
        [0, "p", 1, 0, 1],
        [0, "p", 1, 1, 1],
        [0, "p", 1, 2, 1],
        [0, "p", 1, 3, 1],
        [0, "p", 1, 4, 1],
        [0, "p", 1, 5, 1],
        [0, "p", 1, 6, 1],
        [0, "p", 1, 7, 1],
        [1, "r", 1, 0, 7],
        [1, "n", 1, 1, 7],
        [1, "b", 1, 2, 7],
        [1, "q", 1, 3, 7],
        [1, "k", 1, 4, 7],
        [1, "b", 1, 5, 7],
        [1, "n", 1, 6, 7],
        [1, "r", 1, 7, 7],
        [1, "p", 1, 0, 6],
        [1, "p", 1, 1, 6],
        [1, "p", 1, 2, 6],
        [1, "p", 1, 3, 6],
        [1, "p", 1, 4, 6],
        [1, "p", 1, 5, 6],
        [1, "p", 1, 6, 6],
        [1, "p", 1, 7, 6]
    ]
    for i in range(len(ret)):
        ret[i].append(i)
    return ret


def sampleOp():
    return [0, 1, 3, 4]  # piece index = 0,  abs/rel move, to/by x = 3, to/by y=4


def sampleMoves():
    return [
        [12, 0, 4, 3],
        [28, 0, 4, 4],
        [13, 1, 0, 2],
        [28, 1, 1, -1],
        [6, 0, 5, 2],
        [30, 1, 0, -2],
        [6, 1, 1, 2],
        [21, 1, -10, -10],
        [4, 1, 0, 10],
        [19, 0, 6, 4],
        [12, 1, 0, 1],
        [19, 0, 0, 4]
    ]


def sampleMetaOpA():
    return [
        [0, [12, 0, 4, 3]],
        [1, [28, 0, 4, 4]],
        [0, [11, 1, 0, 2]],
        [3, [28, 1, -1, -1]]
    ]


def st2Board(st):
    if len(shape(st)) > 2:
        return list(map(st2Board, st))
    brd = noneFromShape((8, 8))
    for i, pc in enumerate(st):
        if pc[2] == 1:
            brd[pc[3]][pc[4]] = pc
    # return array(brd)
    return brd


def pc2Str(pc):
    if pc == None:
        return "--"
    strr = ""
    if pc[0] == 0:
        strr += "0"
    else:
        strr += "#"
    strr += "" + pc[1]
    return strr


def printBoard(board):
    for ln in transpose(board)[::-1]:
        strr = " ".join(map(pc2Str, ln))
        print(strr)


def printSt(st):
    printBoard(st2Board(st))


## returns [bool, nst]
## bool is whether the operation results in a state change
## nst is the new state after the move (or the same state is move not allowed)
def operate(st, op):
    if op == None:
        return [False, st]

    st = copy.deepcopy(st)
    brd = np.array(st2Board(st))
    ind = op[0]
    pc = st[ind]
    if pc[2] == 0:  ## dead
        return [False, st]

    pos0 = array([pc[3], pc[4]])

    if op[1] == 0:
        pos1 = array([op[2], op[3]])
    else:
        pos1 = pos0 + [op[2], op[3]]

    dpos = pos1 - pos0

    if dpos[0] == 0 and dpos[1] == 0:
        return [False, st]

    orth = False
    diag = False
    if dpos[0] == 0 or dpos[1] == 0:
        orth = True
    if abs(dpos[0]) == abs(dpos[1]):
        diag = True

    if (not orth) and (not diag) and (pc[1] != "n"):
        return [False, st]

    if pc[1] == "p":
        sgn = -pc[0] * 2 + 1
        if dpos[1] * sgn <= 0:  ## cannot move backwards
            return [False, st]
        if dpos[0] == 0:  ## move forwards
            if abs(dpos[1]) > 1:
                if (pos0[1] - 3.5) * sgn == -2.5:  ## initial move
                    dpos[1] = 2 * sgn
                else:
                    dpos[1] = 1 * sgn
            endPos = moveTo(brd, pos0, dpos, pc[0], False)[0]
        else:  ## take
            endPos = pos0 + sign(dpos)
            pc2 = brd[tuple(endPos)]
            if pc2 == None:  ## cannot take nothing
                return [False, st]
            if pc2[0] == pc[0]:  ## cannot take same colour
                return [False, st]
            else:
                st[pc2[5]][2] = 0  ## kill taken peice
        st[ind][3:5] = endPos

    elif pc[1] == "n":
        mov2 = False
        mov1 = False
        for dp in dpos:
            if abs(dp) == 2 and not mov2:
                mov2 = True
            if abs(dp) == 1 and not mov1:
                mov1 = True
        if (not mov2) or (not mov1):  ## not knight move
            return [False, st]
        endPos = pos0 + dpos
        if not onBoard(endPos):  ## move off board
            return [False, st]
        pc2 = brd[tuple(endPos)]
        if pc2 != None:
            if pc2[0] == pc[0]:  ## cannot take same colour
                return [False, st]
            else:
                st[pc2[5]][2] = 0  ## kill taken peice
        st[ind][3:5] = endPos

    if pc[1] == "k":
        dpos = sign(dpos)  ## kings only move 1
    if pc[1] in ["k", "b", "r", "q"]:
        out = moveTo(brd, pos0, dpos, pc[0], True)
        if out[1] != None:
            st[out[1]][2] = 0  ## kill taken piece
        st[ind][3:5] = out[0]

    return [True, st]


def moveTo(brd, pos0, dpos, col, allowTake=True):  # assumes either diag or orth
    lenn = max(abs(dpos))
    ddpos = sign(dpos)
    mi = 0
    takenInd = None
    for i in arange(lenn) + 1:
        tmpPos = pos0 + i * ddpos
        if not onBoard(tmpPos):
            break
        pc2 = brd[tuple(tmpPos)]
        if pc2 == None:
            mi = i
            continue
        else:
            if allowTake and col != pc2[0]:
                takenInd = pc2[5]
                mi = i
            break
    endPos = pos0 + mi * ddpos
    return [endPos, takenInd]


def onBoard(pos):  ## returns true if 0 <= x,y < 8
    for p in pos:
        if p < 0 or p >= 8:
            return False
    return True


def propagateOps(opA, st=startSt()):
    stA = [st]
    for i, op in enumerate(opA):
        if op == None:
            stA.append(stA[-1])
            continue

        col = st[op[0]][0]
        if col != i % 2:
            raise Exception("Moved wrong colour!")

        out = operate(stA[-1], op)
        stA.append(out[1])
    return stA


def opAFromMetaOpA(metaOpA):
    opA = []
    sst = startSt()
    for mop in metaOpA:
        ind = mop[0]
        op = mop[1]

        if ind % 2 != sst[op[0]][0]:  ## wrong colour
            raise Exception("Wrong colour")

        if len(opA) <= ind:
            opA += [None] * (ind - len(opA) + 1)
        opA[ind] = op
    return opA


def jsonifyTypes(obj):
    if isDict(obj):
        dic = {}
        for k in obj.keys():
            dic[k] = jsonifyTypes(obj[k])
        return dic
    elif isListOrArray(obj):
        return list(map(jsonifyTypes, obj))
    elif isinstance(obj, numpy.int32):
        return int(obj)
    else:
        return obj


# %%
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    metaOpA = sampleMetaOpA()
    startState = startSt()

    def do_GET(self):
        # print(dir(self))
        # print("\n")
        # print(vars(self))
        self.send_response(200)

        self.send_header('Content-type', 'text/html')
        self.end_headers()

        print("GET path: {}".format(self.path))
        splt = self.path.split("?")
        if len(splt) > 1:
            preQu = splt[0]
            postQu = splt[1]
        else:
            preQu = splt[0]
            postQu = None

        if self.path == "/":
            with open(pathh + "main.html") as f:
                message = f.read()

        elif self.path == "/state":
            message = self.getStandardReturn()

        elif preQu == "/move":
            if postQu is None:
                raise ValueError("Move endpoint requires query parameters")
            moveA = [int(query_param) for query_param in postQu.split(",")]  # time, index, abs/rel, to/by
            if len(moveA) != 5:
                raise Exception("MoveA must equal 5")
            col = self.startState[moveA[1]][0]
            if col != len(self.metaOpA) % 2:
                raise Exception("Moving the wrong colour")
            new_meta_op = [moveA[0], moveA[1:]]
            self.metaOpA.append(new_meta_op)
            message = self.getStandardReturn()

        else:
            try:
                f = open(pathh + self.path)
                message = f.read()
            except:
                print("ERROR")
                message = "ERROR"

        # print(message)
        self.wfile.write(bytes(message, "utf8"))
        return

    def getStandardReturn(self):
        opA = opAFromMetaOpA(self.metaOpA)
        stA = propagateOps(opA)
        brdA = st2Board(stA)
        toPlay = len(self.metaOpA) % 2
        opA2 = copy.deepcopy(opA)
        for i, el in enumerate(opA2):
            if el == None:
                opA2[i] = "none"
        return json.dumps(jsonifyTypes(
            {"metaOpA": self.metaOpA, "stA": stA, "opA": opA, "brdA": brdA,
             "tim": time.time(), "toPlay": toPlay}))


try:
    print('starting server...')
    server_address = ('127.0.0.1', 8080)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()
except  KeyboardInterrupt:
    print('^C received, shutting down the web server')
    httpd.server_close()
