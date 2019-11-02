# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:42:03 2017

@author: fredh
"""
import os
import time

import numpy as np
import my_common as mc
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys

pathh = "D:/programming/chronochess-pycharm/"


def sample_piece():
    return [0, "k", 1, 3, 4, 10]  # white, king, alive, x=3, y=4, index in st = 10


def start_state():
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


def sample_operation():
    return [0, 1, 3, 4]  # piece index = 0,  abs/rel move, to/by x = 3, to/by y=4


def sample_moves():
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


def sample_meta_operation_array():
    return [
        [0, [12, 0, 4, 3]],
        [1, [28, 0, 4, 4]],
        [0, [11, 1, 0, 2]],
        [3, [28, 1, -1, -1]]
    ]


def board_from_state(st):
    if len(np.shape(st)) > 2:
        return list(map(board_from_state, st))
    brd = mc.noneFromShape((8, 8))
    for i, pc in enumerate(st):
        if pc[2] == 1:
            brd[pc[3]][pc[4]] = pc
    # return array(brd)
    return brd


def piece_to_string(pc):
    if pc == None:
        return "--"
    strr = ""
    if pc[0] == 0:
        strr += "0"
    else:
        strr += "#"
    strr += "" + pc[1]
    return strr


def print_board(board):
    for ln in np.transpose(board)[::-1]:
        strr = " ".join(map(piece_to_string, ln))
        print(strr)


def print_state(st):
    print_board(board_from_state(st))


## returns [bool, nst]
## bool is whether the operation results in a state change
## nst is the new state after the move (or the same state is move not allowed)
def operate(st, op):
    takenPiece = -1
    if op == None:
        return [False, st, takenPiece]

    st = copy.deepcopy(st)
    brd = np.array(board_from_state(st))
    ind = op[0]
    pc = st[ind]
    if pc[2] == 0:  ## dead
        return [False, st, takenPiece]

    pos0 = np.array([pc[3], pc[4]])

    if op[1] == 0:
        pos1 = np.array([op[2], op[3]])
    else:
        pos1 = pos0 + [op[2], op[3]]

    dpos = pos1 - pos0

    if dpos[0] == 0 and dpos[1] == 0:
        return [False, st, takenPiece]

    orth = False
    diag = False
    if dpos[0] == 0 or dpos[1] == 0:
        orth = True
    if abs(dpos[0]) == abs(dpos[1]):
        diag = True

    if (not orth) and (not diag) and (pc[1] != "n"):
        return [False, st, takenPiece]
    if orth and (pc[1] == "b"):
        return [False, st, takenPiece] ## bishops move diagonally
    if diag and (pc[1] == "r"):
        return [False, st, takenPiece] ## rocks move orthogonally

    if pc[1] == "p":
        sgn = -pc[0] * 2 + 1
        if dpos[1] * sgn <= 0:  ## cannot move backwards
            return [False, st, takenPiece]
        if dpos[0] == 0:  ## move forwards
            if abs(dpos[1]) > 1:
                if (pos0[1] - 3.5) * sgn == -2.5:  ## initial move
                    dpos[1] = 2 * sgn
                else:
                    dpos[1] = 1 * sgn
            endPos = move_to(brd, pos0, dpos, pc[0], False)[0]
        else:  ## take
            endPos = pos0 + np.sign(dpos)
            pc2 = brd[tuple(endPos)]
            if pc2 == None:  ## cannot take nothing
                return [False, st, takenPiece]
            if pc2[0] == pc[0]:  ## cannot take same colour
                return [False, st, takenPiece]
            else:
                st[pc2[5]][2] = 0  ## kill taken peice
                takenPiece = pc2[5]
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
            return [False, st, takenPiece]
        endPos = pos0 + dpos
        if not on_board(endPos):  ## move off board
            return [False, st, takenPiece]
        pc2 = brd[tuple(endPos)]
        if pc2 != None:
            if pc2[0] == pc[0]:  ## cannot take same colour
                return [False, st, takenPiece]
            else:
                st[pc2[5]][2] = 0  ## kill taken peice
                takenPiece = pc2[5]
        st[ind][3:5] = endPos

    if pc[1] == "k":
        dpos = np.sign(dpos)  ## kings only move 1
    if pc[1] in ["k", "b", "r", "q"]:
        out = move_to(brd, pos0, dpos, pc[0], True)
        if out[1] != None:
            st[out[1]][2] = 0  ## kill taken piece
            takenPiece = out[1]
        st[ind][3:5] = out[0]

    return [True, st, takenPiece]


def move_to(brd, pos0, dpos, col, allowTake=True):  # assumes either diag or orth
    lenn = max(abs(dpos))
    ddpos = np.sign(dpos)
    mi = 0
    takenInd = None
    for i in np.arange(lenn) + 1:
        tmpPos = pos0 + i * ddpos
        if not on_board(tmpPos):
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


def on_board(pos):  ## returns true if 0 <= x,y < 8
    for p in pos:
        if p < 0 or p >= 8:
            return False
    return True


def propagate_operations(opA, st=start_state()):
    takenIndA = []
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
        takenIndA.append(out[2])
    return stA,takenIndA


def operation_array_from_meta_array(metaOpA):
    opA = []
    sst = start_state()
    for mop in metaOpA:
        ind = mop[0]
        op = mop[1]

        if ind % 2 != sst[op[0]][0]:  ## wrong colour
            raise Exception("Wrong colour")

        if len(opA) <= ind:
            opA += [None] * (ind - len(opA) + 1)
        opA[ind] = op
    return opA


def jsonify_types(obj):
    if mc.isDict(obj):
        dic = {}
        for k in obj.keys():
            dic[k] = jsonify_types(obj[k])
        return dic
    elif mc.isListOrArray(obj):
        return list(map(jsonify_types, obj))
    elif isinstance(obj, np.int32):
        return int(obj)
    else:
        return obj


# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    metaOpA = [] #sample_meta_operation_array()
    startState = start_state()
    chrono_points = [0, 0]
    turnA = [0] ## integers are overwritten to 0 for some reason

    def do_GET(self):
        # print(dir(self))
        # print("\n")
        # print(vars(self))
        self.send_response(200)

        self.send_header('Content-type', 'text/html')
        self.end_headers()

        print("GET path: {}".format(self.path))
        splt = self.path.split("?")

        try:
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
                message = self.get_standard_return()

            elif preQu == "/move":
                if postQu is None:
                    raise ValueError("Move endpoint requires query parameters")
                move = [int(query_param) for query_param in postQu.split(",")]  # time, index, abs/rel, to/by
                out = self.do_move(move)
                if out == "success":
                    message = self.get_standard_return()
                else:
                    message = "move not permitted: " + out

            else:
                try:
                    f = open(pathh + self.path)
                    message = f.read()
                except Exception as e:
                    print("ERROR")
                    message = "{ERROR:" + str(e) + "}"

            # print(message)
            self.wfile.write(bytes(message, "utf8"))
            return

        except Exception as e:
            mc.print_exception()
            self.wfile.write(bytes("{'ERROR':'ERROR'}", "utf8"))

    def get_standard_return(self):
        opA = operation_array_from_meta_array(self.metaOpA)
        stA, takenIndA = propagate_operations(opA)
        brdA = board_from_state(stA)
        toPlay = self.current_player()
        opA2 = copy.deepcopy(opA)
        for i, el in enumerate(opA2):
            if el == None:
                opA2[i] = "none"
        return json.dumps(jsonify_types(
            {"metaOpA": self.metaOpA, "stA": stA, "opA": opA, "brdA": brdA,
             "tim": time.time(), "toPlay": toPlay, "chrono_points": self.chrono_points,
             "turn": self.get_turn(), "takenIndA": takenIndA}))

    def do_move(self, move):
        if len(move) != 5:
            return "Move array must have length 5"
        col = self.startState[move[1]][0]
        player = self.current_player()
        if col != player:
            return "Moving the wrong colour"

        current_chrono_points = self.chrono_points[player]
        spending_chrono_points = (self.get_turn() - move[0]) / 2
        if spending_chrono_points < 0:
            spending_chrono_points = 0
        if current_chrono_points < spending_chrono_points:
            return f"Insufficent chronopoints {current_chrono_points} < ({self.get_turn()} - {move[0]})/2"

        new_meta_op = [move[0], move[1:]]
        self.metaOpA.append(new_meta_op)
        self.chrono_points[player] -= spending_chrono_points
        self.chrono_points[1-player] += 1
        self.turnA[0] = self.turnA[0] + 1
        return "success"

    def current_player(self):
        return len(self.metaOpA) % 2

    def get_turn(self):
        return self.turnA[0]

try:
    print('starting server...')
    server_address = ('127.0.0.1', 8080)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()
except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    httpd.server_close()


class ChessException(Exception):
    def __init__(self, message):
        self.message = message
