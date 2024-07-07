import os
import time
import signal
import random
import bitarray
import subprocess

def makeInputFile(pos, neg, alphabet):
    f = open("subInput", 'w')
    for p in pos:
        f.write(str(p))
        f.write("\n")
    f.write("---\n")
    for n in neg:
        f.write(str(n))
        f.write("\n")
    f.write("---\n")
    f.write("All operators\n")
    f.write("---\n")
    f.write(alphabet)
    f.close()

def runGPU(pos, neg, alphabet, costfun, maxCost, AprUnqChkTyp, NegType):
    makeInputFile(pos, neg, alphabet)
    c1 = costfun[0]
    c2 = costfun[1]
    c3 = costfun[2]
    c4 = costfun[3]
    c5 = costfun[4]
    c6 = costfun[5]
    c7 = costfun[6]
    c8 = costfun[7]
    output = subprocess.run(
        ["./ltli6463",
        "subInput",
        str(c1), str(c2), str(c3), str(c4), str(c5), str(c6), str(c7), str(c8),
        str(maxCost),
        str(AprUnqChkTyp),
        str(NegType)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    return(str(output.stdout).replace('\\n', '\n')[1:])

def get_all_file_paths_in_folder(path):
    all_file_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if ".trace" in file_path:
                all_file_paths.append(file_path)
    return all_file_paths

def Negation(bvList):
    return [~bv for bv in bvList]

def Intersection(bvList1, bvList2):
    outList = []
    for i in range(len(bvList1)):
        outList.append(bvList1[i] & bvList2[i])
    return outList

def Union(bvList1, bvList2):
    outList = []
    for i in range(len(bvList1)):
        outList.append(bvList1[i] | bvList2[i])
    return outList

def Next(bvList):
    return [bv >> 1 for bv in bvList]

def Finally(bvList):
    outList = []
    for bv in bvList:
        bv |= bv >> 1
        bv |= bv >> 2
        bv |= bv >> 4
        bv |= bv >> 8
        bv |= bv >> 16
        bv |= bv >> 32
        outList.append(bv)
    return outList

def Globally(bvList):
    outList = []
    for bv in bvList:
        cs = ~bv
        cs |= cs >> 1
        cs |= cs >> 2
        cs |= cs >> 4
        cs |= cs >> 8
        cs |= cs >> 16
        cs |= cs >> 32
        bv &= ~cs
        outList.append(bv)
    return outList

def Until(bvList1, bvList2):
    outList = []
    for i in range(len(bvList1)):
        bv1 = bvList1[i]
        bv2 = bvList2[i]
        bv2 |= bv1 & (bv2 >> 1)
        bv1 &= bv1 >> 1
        bv2 |= bv1 & (bv2 >> 2)
        bv1 &= bv1 >> 2
        bv2 |= bv1 & (bv2 >> 4)
        bv1 &= bv1 >> 4
        bv2 |= bv1 & (bv2 >> 8)
        bv1 &= bv1 >> 8
        bv2 |= bv1 & (bv2 >> 16)
        bv1 &= bv1 >> 16
        bv2 |= bv1 & (bv2 >> 32)
        outList.append(bv2)
    return outList

def Until(bvList1, bvList2):
    outList = []
    for i in range(len(bvList1)):
        bv1 = bvList1[i]
        bv2 = bvList2[i]
        bv = bv2
        idx = len(bv1) - 1
        while idx >= 0:
            if (bv2[idx]):
                idx2 = idx + 1
                while idx2 < len(bv1) and bv1[idx2]:
                    bv[idx2] = 1
                    idx2 += 1
                while idx >= 0 and bv2[idx]:
                    idx -= 1
            idx -= 1
        outList.append(bv)
    return outList

def bitVec(traceList, alphabet, formula):

    i = 0
    if formula[0] == '(':
        p = 1
        i += 1
        while p != 0:
            if formula[i] == '(':
                p += 1
            elif formula[i] == ')':
                p -= 1
            i += 1

    if formula[i] == '~':
        return Negation(bitVec(traceList, alphabet, formula[2:-1]))

    elif formula[i] == '&':
        l = bitVec(traceList, alphabet, formula[1:i-1])
        r = bitVec(traceList, alphabet, formula[i+2:-1])
        return Intersection(l, r)

    elif formula[i] == '|':
        l = bitVec(traceList, alphabet, formula[1:i-1])
        r = bitVec(traceList, alphabet, formula[i+2:-1])
        return Union(l, r)

    elif formula[i] == 'X':
        return Next(bitVec(traceList, alphabet, formula[2:-1]))

    elif formula[i] == 'F':
        return Finally(bitVec(traceList, alphabet, formula[2:-1]))

    elif formula[i] == 'G':
        return Globally(bitVec(traceList, alphabet, formula[2:-1]))

    elif formula[i] == 'U':
        l = bitVec(traceList, alphabet, formula[1:i-1])
        r = bitVec(traceList, alphabet, formula[i+2:-1])
        return Until(l, r)

    else: # char
        j = alphabet.split(',').index(formula[i])
        bvList = []
        for trace in traceList:
            bv = [int(token.split(',')[j]) for token in trace.split(';')]
            bv.reverse()
            bvList.append(bitarray.bitarray(bv))
        return bvList

def LTLmatch(traceList, alphabet, formula):
    return [bv[-1] == 1 for bv in bitVec(traceList, alphabet, formula)]

# Divide-and-conquer: Deterministic Splitting
def DetSplit (window, pos, neg, alphabet, costfun, maxCost, AprUnqChkTyp, NegType):

    if len(pos) + len(neg) <= window:
        output = runGPU(pos, neg, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
        if not "not_found" in output:
            return output.split("\n")[-2].split()[-1][1:-1]

    p1 = pos[:int(len(pos)/2)]
    p2 = pos[int(len(pos)/2):]
    n1 = neg[:int(len(neg)/2)]
    n2 = neg[int(len(neg)/2):]

    phi11 = DetSplit(window, p1, n1, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)

    phi11FilterOnP2 = LTLmatch(p2, alphabet, phi11)
    phi11FilterOnN2 = LTLmatch(n2, alphabet, phi11)
    phi11AcceptsTheWholeP2 = not False in phi11FilterOnP2
    phi11RejectsTheWholeN2 = not True  in phi11FilterOnN2
    if phi11AcceptsTheWholeP2 and phi11RejectsTheWholeN2:
        return phi11

    if phi11RejectsTheWholeN2:
        left = phi11
    else:
        n2AndPhi11 = [x for x, flag in zip(n2, phi11FilterOnN2) if flag]
        phi12 = DetSplit(window, p1, n2AndPhi11, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
        Phi11CanBeReplacedByPhi12 = not True in LTLmatch(list(set(neg) - set(n2AndPhi11)), alphabet, phi12)
        if Phi11CanBeReplacedByPhi12:
            left = phi12
        else:
            left = "(" + phi11 + ")&(" + phi12 + ")"
        leftIsAnswer = not False in LTLmatch(p2, alphabet, left)
        if leftIsAnswer:
            return left
    
    leftFilterOnP2 = LTLmatch(p2, alphabet, left)
    P2MinusLeft = [x for x, flag in zip(p2, leftFilterOnP2) if not flag]

    phi21 = DetSplit(window, P2MinusLeft, n1, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
    
    phi21FilterOnP1 = LTLmatch(list(set(pos) - set(P2MinusLeft)), alphabet, phi21)
    phi21FilterOnN2 = LTLmatch(n2, alphabet, phi21)
    phi21AcceptsTheWholeP1 = not False in phi21FilterOnP1
    phi21RejectsTheWholeN2 = not True  in phi21FilterOnN2
    if phi21AcceptsTheWholeP1 and phi21RejectsTheWholeN2:
        return phi21

    if phi21RejectsTheWholeN2:
        right = phi21
    else:
        n2AndPhi21 = [x for x, flag in zip(n2, phi21FilterOnN2) if flag]
        phi22 = DetSplit(window, P2MinusLeft, n2AndPhi21, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
        Phi21CanBeReplacedByPhi22 = not True in LTLmatch(list(set(neg) - set(n2AndPhi21)), alphabet, phi22)
        if Phi21CanBeReplacedByPhi22:
            right = phi22
        else:
            right = "(" + phi21 + ")&(" + phi22 + ")"
        rightIsAnswer = not False in LTLmatch(list(set(pos) - set(P2MinusLeft)), alphabet, right)
        if rightIsAnswer:
            return right

    return "(" + left + ")|(" + right + ")"

# Divide-and-conquer: Random Splitting
def RandSplit(window, pos, neg, alphabet, costfun, maxCost, AprUnqChkTyp, NegType):

    random.seed(0)
    
    win = window

    while True:

        if len(pos) + len(neg) <= win:
            p1 = pos
            n1 = neg
        else:
            if len(pos) <= win / 2:
                p1 = pos
                n1 = random.sample(neg, win - len(p1))
            elif len(neg) <= win / 2:
                n1 = neg
                p1 = random.sample(pos, win - len(n1))
            else:
                p1 = random.sample(pos, int(win / 2))
                n1 = random.sample(neg, win - len(p1))

        output = runGPU(p1, n1, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)

        if not "not_found" in output:
            phi11 = output.split("\n")[-2].split()[-1][1:-1]
            break
        else:
            win = int(win / 2)

    phi11FilterOnP = LTLmatch(pos, alphabet, phi11)
    phi11FilterOnN = LTLmatch(neg, alphabet, phi11)

    p2 = [x for x, flag in zip(pos, phi11FilterOnP) if not flag]
    n2 = [x for x, flag in zip(neg, phi11FilterOnN) if flag]

    p1 = list(set(pos) - set(p2))
    n1 = list(set(neg) - set(n2))

    if p2 == [] and n2 == []:
        return phi11

    if n2 == []:
        left = phi11
    else:
        phi12 = RandSplit(window, p1, n2, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
        Phi11CanBeReplacedByPhi12 = not True in LTLmatch(n1, alphabet, phi12)
        if Phi11CanBeReplacedByPhi12:
            left = phi12
        else:
            left = "(" + phi11 + ")&(" + phi12 + ")"
        leftIsAnswer = not False in LTLmatch(p2, alphabet, left)
        if leftIsAnswer:
            return left

    phi21 = RandSplit(window, p2, n1, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
    phi21AcceptsTheWholeP1 = not False in LTLmatch(p1, alphabet, phi21)
    phi21RejectsTheWholeN2 = not True  in LTLmatch(n2, alphabet, phi21)
    if phi21AcceptsTheWholeP1 and phi21RejectsTheWholeN2:
        return phi21

    if phi21RejectsTheWholeN2:
        right = phi21
    else:
        phi22 = RandSplit(window, p2, n2, alphabet, costfun, maxCost, AprUnqChkTyp, NegType)
        Phi21CanBeReplacedByPhi22 = not True in LTLmatch(n1, alphabet, phi22)
        if Phi21CanBeReplacedByPhi22:
            right = phi22
        else:
            right = "(" + phi21 + ")&(" + phi22 + ")"
        rightIsAnswer = not False in LTLmatch(p1, alphabet, right)
        if rightIsAnswer:
            return right

    return "(" + left + ")|(" + right + ")"

def c(ltl):
    return(len(ltl.replace(',', '').replace(' ', '').replace('(', '').replace(')', '')))

def LTLOfSingleTrace(alphabet, trace, LTL=''):
    if trace == []:
        return LTL
    if LTL == '':
        LTL = f"{'&'.join(trace[-1])}&~X({alphabet[0]}|~{alphabet[0]})"
    else:
        LTL = f"{'&'.join(trace[-1])}&X({LTL})"
    return LTLOfSingleTrace(alphabet, trace[:-1], LTL)

def overFittedCase(alphabet, pos):
    out = []
    for p in pos:
        trace = [t.split(',') for t in p.split(';')]
        for i in range(len(trace)):
            for j in range(len(alphabet)):
                if trace[i][j] == '1':
                    trace[i][j] = alphabet[j]
                else:
                    trace[i][j] = '~' + alphabet[j]
        singleTraceLTL = LTLOfSingleTrace(alphabet, trace)
        out.append(singleTraceLTL)
    if len(out) == 1:
        return out[0]
    else:
        return '|'.join([f"({tr})" for tr in out])

def runWithTimeout(func, args, timeout):
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = func(*args)
    except TimeoutError as exc:
        result = 'Function timed out.'
    finally:
        signal.alarm(0)
    return result
    
def runDC(filePath, dcAlgo = RandSplit, window = 64, costfun = [1, 1, 1, 1, 1, 1, 1, 1], maxCost = 500, RlxUnqChkTyp = 3, NegTyp = 2, timeout = 2000):

    with open(filePath, 'r') as file:
        content = file.read()

    pos = content.split("\n")[:content.split("\n").index("---")]
    neg = content.split("\n")[content.split("\n").index("---")+1:-4]
    alphabet = content.split("\n")[-1]

    start_time = time.time()
    res = runWithTimeout(dcAlgo, [window, pos, neg, alphabet, costfun, maxCost, RlxUnqChkTyp, NegTyp], timeout=timeout)
    end_time = time.time()

    print(f"Size(P, N): ({len(pos)}, {len(neg)})")
    print(f"Time: {round(end_time - start_time, 3)}s")
    print(f"LTL: {res}")