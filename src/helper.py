#######################
#   Helper Function   #
#######################
import copy
import re
from Levenshtein import distance

import config


def extractFieldConcept(name: str, isConcept: bool, NNSet: set):
    name = re.sub(r'[^A-Za-z]+', '', name)
    nameTmp = re.sub(r'[^A-Za-z]+', '', name)
    names = re.findall('[A-Z][^A-Z]*', nameTmp[0:1].capitalize() + nameTmp[1:])
    concepts = []
    if isConcept:
        for name in names:
            if name.lower() in NNSet:
                concepts.append(name.lower())
    else:
        for name in names:
            concepts.append(name.lower())
    return concepts

def isConceptConflictFree(name1: str, name2: str, isMethod1: bool, isMethod2: bool, NNSet: set):
    concepts1 = extractFieldConcept(name1, isMethod1, NNSet)
    concepts2 = extractFieldConcept(name2, isMethod2, NNSet)
    for concept in concepts1:
        if concept not in concepts2 and len(concepts2) != 0:
            return False
    for concept in concepts2:
        if concept not in concepts1 and len(concepts1) != 0:
            return False
    return True

def isConceptCongruence(name1: str, name2: str, isMethod1: bool, isMethod2: bool, NNSet: set):
    concepts1 = extractFieldConcept(name1, isMethod1, NNSet)
    concepts2 = extractFieldConcept(name2, isMethod2, NNSet)
    if len(concepts1) == 0 or len(concepts2) == 0:
        return False
    for concept in concepts1:
        if concept not in concepts2:
            return False
    for concept in concepts2:
        if concept not in concepts1:
            return False
    return True

def computeCommonConcepts(name1: str, name2: str, isMethod1: bool, isMethod2: bool, NNSet: set):
    concepts1 = extractFieldConcept(name1, isMethod1, NNSet)
    concepts2 = extractFieldConcept(name2, isMethod2, NNSet)
    return set(concepts1).intersection(set(concepts2))

def computeName(name1: str, name2: str, NNSet: set, isMethod1: bool, isMethod2: bool):
    name1 = re.sub(r'[^A-Za-z]+', '', name1)
    name2 = re.sub(r'[^A-Za-z]+', '', name2)

    nameTmp = re.sub(r'[^A-Za-z]+', '', name1)
    names = re.findall('[A-Z][^A-Z]*', nameTmp[0:1].capitalize() + nameTmp[1:])
    names1 = []
    if isMethod1:
        for name in names:
            if name.lower() in NNSet:
                names1.append(name.lower())
    else:
        for name in names:
            names1.append(name.lower())

    nameTmp = re.sub(r'[^A-Za-z]+', '', name2)
    names = re.findall('[A-Z][^A-Z]*', nameTmp[0:1].capitalize() + nameTmp[1:])
    names2 = []
    if isMethod2:
        for name in names:
            if name.lower() in NNSet:
                names2.append(name.lower())
    else:
        for name in names:
            names2.append(name.lower())

    print(names1)
    print(names2)

    if len(names1) == 0 or len(names2) == 0:
        return 1

    hit1 = 0
    hit2 = 0
    for subname1 in names1:
        for subname2 in names2:
            if checkStringEquality(subname1, subname2):
                hit2 += 1
    for subname2 in names2:
        for subname1 in names1:
            if checkStringEquality(subname2, subname1):
                hit1 += 1
    return (hit1 + hit2) * 1.0 / (len(names1) + len(names2))

def computeNameSimilarity(name1: str, name2: str, NNSet: set, isMethod1: bool, isMethod2: bool):
    name1 = re.sub(r'[^A-Za-z]+', '', name1)
    name2 = re.sub(r'[^A-Za-z]+', '', name2)

    nameTmp = re.sub(r'[^A-Za-z]+', '', name1)
    names = re.findall('[A-Z][^A-Z]*', nameTmp[0:1].capitalize() + nameTmp[1:])
    config.tagTokenCnt += len(names)

    names1 = []
    if isMethod1:
        for name in names:
            if name.lower() in NNSet:
                names1.append(name.lower())
    else:
        for name in names:
            names1.append(name.lower())

    nameTmp = re.sub(r'[^A-Za-z]+', '', name2)
    names = re.findall('[A-Z][^A-Z]*', nameTmp[0:1].capitalize() + nameTmp[1:])
    config.tagTokenCnt += len(names)

    names2 = []
    if isMethod2:
        for name in names:
            if name.lower() in NNSet:
                names2.append(name.lower())
    else:
        for name in names:
            names2.append(name.lower())

    if len(names1) == 0 or len(names2) == 0:
        return 1

    hit1 = 0
    hit2 = 0
    for subname1 in names1:
        for subname2 in names2:
            if checkStringEquality(subname1, subname2):
                hit2 += 1
    for subname2 in names2:
        for subname1 in names1:
            if checkStringEquality(subname2, subname1):
                hit1 += 1
    return (hit1 + hit2) * 1.0 / (len(names1) + len(names2))
    # if name1 == "" or name2 == "":
    #     return 1
    # else:
    #     return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

def getTypeSimilarity(type1, type2, CHADic):
    if type1 == type2:
        return 0
    if len(type1) <= 2 and type2 != "java.lang.Object" and type2 != type1 and type2 != "Object":
        return -1
    if len(type2) <= 2 and type1 != "java.lang.Object" and type2 != type1 and type1 != "Object":
        return -1

    isObj1 = type1 == "Object" or len(type1) <= 2 or type1 == "java.lang.Object"
    isObj2 = type2 == "Object" or len(type2) <= 2 or type2 == "java.lang.Object"
    if isObj1 and isObj2:
        return 0
    if isObj1 and (not isPrimitiveType(type2)):
        return 1
    if isObj2 and (not isPrimitiveType(type1)):
        return 1

    if type1 not in CHADic and type2 not in CHADic:
        return -1
    elif type1 in CHADic:
        if type2 not in CHADic[type1]:
            return -1
    elif type2 in CHADic:
        if type1 not in CHADic[type2]:
            return -1

    if type1 not in CHADic and type2 not in CHADic:
        return -1
    dis1 = longest_path(type1, type2, CHADic)
    dis2 = longest_path(type2, type1, CHADic)
    if dis1 == -1 and dis2 == -1:
        return -1
    if dis1 == -1:
        return dis2
    if dis2 == -1:
        return dis1
    return -1

def longest_path(start_node, end_node, CHADic):
    # Create a dictionary to store the longest path to each node
    longest_paths = {}

    # Initialize the longest path to the start node to be 0
    longest_paths[start_node] = 0

    # Create a stack to store the topologically sorted nodes
    stack = [start_node]
    history = {start_node}

    # Topologically sort the nodes in the DAG
    while stack:
        node = stack.pop()

        # If we have reached the end node, return the longest path to it
        if node == end_node:
            return longest_paths[node]

        # If the node is not in the DAG, return -1
        if node not in CHADic:
            return -1

        # Update the longest path to each child node
        for child in CHADic[node]:
            if child not in longest_paths:
                longest_paths[child] = longest_paths[node] + 1
            else:
                longest_paths[child] = max(longest_paths[child], longest_paths[node] + 1)

            # Add the child node to the stack
            stack.append(child)

            if child not in history:
                history.add(child)
            else:
                return longest_paths[child]

    # If we have not found the end node, return -1
    return -1

def isPrimitiveType(type: str):
    s = {"byte", "short", "int", "long", "float", "double", "char", "boolean", "void"}
    if type in s:
        return True
    else:
        return False


def splitCamelCase(s):
    """
    Split a Camel-Case string into a list of tokens.

    Parameters:
    s (str): The Camel-Case string to split.

    Returns:
    list: The list of tokens.
    """
    # Initialize the list of tokens.
    tokens = []

    # Iterate over the characters in the string.
    i = 0
    while i < len(s):
        # If the current character is uppercase, start a new token.
        if s[i].isupper():
            # Find the end of the token.
            j = i + 1
            while j < len(s) and s[j].islower():
                j += 1

            # Add the token to the list.
            tokens.append(s[i:j].lower())

            # Move the index to the end of the token.
            i = j
        else:
            # If the current character is lowercase, skip it.
            i += 1

    # Return the list of tokens.
    return tokens


def splitMethodSignatureFromCodeQL(sig: str):
    methodName = sig[:sig.index("(")]
    paraStr = sig[sig.index("(") + 1 : sig.index(")")]
    return methodName, paraStr.split(",")


def splitMethodSignatureFromJavaDoc(originalSig: str):
    sig = originalSig.replace("abstract", "").strip(" ").replace("static", "").strip(" ").strip(">")
    sig = removeTypeParaInSigFromJavaDoc(sig)
    retType = sig[:sig.index(" ")]
    trimSig = sig[sig.index(" ") + 1:].strip(" ")
    methodName = trimSig[:trimSig.index("(")]
    paraStr = trimSig[trimSig.index("(") + 1 : trimSig.index(")")]
    paraTypes = []
    paraNames = []
    for para in paraStr.split(","):
        paraType = para.strip(" ").split(' ')[0].strip(" ")
        if paraType != '':
            paraTypes.append(paraType)
            if len(para.strip(" ").split(' ')) == 1:
                return None, None, None, None
            paraName = para.strip(" ").split(' ')[1].strip(" ")
            paraNames.append(paraName)
    return retType, methodName, paraTypes, paraNames


def checkStringEquality(s1: str, s2: str, theta = 0.9):
    lev_distance = distance(s1.lower(), s2.lower())
    similarity = 1 - (lev_distance / max(len(s1), len(s2)))
    if similarity > 0.9:
        return True
    else:
        return False


def computeStrListSimilarity(list1: [str], list2: [str]):
    """
    Compute the ratio of the number of common elements in two lists
    and the size of their union.

    Parameters:
    list1 (list): The first list of strings.
    list2 (list): The second list of strings.

    Returns:
    float: The ratio of the number of common elements and the size of the union.
    """
    # Compute the number of common elements.
    common = len(set(list1).intersection(set(list2)))

    # Compute the size of the union.
    union = len(set(list1).union(set(list2)))

    # Compute the ratio.
    if union == 0:
        ratio = 0.0
    else:
        ratio = float(common) / union
    return ratio


def printCandidateSpecs(specs: list):
    for spec in specs:
        (className, methodSig1, methodSig2, index) = spec
        (retType1, methodName1, ParaTypeList1, sig1) = methodSig1
        (retType2, methodName2, ParaTypeList2, sig2) = methodSig2
        tokenList1 = splitCamelCase(methodName1)
        tokenList2 = splitCamelCase(methodName2)
        similarity = computeStrListSimilarity(tokenList1, tokenList2)
        print(className)
        print(methodName1, methodName2, similarity)
        print(retType1, ParaTypeList1)
        print(retType2, ParaTypeList2)
        print(index)
        print("\n")
    print(len(specs))


def commonSubstring(s1, s2):
    """
    Returns the longest common substring of s1 and s2.
    """
    m = len(s1)
    n = len(s2)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    result = ''
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > longest:
                    longest = table[i][j]
                    result = s1[i - longest:i]
            else:
                table[i][j] = 0
    return result


def eraseTypeInGenerics(type: str):
    if len(type) == 1:
        return "Object"
    type = type[type.rfind(".") + 1:]
    if type.find("<") == -1:
        return type[type.rfind(".") + 1:]
    else:
        u = type[0: type.find("<")]
        return u[u.rfind(".") + 1:]

def removeTypeParaInSigFromJavaDoc(s: str):
    result = ""
    inside_tag = False
    for char in s:
        if char == "<":
            inside_tag = True
        elif char == ">":
            inside_tag = False
        elif not inside_tag:
            result += char
    result = result.replace("  ", " ").strip(" ")
    return result

def computeSpecMethodIndex(clusteredSuccObjectMethodPairDic: dict, spec):
    # format of clusteredSuccObjectMethodPairDic
    # [ {loc1 -> [ loc2 -> [sig1, sig2, receiverType1, receiverType2, returnType1, returnType2, loc1, loc2]]

    def collectAllPath(graph: dict):
        entryLocs = set(graph.keys())
        for loc in graph:
            entryLocs = entryLocs - set(graph[loc].keys())
        # print(len(entryLocs))
        paths = []
        for entryLoc in entryLocs:
            paths.extend(searchInGraph(graph, entryLoc, [[]], set([])))
        return paths

    def searchInGraph(graph: dict, curLoc: str, paths: list, historyLoc: set):
        if curLoc not in graph:
            return paths
        if curLoc in historyLoc:
            return paths
        extendedPaths = []
        for loc in graph[curLoc]:
            [sig1, sig2, receiverType1, receiverType2, returnType1, returnType2, loc1, loc2] = graph[curLoc][loc]
            call = [sig2, receiverType2, returnType2, loc2]
            for path in paths:
                path.append(call)
            newHistoryLoc = copy.deepcopy(historyLoc)
            newHistoryLoc.add(curLoc)
            extendedPaths.extend(searchInGraph(graph, loc, paths, newHistoryLoc))
        return extendedPaths


    (className, (returnType1, sig1), (returnType2, sig2), k) = spec
    indexPairs = []
    k = 0
    correctCnt = 0
    wrongCnt = 0
    for graph in clusteredSuccObjectMethodPairDic:
        k += 1
        paths = collectAllPath(graph)
        # print("path number: ", len(paths))
        for path in paths:
            index1 = -1
            index2 = -1
            for i in range(len(path)):
                [sig, receiverType, returnType, loc] = path[i]
                typeErasedReceiverType = eraseTypeInGenerics(receiverType)
                if sig1 == sig and className == typeErasedReceiverType and returnType1 == returnType and index1 == -1:
                    index1 = i
                if sig2 == sig and className == typeErasedReceiverType and returnType2 == returnType and index2 == -1:
                    index2 = i
            if index1 != -1 and index2 != -1:
                if index1 < index2:
                    correctCnt += 1
                    return indexPairs, correctCnt > 0
                elif index1 > index2:
                    wrongCnt +=1
            indexPairs.append([index1, index2])
    return indexPairs, correctCnt > 0

def eraseSig(methodSig1: str):
    newMethodSig1 = methodSig1[:methodSig1.find("(")] + "("
    parameterList1 = methodSig1[methodSig1.find("(") + 1 : methodSig1.find(")")].split(",")
    for para in parameterList1:
        if para == '':
            continue
        if para.find(".") != -1:
            newMethodSig1 += "java.lang.Object, "
        else:
            newMethodSig1 += para + ", "
    newMethodSig1 = newMethodSig1.rstrip(", ") + ")"
    return newMethodSig1

def eraseClassName(className: str):
    if className.find("<") == -1:
        return className
    newClassName = className[:className.find("<")] + "<"
    typeParaNum = className.count(",") + 1
    for i in range(typeParaNum - 1):
        newClassName += "java.lang.Object, "
    newClassName += "java.lang.Object>"
    return newClassName

def erase(specs: set):
    erasedSpecs = set([])
    for spec in specs:
        (className, (retType1, methodSig1), (retType2, methodSig2), index) = spec
        newClassName = eraseClassName(className)
        if className == eraseClassName(className):
            erasedSpecs.add(spec)
            continue

        newRetType1 = eraseClassName(retType1)
        newRetType2 = eraseClassName(retType2)

        newMethodSig1 = eraseSig(methodSig1)
        newMethodSig2 = eraseSig(methodSig2)
        newSpec = (newClassName, (newRetType1, newMethodSig1), (newRetType2, newMethodSig2), index)
        erasedSpecs.add(newSpec)
    return erasedSpecs