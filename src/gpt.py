import sys

import config
from helper import *
from prompt import *
import openai # For GPT-3 API ...
import nltk
from collections import Counter
import random
from config import *
import tiktoken

response = None
received = False
engine = "gpt-3.5-turbo-0301"


def retrieveDoc(className: str, retType: str, methodName: str):
    result = ""
    try:
        message = "Please describe the specification of the method "
        message += "`" + retType + " " + methodName + "`"
        message += " of the class " + className
        message += " in no more than three sentences."

        s = ['AI']
        input = [
            # {"role": "system", "content": "here is an query about write reports "},
            # {"role": "user", "content": "report should contain " + s[0] + "\n now, please complete writing: \n" + s[1]}
            {"role": "user", "content": message}
        ]

        config.key_id, openai.api_key = getKey(config.key_id)
        response = openai.ChatCompletion.create(
            model=engine,
            messages=input,
            # max_tokens=max_length,
            # temperature=0,
            # stop=stop
        )
        received = True
        result = response["choices"][0]["message"]["content"]
    except:
        error = sys.exc_info()[0]
        # if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
        #     print(f"InvalidRequestError\nPrompt passed in:\n\n{input}\n\n")
        #     assert False
        print("API error:", sys.exc_info())
    return result


def retrieveMagicWords(promptMode, n, t1=1):
    results = {}
    if "manualPrompt" in promptMode:
        return {}
    elif "autoPrompt_TwoTypes" in promptMode:
        messages = {
            "memory read": readMagicWordQuestion,
            "memory write": writeMagicWordQuestion
        }
    elif "autoPrompt_FourTypes" in promptMode:
        messages = {
            "memory read": readMagicWordQuestion,
            "memory write": writeMagicWordQuestion,
            "deletion upon memory": deleteMagicWordQuestion,
            "insertion upon memory": insertMagicWordQuestion
        }
    allMagicWords = []
    for i in range(n):
        for k in messages:
            isRecieved = False
            while not isRecieved:
                try:
                    message = messages[k]

                    s = ['AI']
                    input = [
                        {"role": "system", "content": "You are a good Java programmer and are very good at choosing proper names for Java methods."},
                        {"role": "user", "content": message}
                    ]

                    config.key_id, openai.api_key = getKey(config.key_id)
                    response = openai.ChatCompletion.create(
                        model=engine,
                        messages=input,
                        # max_tokens=max_length,
                        temperature=t1,
                        # stop=stop
                    )
                    results[k] = response["choices"][0]["message"]["content"]
                    print(results[k])
                    isRecieved = True
                except:
                    error = sys.exc_info()[0]
                    print("API error:", sys.exc_info())


        singleMagicWords = {}
        wordtags = nltk.ConditionalFreqDist((w.lower(), t)
                                            for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
        for k in results:
            singleMagicWords[k] = []
            words = results[k].split()
            for word in words:
                word = re.sub(r'[^A-Za-z]+', '', word)
                tags = dict(wordtags[word.lower()])
                if word == '':
                    continue
                isVerb = True
                if 'VERB' in tags:
                    for tag in tags:
                        print(word, tag, tags[tag])
                        if tags[tag] > tags['VERB']:
                            isVerb = False
                else:
                    isVerb = False
                if isVerb:
                    singleMagicWords[k].append(word.lower())
        print(singleMagicWords)
        allMagicWords.append(singleMagicWords)
    magicWords = {}
    for k in messages:
        wordScoreDic = {}
        for singleMagicWords in allMagicWords:
            cnt = 1
            print(singleMagicWords)
            for word in singleMagicWords[k]:
                cnt += 1
                if word not in wordScoreDic:
                    wordScoreDic[word] = 1.0 / cnt
                else:
                    wordScoreDic[word] += 1.0 / cnt
        print("wordScoreDic", wordScoreDic)
        magicWords[k] = max(wordScoreDic, key=wordScoreDic.get)
    print(magicWords)
    return magicWords


def retrieveMemoryOperationType(className: str, methodSig: str, methodDoc: str, magicWords: dict, promptMode: str, m, t2 = 1.0, indx = 0):
    result = {}
    print(indx)
    modelIndex = indx % 5
    keyIndex = indx % len(keys)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    if "manualPrompt" in promptMode:
        for memoryType in {"memory read", "memory write", "deletion upon memory", "insertion upon memory"}:
            result[memoryType] = []
    elif "autoPrompt_TwoTypes" in promptMode:
        for memoryType in magicWords:
            result[memoryType] = []
    elif "autoPrompt_FourTypes" in promptMode:
        for memoryType in magicWords:
            result[memoryType] = []

    for i in range(m):
        recieved = False
        singleResult = {}
        tryCnt = 0
        while not recieved:
            tryCnt += 1
            try:
                message = "Now we provide the specification description of the method " + methodSig + " in the class " + className + " as follows:\n"
                message += methodDoc + "\n"
                message += getQuestion(promptMode)
                answerLength = None
                typeList = None

                if "manualPrompt" in promptMode:
                    systemContent = getInitialPromptForMemoryOperationType(promptMode)
                    answerLength = 4
                    typeList = ["memory read", "memory write", "deletion upon memory", "insertion upon memory"]
                elif "autoPrompt_TwoTypes" in promptMode:
                    systemContent = getInitialPromptForMemoryOperationType(promptMode, magicWords)
                    answerLength = 2
                    typeList = ["memory read", "memory write"]
                elif "autoPrompt_FourTypes" in promptMode:
                    systemContent = getInitialPromptForMemoryOperationType(promptMode, magicWords)
                    answerLength = 4
                    typeList = ["memory read", "memory write", "deletion upon memory", "insertion upon memory"]
                else:
                    print("wrong setting")
                    exit(0)

                input = [
                    {"role": "system", "content": systemContent},
                    # {"role": "system", "content": ""},
                    {"role": "user", "content": message}
                ]

                # config.key_id, openai.api_key = getKey(config.key_id)

                openai.api_key = keys[keyIndex]

                if modelIndex == 4:
                    engine = "gpt-3.5-turbo-0301"
                elif modelIndex == 3:
                    engine = "gpt-3.5-turbo"
                elif modelIndex == 2:
                    engine = "gpt-3.5-turbo-0613"
                elif modelIndex == 1:
                    engine = "gpt-3.5-turbo-16k"
                else:
                    engine = "gpt-3.5-turbo-16k-0613"

                print(config.tagTokenCnt)
                # config.LLMTokenCnt += len(message.split(" ")) + len(systemContent.split(" "))
                # return None

                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=input,
                    # max_tokens=max_length,
                    temperature=t2,
                    # stop=stop
                )

                output = response["choices"][0]["message"]["content"]

                print("OUITPUT: " , len(encoding.encode(systemContent)) + len(encoding.encode(message)))
                config.LLMTokenCnt += len(encoding.encode(systemContent)) + len(encoding.encode(message))

                print("--------------------------------------------------------")
                print((config.global_m, config.global_n, config.global_t1, config.global_t2))
                print(message)
                print(output)
                print("--------------------------------------------------------")
                outputs = output.split(",")
                if len(outputs) != answerLength:
                    recieved = False
                    if tryCnt > 10:
                        for memoryType in typeList:
                            singleResult[memoryType] = False
                        break
                    continue

                recieved = True
                i = 0
                # for memoryType in ["memory read", "memory write", "deletion upon memory", "insertion upon memory"]:
                for memoryType in typeList:
                    if "no" in outputs[i] or "No" in outputs[i]:
                        singleResult[memoryType] = False
                    elif "yes" in outputs[i] or "Yes" in outputs[i]:
                        singleResult[memoryType] = True
                    else:
                        singleResult[memoryType] = None
                        recieved = False
                    i += 1
            except:
                error = sys.exc_info()[0]
                print("API error:", sys.exc_info())
                recieved = False
                if tryCnt > 10:
                    for memoryType in typeList:
                        singleResult[memoryType] = False
                        break
        for memoryType in singleResult:
            result[memoryType].append(singleResult[memoryType])

    for memoryType in result:
        result[memoryType] = Counter(result[memoryType]).most_common()
        if len(result[memoryType]) == 1:
            result[memoryType] = result[memoryType][0][0]
        else:
            result[memoryType] = random.choice(result[memoryType])[0]
    return result


if __name__ == '__main__':
    loadMagicWords, storeMagicWords = retrieveMagicWords("manualPrompt", 5)
    print(loadMagicWords)
    print(storeMagicWords)
