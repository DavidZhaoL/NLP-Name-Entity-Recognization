
"""
Created on Tue Mar 19 14:00 2019

lab3.py: a script to predict name entity

Instruction: I used 4 features: word_tag, tag_tag, tag_tag_tag, wordType_tag

             self.weight_oneFea: the weight using feature1: word_tag
             self.weight_twoFea: the weight using feature1+feature2: word_tag + tag_tag
             self.weight_threeFea: the weight using feature1+feature2+feature3: word_tag + tag_tag + trigram_tag
             self.weight_fourFea: the weight using feature1+feature2+feature3+feature4: word_tag + tag_tag + trigram_tag + wordType_tag

@author: Lei Zhao
"""

from sklearn.metrics import f1_score
import re,getopt,sys,random
from collections import Counter
import numpy as np
import time

#get commandLine input
class CommandLine:

    def __init__(self):
        opts,args=getopt.getopt(sys.argv[1:],'')
        self.trainfile=args[0]
        self.testfile=args[1]

    def getFilePath(self):
        return self.trainfile, self.testfile


class Perceptron:

    def __init__(self,trainFile,testFile):

        self.trainSet=self.load_dataset_sents(trainFile) # get the dataset from trainFile after preprocess
        self.testSet=self.load_dataset_sents(testFile) # get the dataset from testFile after preprocess

        self.trainSetSepa=self.load_dataset_sents(trainFile,as_zip=False)
        self.testSetSepa=self.load_dataset_sents(testFile,as_zip=False)

        self.weight_oneFea={} #the weight for one feature
        self.weight_twoFea={} #the weight for two feature
        self.weight_threeFea={} #the weight for three feature
        self.weight_fourFea={} #the weight for four feature

        self.maxIter=5 #iterate 5 times

        self.tagSeqList=[] #all possible candidate tag sequence


    # preprocess data
    def load_dataset_sents(self,file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
        targets = []
        inputs = []
        zip_inps = []
        with open(file_path) as f:
            for line in f:
                sent, tags = line.split('\t')
                words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
                ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]

                inputs.append(words)
                targets.append(ner_tags)
                zip_inps.append(list(zip(words, ner_tags)))

        return zip_inps if as_zip else (inputs, targets)

#=======================feature1============================feature1===========================================
    #get the current word-label counts in corpus
    def wordCurrentCount(self):
        listTrainSet=[]
        for sentence in self.trainSet:
            for pair in sentence:
                listTrainSet.append(pair)

        cw_cl_counts=Counter(listTrainSet)

        self.my_cw_cl_counts={k for k,v in cw_cl_counts.items() if v>=3}

        self.weight_oneFea={k:0 for k in self.my_cw_cl_counts}

    #get the counts for feature1 word_tag
    def phi_1(self,x,y,corpus_w_t_dict):
        word_word_list=list(zip(x,y))
        w_w_count=Counter(word_word_list)

        w_w_count_new={}
        w_w_count_new={k:v for k,v in w_w_count.items() if k in corpus_w_t_dict}
        return w_w_count_new

#=====================feature2========================feature2================================================
    #get the current tag tag counts  in corpus
    def tag_tagCount(self):
        listTrainSet=[]
        for sentence in self.trainSet:
            sentenceTag=[pair[1] for pair in sentence].copy()
            sentenceTag.insert(0,None)
            bigramList=list(zip(*[sentenceTag[i:] for i in range(2)]))
            for bigram in bigramList:
                listTrainSet.append(bigram)
        cl_cl_counts=Counter(listTrainSet)

        self.my_cl_cl_counts={k for k,v in cl_cl_counts.items() if v>=3}

        weight_twoFeature={k:0 for k in self.my_cl_cl_counts}
        self.weight_twoFea={**self.weight_oneFea,**weight_twoFeature}


    #get the counts for feature2 tag_tag
    def phi_2(self,y,corpus_t_t_dict):

        tagSeq=y.copy()
        tagSeq.insert(0,None)
        bigramList=list(zip(*[tagSeq[i:] for i in range(2)]))

        t_t_count=Counter(bigramList)
        t_t_count_new={k:v for k,v in t_t_count.items() if k in corpus_t_t_dict}
        return t_t_count_new

#=======================feature3===========================feature3=============================================
    #get the current tag tag tag counts  in corpus
    def trigramCount(self):
        listTrainSet=[]
        for sentence in self.trainSet:
            sentenceTag=[pair[1] for pair in sentence].copy()
            sentenceTag.insert(0,None)
            sentenceTag.append(None)
            trigramList=list(zip(*[sentenceTag[i:] for i in range(3)]))
            for trigram in trigramList:
                listTrainSet.append(trigram)

        trigram_counts=Counter(listTrainSet)

        self.my_trigram_counts={k for k,v in trigram_counts.items() if v>=5}

        weight_threeFeature={k:0 for k in self.my_trigram_counts}
        self.weight_threeFea={**self.weight_twoFea,**weight_threeFeature}


    #get the counts for feature3 tag_tag_tag
    def phi_3(self,y,trigramDict):

        tagSeq=y.copy()
        tagSeq.insert(0,None)
        tagSeq.append(None)
        trigramList=list(zip(*[tagSeq[i:] for i in range(3)]))

        trigramCount=Counter(trigramList)
        trigramCount_new={k:v for k,v in trigramCount.items() if k in trigramDict}
        return trigramCount_new

#=======================feature4======================feature4=================================================

    #Divide each word into three types: allUpper,initialsUpper or others
    #this is for feature4
    def getWordType(self,word):
        if word.isupper():
            return 'allUpper'
        elif word[0].isupper():
            return 'initialsUpper'
        #elif not word.isalpha():
        #    return 'noAlpha'
        else:
            return 'otherCase'

    #get the current wordType-label counts in corpus
    def wordTypeCurrentCount(self):
        listTrainSet=[]
        for sentence in self.trainSet:
            for pair in sentence:
                wordType=self.getWordType(pair[0])
                listTrainSet.append((wordType,pair[1]))

        type_tag_counts=Counter(listTrainSet)

        self.type_tag_countSet={k for k,v in type_tag_counts.items() if v>=6}

        weight_fourFea={k:0 for k in self.type_tag_countSet}
        self.weight_fourFea={**self.weight_threeFea,**weight_fourFea}

    #get the counts for feature4 wordType_tag
    def phi_4(self,x,y,wordTypeDict):
        wordType_tag_list=[]
        for i in range(len(x)):
            type=self.getWordType(x[i])
            wordType_tag_list.append((type,y[i]))
        wt_t_count=Counter(wordType_tag_list)

        wt_t_count_new={k:v for k,v in wt_t_count.items() if k in wordTypeDict}
        return wt_t_count_new

#=============================================================================================================
    #get y with highest argmax value
    def getArgmax(self,wordSeq):

        senLength=len(wordSeq) #get the tag sequence length
        tagCandidateSet=self.tagSeqList[senLength] #get all possible candidate tag sequence

        scoreFea1List=np.zeros(len(tagCandidateSet)) #store the score of all tag sequence for one feature
        scoreFea2List=np.zeros(len(tagCandidateSet)) #store the score of all tag sequence for two feature
        scoreFea3List=np.zeros(len(tagCandidateSet)) #store the score of all tag sequence for two feature
        scoreFea4List=np.zeros(len(tagCandidateSet)) #store the score of all tag sequence for two feature

        for index in range(len(tagCandidateSet)): #iterate all possible tag sequence

            scoreFeature1=0 #initialise each scpre as 0
            scoreFeature2=0
            scoreFeature3=0
            scoreFeature4=0

            tagSeq=tagCandidateSet[index] #get the tag sequence

            feature1=self.phi_1(wordSeq,tagSeq,self.my_cw_cl_counts) #get the feature1 w_t of this tag
            feature2=self.phi_2(tagSeq,self.my_cl_cl_counts) #get the feature2 t_t of this tag
            feature3=self.phi_3(tagSeq,self.my_trigram_counts) #get the feature3 tag_tag_tag of this tag
            feature4=self.phi_4(wordSeq,tagSeq,self.type_tag_countSet) #get the feature4 wordType_tag of this tag

            #get the score using different feature
            for key in feature1.keys():
                scoreFeature1+=self.weight_oneFea[key]*feature1[key]
                scoreFeature2+=self.weight_twoFea[key]*feature1[key]
                scoreFeature3+=self.weight_threeFea[key]*feature1[key]
                scoreFeature4+=self.weight_fourFea[key]*feature1[key]

            for key in feature2.keys():
                scoreFeature2+=self.weight_twoFea[key]*feature2[key]
                scoreFeature3+=self.weight_threeFea[key]*feature2[key]
                scoreFeature4+=self.weight_fourFea[key]*feature2[key]

            for key in feature3.keys():
                scoreFeature3+=self.weight_threeFea[key]*feature3[key]
                scoreFeature4+=self.weight_fourFea[key]*feature3[key]

            for key in feature4.keys():
                scoreFeature4+=self.weight_fourFea[key]*feature4[key]

            #add this tagSequence's score to the score list
            scoreFea1List[index]=scoreFeature1
            scoreFea2List[index]=scoreFeature2
            scoreFea3List[index]=scoreFeature3
            scoreFea4List[index]=scoreFeature4

        #predicted tag sequence using the argmax
        prediFeature1=tagCandidateSet[np.argmax(scoreFea1List)]
        prediFeature2=tagCandidateSet[np.argmax(scoreFea2List)]
        prediFeature3=tagCandidateSet[np.argmax(scoreFea3List)]
        prediFeature4=tagCandidateSet[np.argmax(scoreFea4List)]

        return prediFeature1,prediFeature2,prediFeature3,prediFeature4

    #update the weight_oneFea
    def updateWeight1(self,wordSeq,predictTag,trueTag):
        predFeature=self.phi_1(wordSeq,predictTag,self.my_cw_cl_counts)
        trueFeature=self.phi_1(wordSeq,trueTag,self.my_cw_cl_counts)

        for key in predFeature.keys():#minus wrong predicted tagSeq feature1
            self.weight_oneFea[key]-=predFeature[key]

        for key in trueFeature.keys():
            self.weight_oneFea[key]+=trueFeature[key]


    #update the weight_twoFea
    def updateWeight2(self,wordSeq,predictTag,trueTag):
        predFeature1=self.phi_1(wordSeq,predictTag,self.my_cw_cl_counts)
        trueFeature1=self.phi_1(wordSeq,trueTag,self.my_cw_cl_counts)

        predFeature2=self.phi_2(predictTag,self.my_cl_cl_counts)
        trueFeature2=self.phi_2(trueTag,self.my_cl_cl_counts)

        for key in predFeature1.keys():#minus wrong predicted tagSeq feature1
            self.weight_twoFea[key]-=predFeature1[key]

        for key in trueFeature1.keys():#add true tagSeq feature1
            self.weight_twoFea[key]+=trueFeature1[key]

        for key in predFeature2.keys():#minus wrong predicted tagSeq feature2
            self.weight_twoFea[key]-=predFeature2[key]

        for key in trueFeature2.keys():#add true tagSeq feature2
            self.weight_twoFea[key]+=trueFeature2[key]

    #update the weight_threeFea
    def updateWeight3(self,wordSeq,predictTag,trueTag):
        predFeature1=self.phi_1(wordSeq,predictTag,self.my_cw_cl_counts)
        trueFeature1=self.phi_1(wordSeq,trueTag,self.my_cw_cl_counts)

        predFeature2=self.phi_2(predictTag,self.my_cl_cl_counts)
        trueFeature2=self.phi_2(trueTag,self.my_cl_cl_counts)

        predFeature3=self.phi_3(predictTag,self.my_trigram_counts)
        trueFeature3=self.phi_3(trueTag,self.my_trigram_counts)

        for key in predFeature1.keys(): #minus wrong predicted tagSeq feature1
            self.weight_threeFea[key]-=predFeature1[key]

        for key in trueFeature1.keys(): #add true tagSeq feature1
            self.weight_threeFea[key]+=trueFeature1[key]

        for key in predFeature2.keys():#minus wrong predicted tagSeq feature2
            self.weight_threeFea[key]-=predFeature2[key]

        for key in trueFeature2.keys():#add true tagSeq feature2
            self.weight_threeFea[key]+=trueFeature2[key]

        for key in predFeature3.keys():#minus wrong predicted tagSeq feature3
            self.weight_threeFea[key]-=predFeature3[key]

        for key in trueFeature3.keys():#add true tagSeq feature3
            self.weight_threeFea[key]+=trueFeature3[key]

    #update the weight_fourFea
    def updateWeight4(self,wordSeq,predictTag,trueTag):
        predFeature1=self.phi_1(wordSeq,predictTag,self.my_cw_cl_counts)
        trueFeature1=self.phi_1(wordSeq,trueTag,self.my_cw_cl_counts)

        predFeature2=self.phi_2(predictTag,self.my_cl_cl_counts)
        trueFeature2=self.phi_2(trueTag,self.my_cl_cl_counts)

        predFeature3=self.phi_3(predictTag,self.my_trigram_counts)
        trueFeature3=self.phi_3(trueTag,self.my_trigram_counts)

        predFeature4=self.phi_4(wordSeq,predictTag,self.type_tag_countSet)
        trueFeature4=self.phi_4(wordSeq,trueTag,self.type_tag_countSet)

        for key in predFeature1.keys():#minus wrong predicted tagSeq feature1
            self.weight_fourFea[key]-=predFeature1[key]

        for key in trueFeature1.keys():#add true tagSeq feature1
            self.weight_fourFea[key]+=trueFeature1[key]

        for key in predFeature2.keys():#minus wrong predicted tagSeq feature2
            self.weight_fourFea[key]-=predFeature2[key]

        for key in trueFeature2.keys():#add true tagSeq feature2
            self.weight_fourFea[key]+=trueFeature2[key]

        for key in predFeature3.keys():#minus wrong predicted tagSeq feature3
            self.weight_fourFea[key]-=predFeature3[key]

        for key in trueFeature3.keys():#add true tagSeq feature3
            self.weight_fourFea[key]+=trueFeature3[key]

        for key in predFeature4.keys():#minus wrong predicted tagSeq feature4
            self.weight_fourFea[key]-=predFeature4[key]

        for key in trueFeature4.keys():#add true tagSeq feature4
            self.weight_fourFea[key]+=trueFeature4[key]


    def train(self):
        c=0 #initialise counter as 0 in order to average weight
        listTrainSet=self.trainSet #get trainSet

        sum_weight_featu1={k:0 for k in self.weight_oneFea.keys()} #initialise this dict to sum weight_one each time
        sum_weight_featu2={k:0 for k in self.weight_twoFea.keys()} #initialise this dict to sum weight_two each time
        sum_weight_featu3={k:0 for k in self.weight_threeFea.keys()} #initialise this dict to sum weight_three each time
        sum_weight_featu4={k:0 for k in self.weight_fourFea.keys()} #initialise this dict to sum weight_three each time

        for num in range(self.maxIter): #iterate 5 times
            random.Random(10).shuffle(listTrainSet) #shuffle
            for sentence in listTrainSet:
                wordSeq=[pair[0] for pair in sentence] #get the word sequence
                tagY=[pair[1] for pair in sentence] #get the true tag sequence

                prediFeature1,prediFeature2,prediFeature3,prediFeature4=self.getArgmax(wordSeq) #get y with highest argmax value

                if(prediFeature1!=tagY):
                    self.updateWeight1(wordSeq,prediFeature1,tagY) #update the weight_oneFea

                if(prediFeature2!=tagY):
                    self.updateWeight2(wordSeq,prediFeature2,tagY) #update the weight_twoFea

                if(prediFeature3!=tagY):
                    self.updateWeight3(wordSeq,prediFeature3,tagY) #update the weight_threeFea

                if(prediFeature4!=tagY):
                    self.updateWeight4(wordSeq,prediFeature4,tagY) #update the weight_fourFea

                # multipass and sum
                for key in sum_weight_featu1.keys(): #sum the weight_oneFea
                    sum_weight_featu1[key]+=self.weight_oneFea[key]

                for key in sum_weight_featu2.keys(): #sum the weight_twoFea
                    sum_weight_featu2[key]+=self.weight_twoFea[key]

                for key in sum_weight_featu3.keys(): #sum the weight_threeFea
                    sum_weight_featu3[key]+=self.weight_threeFea[key]

                for key in sum_weight_featu4.keys(): #sum the weight_fourFea
                    sum_weight_featu4[key]+=self.weight_fourFea[key]
                c+=1 #counter plus 1

        #average the weight
        for key in sum_weight_featu1.keys(): #average weight1
            self.weight_oneFea[key]= sum_weight_featu1[key]/c

        for key in sum_weight_featu2.keys(): #average weight2
            self.weight_twoFea[key]= sum_weight_featu2[key]/c

        for key in sum_weight_featu3.keys():
            self.weight_threeFea[key]=sum_weight_featu3[key]/c

        for key in sum_weight_featu4.keys():
            self.weight_fourFea[key]=sum_weight_featu4[key]/c


    #prediction for test dataset
    def test(self):
        listTestSet=self.testSet

        fscore1=0
        fscore2=0
        fscore3=0
        fscore4=0

        correct=[]
        prediction1=[]
        prediction2=[]
        prediction3=[]
        prediction4=[]

        for sentence in listTestSet:
            wordSeq=[pair[0] for pair in sentence]
            tagY=[pair[1] for pair in sentence]

            #get tag sequence with highest argmax value
            prediFeature1,prediFeature2,prediFeature3,prediFeature4=self.getArgmax(wordSeq)

            correct+=tagY
            prediction1=prediction1+prediFeature1 #prediction using weight_oneFea
            prediction2=prediction2+prediFeature2 #prediction using weight_twoFea
            prediction3=prediction3+prediFeature3 #prediction using weight_threeFea
            prediction4=prediction4+prediFeature4 #prediction using weight_fourFea

        fscore1=f1_score(correct, prediction1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
        fscore2=f1_score(correct, prediction2, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
        fscore3=f1_score(correct, prediction3, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
        fscore4=f1_score(correct, prediction4, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])

        print('fscore using one feature: ',fscore1,'\n')
        print('fscore using two feature: ',fscore2,'\n')
        print('fscore using three feature: ',fscore3,'\n')
        print('fscore using four feature: ',fscore4,'\n')

    # start to run the perceptron class
    def implement(self):
        self.tagSeqList=self.getAllPossiTagSeq() #get all possible tag sequence

        self.wordCurrentCount() #get the current word-label counts in corpus
        self.tag_tagCount() #get the current label-label counts in corpus
        self.trigramCount() #get the current label-label-label counts in corpus
        self.wordTypeCurrentCount() #get the current wordType-label counts in corpus

        self.train() #using training data to train weight
        self.test() #using test data to test weight
        self.printTopFeature()


    # get all candidate tag sequence for each words sequence by length
    def candidateTagSet(self,length):
        tagSet=('O','PER','LOC','ORG','MISC')
        n,tagList=1,[['O'],['PER'],['LOC'],['ORG'],['MISC']]

        while n<length:
            CurrentTagList=[]
            for item in tagList:
                for tag in tagSet:
                    currentTag=item.copy()
                    currentTag.append(tag)
                    CurrentTagList.append(currentTag)
            tagList=CurrentTagList
            n+=1
        return tagList

    # store all possible tag sequence in case that calculate that every time
    def getAllPossiTagSeq(self):
        tagSeqList=[0]
        tagSeqList.append(self.candidateTagSet(1))
        tagSeqList.append(self.candidateTagSet(2))
        tagSeqList.append(self.candidateTagSet(3))
        tagSeqList.append(self.candidateTagSet(4))
        tagSeqList.append(self.candidateTagSet(5))
        tagSeqList.append(self.candidateTagSet(6))
        tagSeqList.append(self.candidateTagSet(7))

        return tagSeqList

    # get and print top10 for each tag
    def printTopFeature(self):
        self.getTopForEachTag(self.weight_oneFea,'one')
        self.getTopForEachTag(self.weight_twoFea,'two')
        self.getTopForEachTag(self.weight_threeFea,'three')
        self.getTopForEachTag(self.weight_fourFea,'four')

    #split weight by tag and get top10 for each feature
    def getTopForEachTag(self,weight,featureCount):
        dictPER={}
        dictLOC={}
        dictORG={}
        dictMISC={}
        dictO={}
        for key,v in weight.items(): #separate the weight by tag
            if key[-1]=='PER':
                dictPER[key]=v
            elif key[-1]=='LOC':
                dictLOC[key]=v
            elif key[-1]=='ORG':
                dictORG[key]=v
            elif key[-1]=='MISC':
                dictMISC[key]=v
            elif key[-1]=='O':
                dictO[key]=v
            else:
                continue

        topFeaturePER=sorted(dictPER.items(),reverse=True,key=lambda x:x[1])
        topFeatureLOC=sorted(dictLOC.items(),reverse=True,key=lambda x:x[1])
        topFeatureORG=sorted(dictORG.items(),reverse=True,key=lambda x:x[1])
        topFeatureMISC=sorted(dictMISC.items(),reverse=True,key=lambda x:x[1])
        topFeatureO=sorted(dictO.items(),reverse=True,key=lambda x:x[1])

        pos10_PER=[item[0][0] for item in topFeaturePER[:10]]
        pos10_LOC=[item[0][0] for item in topFeatureLOC[:10]]
        pos10_ORG=[item[0][0] for item in topFeatureORG[:10]]
        pos10_MISC=[item[0][0] for item in topFeatureMISC[:10]]
        pos10_O=[item[0][0] for item in topFeatureO[:10]]

        print('The top 10 for each class using ',featureCount,'feature is :')
        print('For PER: ',pos10_PER,'\n')
        print('For LOC: ',pos10_LOC,'\n')
        print('For ORG: ',pos10_ORG,'\n')
        print('For MISC: ',pos10_MISC,'\n')
        print('For O: ',pos10_O,'\n','\n','\n','\n')


#=================main=============================================================================
if __name__=="__main__":

    timeStart=time.time()
    config=CommandLine()
    trainFile,testFile=config.getFilePath()

    myPercep=Perceptron(trainFile,testFile)
    myPercep.implement()

    timeEnd=time.time()
    print('Duration is ',timeEnd-timeStart)
