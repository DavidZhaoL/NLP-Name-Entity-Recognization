###imports

from collections import Counter
import sys
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import argparse

random.seed(11242)

depochs = 5
feat_red = 0


print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)


print("\nLoading the data \n")


"""Choose the algorithms """
parser=argparse.ArgumentParser()
parser.add_argument('-v','--viterbi', help="type is veiterbi",action="store_true")
parser.add_argument('-b','--beamSearch', help="type is beamSearch",action="store_true")
parser.add_argument("trainPath", help="get train path")
parser.add_argument("testPath", help="get test path")
args=parser.parse_args()
if args.viterbi:
    algorithm='v'
elif args.beamSearch:
    algorithm='b'
else:
    print('please select viterbi or beamSearch')
    exit(1);


"""Loading the data"""

### Load the dataset
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
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

train_data = load_dataset_sents(args.trainPath)
test_data = load_dataset_sents(args.testPath)

## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

""" Defining our feature space """

print("\nDefining the feature space \n")


# feature space of cw_ct
def cw_ct_counts(data, freq_thresh = 5): # data inputted as (cur_word, cur_tag)

    cw_c1_c = Counter()

    for doc in data:

        cw_c1_c.update(Counter(doc))

    return Counter({k:v for k,v in cw_c1_c.items() if v > freq_thresh})

cw_ct_count = cw_ct_counts(train_data, freq_thresh = feat_red)

# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts): # sent as (cur_word, cur_tag)

    phi_1 = Counter()

    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_count.keys()])

    return phi_1

sent = train_data[0]

phi_1(sent, cw_ct_count)

# feature space of pt-ct
def pt_ct_counts(data, freq_thresh = 5): # input (cur_word, cur_tag)

    tagtag = Counter()

    for doc in data:

        tags = list(zip(*doc))[1]

        for i in range(len(tags)):

            if i == 0:

                tagtag.update([("*", tags[i])])

            else:

                tagtag.update([(tags[i-1], tags[i])])

    # return feature space with features with counts above freq_thresh
    return Counter({k:v for k,v in tagtag.items() if v > freq_thresh})

pt_ct_count = pt_ct_counts(train_data, freq_thresh = feat_red)

# combining feature spaces
comb_featspaces = pt_ct_count + cw_ct_count

# creating our sentence features
def phi_2(sent, pt_ct_count):

    sentence, tags = zip(*sent)

    tags = ["*"] + list(tags)

    # returning features if found in the feature space
    tags = [(tags[i], tags[i+1]) for i in range(len(tags)-1) if (tags[i], tags[i+1]) in pt_ct_count]

    return Counter(tags)

sent = train_data[0]
phi_2(sent, pt_ct_count)



"""Perceprton"""
class Perceptron():

    def __init__(self,all_tags):
        super(Perceptron, self).__init__()
        self.all_tags = all_tags

    # creating all possible combinaions of
    def pos_combos(self,sentence):

        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags,repeat=len(sentence))]

        return combos

    #using viterbi
    def viterbi(self,doc,weights):
        sentence, tags = list(zip(*doc))

        rowNum=len(sentence) #get the sentence length

        allTag=["O", "PER", "LOC", "ORG", "MISC"]

        V=np.zeros((rowNum,len(allTag))) #store the score

        B=np.zeros((rowNum,len(allTag)),dtype=int) #store the back pointer

        #for the first word, initialise the first row of V
        for column in range(5):
            pair=(sentence[0],allTag[column]) #form the cw_ct pair
            if pair in weights: #if the pair in the cw_ct_count
                score=weights[pair]
            else:
                score=0
            V[0][column]=score #set the scoreof all the tag of first word
            B[0][column]=0 #set 0 to the backpointer fo first word

        #traverse all the words
        for row in range(1,rowNum):
            for column in range(5):
                pair=(sentence[row],allTag[column]) #form the cw_ct pair

                back=np.argmax(V[row-1]) #set the backpointer for each term

                if pair in weights:
                    score=weights[pair]+max(V[row-1]) #get the score for each term by adding max previous one
                else:
                    score=max(V[row-1])

                V[row][column]=score #store the score
                B[row][column]=back #store the back pointer

        tagSequence=[] #the aim tag sequence

        lastTag=np.argmax(V[rowNum-1]) #get the biggest score overall

        tagSequence.insert(0,allTag[lastTag])

        row=rowNum-1
        column=lastTag

        while row>0: #find from back to front using backpointer
            tagSequence.insert(0,allTag[B[row][column]]) #append to the tag sequence
            column=B[row][column]
            row-=1

        predictTag=list(zip(sentence,tagSequence))
        return predictTag


    #using beam search
    def beamSearch(self,doc,weights,width):
        sentence, tags = list(zip(*doc))

        length=len(sentence)

        allTag=["O", "PER", "LOC", "ORG", "MISC"]

        listB=[] #keep the top k
        listB1=[] #store all possible for the next turn

        #for the first word, doesn't have previous tag
        for tag in allTag:
            pair=(sentence[0],tag)
            score=weights[pair]
            tagWithScore=(tag,score)
            listB1.append(tagWithScore)

        listB=sorted(listB1, key=lambda x:x[1], reverse=True)[:width]

        #traverse each word
        for wordIndex in range(1,length):

            listB1=[] #store all possible for the next turn

            for previousTag in listB:

                for tag in allTag:
                    currentTagSeq=previousTag[0]+' '+tag #for b in listB append each tag in all tag

                    pair=(sentence[wordIndex],tag) #current word with tag

                    score=previousTag[1]+weights[pair] #get the score

                    tagWithScore=(currentTagSeq,score) #combine the tag sequence wuth score

                    listB1.append(tagWithScore) #store in listB1

            listB=sorted(listB1, key=lambda x:x[1], reverse=True)[:width] # get top k

        #get the tag sequence with highest score
        tagSequence=sorted(listB, key=lambda x:x[1], reverse=True)[:1][0][0].split()

        predictTag=list(zip(sentence,tagSequence))

        return predictTag


    def train_perceptron(self, data, epochs, algori_type, shuffle = True, extra_feat = False):

        # variables used as metrics for performance and accuracy
        iterations = range(len(data)*epochs)
        false_prediction = 0
        false_predictions = []

        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()

        start = time.time()

        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()

            # going through each sentence-tag_seq pair in training_data

            # shuffling if necessary
            if shuffle == True:

                random.shuffle(data)

            for doc in data:
                # retrieve the highest scoring sequence

                #max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)

                if algori_type=='v':
                    max_scoring_seq=self.viterbi(doc,weights)
                elif algori_type=='b':
                    max_scoring_seq=self.beamSearch(doc,weights,1)
                else:
                    print('Please choose viterbi or beamSearch')
                    exit(1)


                # if the prediction is wrong
                if max_scoring_seq != doc:

                    correct = Counter(doc)

                    # negate the sign of predicted wrong
                    predicted = Counter({k:-v for k,v in Counter(max_scoring_seq).items()})

                    # add correct
                    weights.update(correct)

                    # negate false
                    weights.update(predicted)


                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)


            print("Epoch: ", epoch+1,
                  " / Time for epoch: ", round(time.time() - now,2),
                 " / No. of false predictions: ", false)


        return weights, false_predictions, iterations

    # testing the learned weights
    def test_perceptron(self,data, weights, algori_type, extra_feat = False):

        correct_tags = []
        predicted_tags = []

        i = 0

        for doc in data:

            _, tags = list(zip(*doc))

            correct_tags.extend(tags)

            if algori_type=='v':
                max_scoring_seq=self.viterbi(doc,weights)
            elif algori_type=='b':
                max_scoring_seq=self.beamSearch(doc,weights,1)
            else:
                print('Please choose viterbi or beamSearch')
                exit(1)

            _, pred_tags = list(zip(*max_scoring_seq))

            predicted_tags.extend(pred_tags)

        return correct_tags, predicted_tags

    def evaluate(self,correct_tags, predicted_tags):

        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=self.all_tags)

        print("F1 Score: ", round(f1, 5))

        return f1


perceptron = Perceptron(all_tags)

print("\nTraining the perceptron with (cur_word, cur_tag) \n")

weights, false_predictions, iterations = perceptron.train_perceptron(train_data, depochs,algorithm)

print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")

correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights,algorithm)

f1 = perceptron.evaluate(correct_tags, predicted_tags)
