import numpy as np
import math
import sys

"""
Constants:
SPECIALS: characters to remove in data cleaning
UNIQUE_CHARS: number of valid characters in a position in a trigram
FIRST_N : first n languages to show from most similar to least similar 
    for each test document.  
"""
SPECIALS = set('!@#$%^&*()[]{};:,./<>?\\|`~-=_+\'\"\t\n0123456789')
UNIQUE_CHARS = 27
FIRST_N = 4


"""
Data cleaning for a single character, turns uppercase letters to lowercase. 
Returns empty string for special characters. Also returns spaces for spaces.
"""


def clean_char(char: str) -> str:
    if char in SPECIALS:
        return ""
    elif char.isupper():
        return char.lower()
    else:
        if not char.islower() and (char != ' '):
            raise ValueError("Invalid character ", char)
        return char


"""
Indexing trigram according to base 27 conversion:
    Imagine each trigram is a 3 digit base 27 number, with the following values.
        ' ' spaces are 0
        'a' through 'z' are 1 to 26, respectively.
    Note: the base can change depending on what the unique character constant is. 
We take each of these trigrams and return an index that is the trigram's base 27 
number after decimal expansion. This guarantees consistent order that takes into 
account ordering within a trigram, while ensuring we have exactly the number of 
unique indices we need. 
"""


def index_trigram(trigram: list) -> int:
    if len(trigram) != 3:
        raise ValueError("Invalid Trigram: ", str(trigram))
    index = 0
    for i in range(3):
        if trigram[i] == ' ':
            index += 0
        elif trigram[i].islower():
            index += (ord(trigram[i]) - 96) * math.pow(UNIQUE_CHARS, 2 - i)
        else:
            raise ValueError("Invalid trigram entry %s", trigram[i])
    return index


"""
Trigram building method, returns counts vector as a list of integers
of each character trigram in the document. Used by both training and
testing. Note, this is NOT normalized data, normalization is done when 
by other methods when they receive the information returned by this. 
"""


def build_trigrams(document: str) -> list:
    counts = []
    # populate counts for zeroes for all possible unique trigrams
    # given the number of unique characters.
    for i in range(int(math.pow(UNIQUE_CHARS, 3))):
        counts.append(0)

    trigram = []
    # advance to the third character, and let trigram[0] be first char
    # trigram[1] be second char, and let trigram[2] be third character
    char_counter = 0
    line_counter = 0
    with open(document) as read_file:
        for read_line in read_file:
            for c in read_line:
                # no normalization in here, but cleaning data here
                cleaned = clean_char(c)
                if cleaned == "":
                    # skip character if we find that its a special
                    continue

                # set first trigram manually
                if line_counter == 0 and (char_counter == 0 or
                                          char_counter == 1 or
                                          char_counter == 2):
                    trigram.append(cleaned)
                    char_counter += 1
                else:
                    # at each character, move previous chars back one step, and then
                    # let new character be the last.
                    trigram[0], trigram[1] = trigram[1], trigram[2]
                    trigram[2] = cleaned
                # each time we get a new trigram, use decimal expansion to compute an
                # appropriate index and add 1 to that items frequency.
                if char_counter >= 3:
                    counts[int(index_trigram(trigram))] += 1
            line_counter += 1
        # if we ever have the edge case of a non-full trigram, we return nothing
        if len(trigram) < 3:
            return []
    return counts


"""
Training method:
Update the trained_vectors for the given language and document. This will
be collecting the non-normalized data, and storing the number of 
trigrams associated with each language for the sake of 
normalization later. 
"""


def training(language: str, document: str,
             trained_vectors: dict, trigram_counts: dict):
    vector = build_trigrams(document)
    for i in range(len(vector)):
        if trained_vectors.get(language) is None:
            trained_vectors[language] = vector
        else:
            trained_vectors.get(language)[i] += vector[i]
        trigram_counts[language] += vector[i]


"""
Run training procedures on all languages and documents to train with.

Keeps track of non-normalized trigram counts for each language 
and then returns the normalized trigram frequency dictionary, with languages
mapped to frequency vectors.  
"""


def train_all(training_map: dict) -> dict:
    # handle training for all languages
    trained_vectors = dict()
    trigram_counts = dict()
    normalized_trained = dict()
    for language in training_map.keys():
        trigram_counts[language] = 0
        for file_name in training_map[language]:
            training(language, file_name, trained_vectors, trigram_counts)
        # normalize for a given language's trigram counts
        normalized_trained[language] = [float(i / trigram_counts[language])
                                        for i in trained_vectors[language]]
        # check for normalization, doesn't add to time complexity
        for i in normalized_trained[language]:
            if (i > 1) or (i < 0):
                raise ValueError("Invalid frequency")

    return normalized_trained


"""
Similarity metric function for two vectors v1 and v2
Currently using cosine similarity.

Takes in two lists and returns a float, so all we need to do
to replace this function in our code is design another function
with that signature and let testing take in the new function.
"""


def cosine_similarity(v1: list, v2: list) -> float:
    a1 = np.array(v1)
    a2 = np.array(v2)
    return np.dot(a1, a2)/(math.sqrt(np.dot(a1, a1)) * math.sqrt(np.dot(a2, a2)))


"""
Testing method:
Gets a file to be tested, all normalized trained vectors, a similarity
function then returns the language and the most likely languages as a 
dictionary with language names mapped to similarity scores. 

Note, similarity_func is a callable, so if we want a new similarity func
it just needs the same signature as cosine_similarity, and pass that to
this function.
"""


def testing(document: str, normalized_trained: dict, similarity_func: callable) -> dict:
    # get non-normalized data on trigrams
    non_normed = build_trigrams(document)
    num_trigrams = sum(non_normed)
    normed_vector = [float(entry / num_trigrams) for entry in non_normed]
    similarity_scores = dict()
    # go through all languages, run similarity_func on them
    for language in normalized_trained.keys():
        similarity_scores[language] = \
            similarity_func(normed_vector, normalized_trained[language])
    return similarity_scores


"""
Prints out the language stats for a given test document, 
specifically their four most likely languages
and the respective scores. Prints into the output file.

output : file object (can't type hint for that). 
"""


def format_test_data(output, document: str, similarity_scores: dict):
    output.write(document + "\n")
    sorted_names = sorted(similarity_scores,
                          key=lambda a: similarity_scores.get(a), reverse=True)
    sorted_scores = sorted(similarity_scores.values(), reverse=True)
    for i in range(min(FIRST_N, len(sorted_scores))):
        output.write("\t%s %d\n" % (sorted_names[i], int(100 * sorted_scores[i])))


"""
Compile the input into two objects, training_map, and test_list
for the rest of the program to handle. This function handles 
assigning file names and language names to training and testing.
"""


def compile_input(input_name: str, training_map: dict, test_list: list):
    with open(input_name, 'r') as input_file:
        for line in input_file:
            if line == "\n":
                continue
            new_line = line.strip().split()

            if new_line[0] == "Unknown":
                test_list.append(new_line[1])
            else:
                if new_line[0] not in training_map.keys():
                    training_map[new_line[0]] = []
                training_map[new_line[0]].append(new_line[1])


"""
Main method:

collect info from input file to run training with to get normalized
trained vectors, then run testing with normalized data for all 
non-labeled documents. 

"""
if __name__ == "__main__":

    # collect all documents to train with, map training documents to language
    # and collect all documents to test

    # keys: languages, values: list of respective filenames to train with
    training_map = dict()
    # list of filenames to test
    to_test = []
    compile_input(str(sys.argv[1]), training_map, to_test)

    # run training with the languages and documents to train
    normalized_trained = train_all(training_map)

    # name of output file
    output_file_name = str(sys.argv[2])

    # testing
    with open(output_file_name, 'w') as file_obj:
        for doc_name in to_test:
            # run tests and format it
            format_test_data(file_obj, doc_name,
                             testing(doc_name, normalized_trained, cosine_similarity))
