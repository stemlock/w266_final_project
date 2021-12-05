import re
import random

def preprocess_unk(x, vocab):
    
    '''
    Preprocesses a given text and replaces all vocab words with [UNK] token.
    If no replacements are found, return an empty string.
    '''
    
    # Compile the regex object using the provided vocab
    regex = re.compile('|'.join(r'\b%s\b' %s for s in map(re.escape, vocab)))    
    text = x['review_text'].lower()
    
    # Substitute all words in vocab with [UNK]
    new_text, sub_count = regex.subn("[UNK]", text)
    
    # If no vocab words were found, set new_text to an empty string
    if new_text == text:
        new_text = ''
    
    return new_text, sub_count

def gender_mapping(f_vocab, m_vocab):
    
    '''
    Takes in both a vocabulary of female and male gendered words
    that are interchangeable by index and creates both a dictionary mapping
    from female->male and from male->female.
    '''
    
    f_mapping = {}
    m_mapping = {}
    
    # For each word in the female vocab, create the associated male word mappings
    for i,f in enumerate(f_vocab):
        m = m_vocab[i]
        
        # If the word is not yet in the mapping dictionary, create the key
        if f_mapping.get(f,0) == 0:
            f_mapping[f] = [m]
        # If the word is already in the dictionary, append the additional mapping
        else:
            f_mapping[f].append(m)
            
    # For each word in the male vocab, create the associated female word mappings
    for i,m in enumerate(m_vocab):
        f = f_vocab[i]
        
        # If the word is not yet in the mapping dictionary, create the key
        if m_mapping.get(m,0) == 0:
            m_mapping[m] = [f]
        # If the word is already in the dictionary, append the additional mapping
        else:
            m_mapping[m].append(f)
            
    return f_mapping, m_mapping

def preprocess_gendered_swap(x, vocab_map, regex):
    
    '''
    Preprocess a given text by swapping out all words that occur in the 
    vocab map with an associated string (swap words:asssociated strings
    provided as a key:value pair), and return the new string. The regex 
    should be compiled with all words in the vocab that wish to be 
    swapped out.
    '''
    
    i = 0
    sub_count = 0
    new_text = ''
    text = x['review_text'].lower()
    
    # While the whole text has not been searched, check for words to swap
    while i < len(text):
        match = regex.search(text[i:])
        
        # If a word to swap is found, swap it with a randomly selected associated word
        # And add the swapped subtext to the new text
        if match:
            word = text[i+match.start():i+match.end()]
            sub = random.choice(vocab_map[word])
            new_text += (text[i:i+match.start()] + sub)
            
            # Increment the counter of where to start the next search
            prev = i
            i = match.end() + prev

            # Increment the counter for number of words substituted
            sub_count += 1
            
        # If a word to swap is not found, append the rest of the subtext
        else:
            new_text += text[i:]
            i = len(text)
    
    return new_text, sub_count
