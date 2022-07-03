import re
import numpy as np
import pronouncing

def mask_rhyme_loc(verse, newline_token = '\n', mask_token = '<mask>',
                   rhyme_freq = 2, from_last = True):
    verse = verse.strip(newline_token)
    lines = verse.split(newline_token)
    lines = [re.split(r'\s+', line.strip()) for line in lines]
    n_lines = len(lines)
    
    if from_last:
        line_ind = [x for x in range(n_lines-1, 0, -rhyme_freq)]
    else:
        line_ind = [x for x in range(0, n_lines-1, rhyme_freq)]

    for ind in line_ind:
        lines[ind][-1] = mask_token
    
    lines = [' '.join(line) for line in lines]
    
    return newline_token.join(lines)

def get_rhyme_ind(word, vocab_to_index_dict, n_phones = 3):
    phones = pronouncing.phones_for_word(word)
    rhymes = []
    for phone in phones:
        last_phones = phone.split()[-n_phones:]
        print(last_phones)
        potential_rhymes = pronouncing.search(' '.join(last_phones)+'$')
        rhymes.extend(potential_rhymes)
    rhymes = set(rhymes)
    print(rhymes)
    if word in rhymes:
        rhymes.remove(word)
    indices = [vocab_to_index_dict.get(rhyme) for rhyme in rhymes]
    indices = set(indices)
    indices.remove(None)
    vec_oh = np.zeros(len(vocab_to_index_dict))
    for ind in indices:
        vec_oh[ind] = 1
    return vec_oh