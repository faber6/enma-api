def get_tokens_as_list(tokenizer_with_prefix_space, word_list):
    "Converts a sequence of words into a list of tokens"
    if word_list:
        tokens_list = []
        for word in word_list:
            tokenized_word = tokenizer_with_prefix_space(
                [word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list
    return None


def get_tokens_as_tuple(tokenizer_with_prefix_space, word):
    return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


def seqbias_to_tuple(tokenizer_with_prefix_space, seqbias_dict):
    # convert strings of sequence bias into tuples
    if seqbias_dict:
        for text in seqbias_dict.copy():
            text_tup = get_tokens_as_tuple(tokenizer_with_prefix_space, text)
            seqbias_dict[text_tup] = seqbias_dict.pop(text)
        return seqbias_dict
    return None
