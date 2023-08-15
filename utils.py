def get_tokens_as_list(word_list, tokenizer_with_prefix_space):
    "Converts a sequence of words into a list of tokens"
    if word_list:
        tokens_list = []
        for word in word_list:
            tokenized_word = tokenizer_with_prefix_space(
                [word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list
    return None
