import pathlib

import sentencepiece as spm

""" This is just a demonstration code for the Medium article. Uncomment the piece of code you want to use. """

CURRENT_PATH = pathlib.Path(__file__).parent
MODEL_PATH = CURRENT_PATH.joinpath('sp_alice.model')

""" 1) Model from file. """

# sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
#
# encoded_input = sp.Encode("'Yes, that's it,' said the Hatter with a sigh: 'it's always tea-time'")
# print(*encoded_input)
#
# tokenized_input = [sp.IdToPiece(id) for id in encoded_input]
# print(*tokenized_input)


""" 2) Model from file - sampled. """

sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))

for _ in range(3):
    encoded_input = sp.Encode("'Yes, that's it,' said the Hatter with a sigh: 'it's always tea-time'",
                              enable_sampling=True,
                              alpha=0.1,        # Inverse of temperature
                              nbest_size=-1     # -1 = all the possibilities
                              )

    tokenized_input = [sp.IdToPiece(id) for id in encoded_input]
    print(*tokenized_input)

