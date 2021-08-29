import io
import pathlib

import sentencepiece as spm

""" This is just a demonstration code for the Medium article. Uncomment the piece of code you want to use. """

CURRENT_PATH = pathlib.Path(__file__).parent
SPLITTED_ALICE_FILE_PATH = CURRENT_PATH.joinpath('dataset').joinpath('splitted_alice_in_wonderland.txt')

""" 1) Train from a corpus file. """

# # Train from a corpus file
# spm.SentencePieceTrainer.Train(input=SPLITTED_ALICE_FILE_PATH,
#                                model_prefix='sp_alice',
#                                vocab_size=1500,
#                                pad_id=0,                # Let's re-define the special tokens for TensorFlow!
#                                unk_id=1,
#                                bos_id=2,
#                                eos_id=3
#                                )


""" 2) Train from a corpus in memory. """

# # Pretend we have the sentences in memory
# with open(SPLITTED_ALICE_FILE_PATH, encoding='utf-8') as infile:
#     corpus = [line for line in infile]
#
# spm.SentencePieceTrainer.Train(sentence_iterator=iter(corpus),
#                                model_prefix='sp_alice',
#                                vocab_size=1500,
#                                pad_id=0,                # Let's re-define the special tokens for TensorFlow!
#                                unk_id=1,
#                                bos_id=2,
#                                eos_id=3
#                                )


""" 3) Leave the model in memory. """

model = io.BytesIO()

# Pretend we have the sentences in our memory
with open(SPLITTED_ALICE_FILE_PATH, encoding='utf-8') as infile:
    corpus = [line for line in infile]

spm.SentencePieceTrainer.Train(sentence_iterator=iter(corpus),
                               model_writer=model,
                               vocab_size=1500,
                               pad_id=0,                # Let's re-define the special tokens for TensorFlow!
                               unk_id=1,
                               bos_id=2,
                               eos_id=3
                               )

# Optionally, save it
with open(CURRENT_PATH.joinpath('sp_alice.model'), 'wb') as outfile:
    outfile.write(model.getvalue())

# Optionally, visualize it
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
encoded_input = sp.Encode("'Yes, that's it,' said the Hatter with a sigh: 'it's always tea-time'")

tokenized_input = [sp.IdToPiece(id) for id in encoded_input]
print(*tokenized_input)

