from collections import Counter
import json
import nltk.tokenize
import pdb
import tensorflow as tf

class Vocabulary(object):
	"""Simple vocabulary wrapper."""

	def __init__(self, vocab, unk_id):
		"""Initializes the vocabulary.

		Args:
			vocab: A dictionary of word to word_id.
			unk_id: Id of the special 'unknown' word.
		"""
		self._vocab = vocab
		self._unk_id = unk_id

	def word_to_id(self, word):
		"""Returns the integer id of a word string."""
		if word in self._vocab:
			return self._vocab[word]
		else:
			return self._unk_id

def _create_vocab(captions):
	"""Creates the vocabulary of word to word_id.

	The vocabulary is saved to disk in a text file of word counts. The id of each
	word in the file is its corresponding 0-based line number.

	Args:
		captions: A list of lists of strings.

	Returns:
		A Vocabulary object.
	"""
	print("Creating vocabulary.")
	counter = Counter()
	for c in captions:
		counter.update(c)
	print("Total words:", len(counter))

	# Filter uncommon words and sort by descending count.
	word_counts = [x for x in counter.items() if x[1] >= 4]
	word_counts.sort(key=lambda x: x[1], reverse=True)
	print("Words in vocabulary:", len(word_counts))

	# Write out the word counts file.
	with tf.gfile.FastGFile('wordcounts.txt', "w") as f:
		f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
	print("Wrote vocabulary file:", 'wordcounts.txt')

	# Create the vocabulary dictionary.
	reverse_vocab = [x[0] for x in word_counts]
	unk_id = len(reverse_vocab)
	vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
	vocab = Vocabulary(vocab_dict, unk_id)

	return vocab


def _process_caption(caption):
	"""Processes a caption string into a list of tonenized words.

	Args:
		caption: A string caption.

	Returns:
		A list of strings; the tokenized caption.
	"""
	tokenized_caption = ['<S>']
	tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
	tokenized_caption.append('</S>')
	return tokenized_caption


def main(unused_argv):

	captions_file = '/home/cvpr-gb/hdd4TBmount/DataSet/mscoco/annotations/captions_train2014.json'
	print( 'loading captions...' )
	with tf.gfile.FastGFile(captions_file, "r") as f:
		caption_data = json.load(f)

	print( 'extracting captions...' )
	raw_captions = []
	for annotation in caption_data["annotations"]:
		raw_captions.append(  annotation["caption"].encode('ascii','ignore') )

	print( 'processing captions...' )
	captions = [ _process_caption(c) for c in raw_captions ]

	print( 'creating vocab...' )
	vocab = _create_vocab(captions)


if __name__ == "__main__":
	tf.app.run()
