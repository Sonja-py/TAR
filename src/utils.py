import os
import torch
import torchaudio
import torchaudio.transforms as transforms

# Creates a 2D tensor of integers to represent each sentence. Each 1D tensor
# stores the ASCII encoding of every character in a word.
def text_to_int(transcript):
  transcript = transcript.strip().split()
  max_word_length = 0

  # Encode each character in ASCII and determine the longest word
  words_ascii = []
  for word in transcript:
    words_ascii.append(torch.ByteTensor(list(bytes(word, 'utf8'))))
    max_word_length = max(words_ascii[-1].size()[0], max_word_length)

  # Adds 0s to pad shorter words
  out = torch.zeros((len(words_ascii), max_word_length), dtype=torch.uint8)
  for i, x in enumerate(words_ascii):
      out[i, 0:x.size()[0]] = x
  
  return out

class input_data():

  def load_l2arctic(self):

    # Returns 2D array: speaker, audio
    def load_audio():
      audio = [0]*17
      for speaker in range(17):
        print(speaker)
        files = os.listdir('../l2arctic_recordings/'+str(speaker))
        individual_audio = [0]*1131
        i=0
        for file in files:
          waveform, sample_rate = torchaudio.load(file)
          
          # Match sampling rate between corpuses
          resample = transforms.Resample(sample_rate, 16000)
          waveform = resample(waveform)

          individual_audio[i] = waveform
          i+=1
        audio[speaker] = individual_audio
      return audio
    
    # Returns array of tensors for each transcript
    def load_transcript():
#      transcript_sentences = [0]*1131
      transcript_words = [0]*1131
      arctic_f = open('../arctic_transcript.txt', 'r')
      i = 0
      for line in arctic_f:
        line = line.strip()
#        transcript_sentences[i] = line
        transcript_words[i] = text_to_int(line.split())
        i+=1
      arctic_f.close()
      return transcript_words

    audio = load_audio()
    transcript = load_transcript()

    # Separate into training/testing data sets
    train_data = [0]*904
    test_data = [0]*225
    train_idx = 0
    test_idx = 0
    for i in range(len(audio)):
      if (i%5 == 0):
        test_data[test_idx] = (audio, transcript)
        test_idx+=1
      else:
        train_data[train_idx] = (audio, transcript)
        train_idx+=1

    return train_data, test_data

  def load_librispeech(self):
    train_data = torchaudio.datasets.LIBRISPEECH('./', url='train-clean-100', download=True)
    test_data = torchaudio.datasets.LIBRISPEECH('./', url='test-clean', download=True)

    def load_from_corpus(data):

      audio = [0]*28539
      label = [0]*28539
#     speaker = [0]*28539     Could be used if we want to train/test on individual speakers
      i = 0
      for (waveform, __, transcript, speaker_id, __, __) in data:
        audio[i] = (waveform)
        label[i] = (text_to_int(transcript))
      #  speaker[i] = (speaker_id)
        i+=1

      return audio, label
    
    train_data = load_from_corpus(train_data)
    test_data = load_from_corpus(test_data)

    return train_data, test_data