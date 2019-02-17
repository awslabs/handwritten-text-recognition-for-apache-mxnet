# Localizing for other languages
To localize the handwriting OCR pipeline for another language you are going to need:
1. A dataset of images. You will need a set of images of pages that contain handwritten text. For each line in the text you will need
the transcription.
2. A language model trained on the text for your language. It may be helpful to train the language model on text from other sources, not just the text for the images in your data set. 
The Gluon NLP toolkit ( http://gluon-nlp.mxnet.io/ ) comes with state-of-the-art language models that you can train on text for a language. Or perhaps if you can tolerate a slightly higher error rate you could skip the steps in the pipeline that use language modeling, e.g. there are use-cases where maximum accuracy is not necessary such as building a search index to return relevant documents.
3. Modifications to the code to handle the character set.

The code could be refactored in future to make it easier to adapt to different writing sets, but for now we give some tips to get you started adapting it to your needs.

Before we dive deep on the code, we give a few tips and tricks for Unicode processing in Python.

It is easy to get the official Unicode name for each character using unicodedata.name():

```python
import unicodedata
for c in "ekpɔ wò": # Ewe: he saw you
    print(unicodedata.name(c))
```
**LATIN SMALL LETTER E**

**LATIN SMALL LETTER K**

**LATIN SMALL LETTER P**

**LATIN SMALL LETTER OPEN O**

**SPACE**

**LATIN SMALL LETTER W**

**LATIN SMALL LETTER O WITH GRAVE**

You can use the official Unicode names in strings rather than having to decipher hexadecimal values:

```python
print("\N{LATIN SMALL LETTER E}\N{LATIN SMALL LETTER K}\N{LATIN SMALL LETTER P}\N{LATIN SMALL LETTER OPEN O}")
```
**ekpɔ**

These named Unicode codepoints can be mixed and matched with simple text in the interest of brevity, e.g.
```python
print("ekp\N{LATIN SMALL LETTER OPEN O}")
```
**ekpɔ**

Some characters have alternate representations in Unicode: precomposed and decomposed. For example, the following character

á 

can be represented as a single Unicode codepoint: U+00E1 LATIN SMALL LETTER A WITH ACUTE. This is called the composed representation.

Alternatively we could represent this using two Unicode codepoints: a lowercase a and a zero width combining accent. This is called the decomposed form. The rendering engine in the operating system of your computer knows to put the diacritic on top of the letter a:

```python
print("a\N{Combining Acute Accent}")
```
**á**


There are a few assumptions in the current implementation:
1. Text is written horizontally.
2. Text is written left-to-right. The visual sequence of characters on the page corresponds to how the sequence of characters is encoded in memory.
2. Text is written using the English Latin alphabet.
3. Words consist of a series of individual atomic characters i.e. precomposed characters. The pipeline does not recognize a letter and an accent diacritic as separate entities.

We address how to modify the pipeline for each of those assumptions in the sections below. 

## Text is written horizontally
The line segmentation would need to be changed for vertical writing systems e.g. traditional Mongolian script (https://en.wikipedia.org/wiki/Mongolian_script).
 
## Text is written left-to-right
The current pipeline assumes that the text is written left-to-right AND that the visual sequence of characters on the page
corresponds to how the sequence of characters is encoded in memory. The writing systems for some languages, such as Arabic, 
violate both these assumptions.

Some languages, e.g. Arabic, are written (mostly) right-to-left. But the in-memory encoding of the characters follows the 
logical order i.e. the first sound of the first word in a line is encoded in memory as the zeroth character. A line of text 
that ends with a question mark will have the question mark as the final codepoint in the in-memory string, but visually it 
will be represented as the leftmost glyph.

To handle these languages, the stages in the OCR pipeline
that locate the text on the page and segment it into horizontal lines can be used as-is.
The point where the process becomes sensitive is calculating the CTC loss – comparing the character guesses from the model
to the reference string.
 
The reference string gets encoded character by character here:
https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/handwriting_line_recognition.py#L213
 
For English it’s fairly straightforward – you just go left to right through the characters for each word, noting the index of the character. At the top of that file you’ll see where we list off the characters of the English alphabet:
 
https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/handwriting_line_recognition.py#L29
 
So for Arabic you would need to update the set of characters. And then you would need to encode the labels so that they match the visual order that the algorithm would be encountering them. For the simple case where all the text is written right to left it’s just a matter of saying that the zeroth label is the last char of the reference string etc.
Of course, Arabic writing is actually bidirectional so you would need handle that as you were encoding the string.

## Text is written using the English Latin alphabet
For languages that use the same Latin alphabet characters as English with no additional characters or diacritics, 
e.g. Swahili, you will not need to make deep changes. Change the data set and language model and retrain.

For languages that use Latin script with some additions e.g. if they add Ɖ 
( https://en.wikipedia.org/wiki/African_D ) you would need to add those characters to the string assigned to the variable alphabet\_encoding here: https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/handwriting_line_recognition.py#L29.

For languages that use Latin script plus additional diacritics you will need to use precomposed forms. For many languages, 
the precomposed codepoints are what is used by default so no additional processing will be necessary. However, if the
text is represented as decomposed codepoints you can convert it to a precomposed representation using the 
unicodedate.normalize() function in Python:

```python
import unicodedata

# an 'a' followed by an accent. The rendering engine will put the accent on top of the a
decomposed = "a\N{Combining Acute Accent}"
print(decomposed)

precomposed = unicodedata.normalize("NFC", decomposed)
print(precomposed)
print(unicodedata.name(precomposed))

print(len(precomposed) == 1)
```
**á**

**á**

**LATIN SMALL LETTER A WITH ACUTE**

**True**

For languages that use non-Latin scripts, you will need to specify the characters.

## Specific examples
Putting all these notes together, here's what you would need to do to adapt the OCR pipeline to some specific languages.

Languages that use a Brahmic script, e.g. Hindi. These scripts are [abugidas](https://en.wikipedia.org/wiki/Abugida). 
There are symbols for syllables and diacritics that modify these symbols e.g. to replace the default vowel. For these 
languages you would need to convert the sequence of syllable plus diacritic(s) to a precomposed form. All possible 
precomposed forms would have to be defined in the variable alphabet\_encoding. These languages are written left-to-right 
so no additional changes should be necessary.

For a language like [Ewe](https://en.wikipedia.org/wiki/Ewe_language), which uses a modified Latin script, you would need 
to add the extra characters in all their precomposed forms. For example, the phrase "ekpɔ wò" ("he saw you") shown previously 
 tells us that we would need to add the LATIN SMALL LETTER OPEN O codepoint and also the precomposed codepoint 
 LATIN SMALL LETTER O WITH GRAVE.
