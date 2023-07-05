from abc import ABC

from path import Path
import numpy as np
import typing
from typing import Any, Tuple
import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import tensorflow_text as tf_text
import json
import re


class SingleChoiceGenerator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
                 context_text_processor,
                 target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


@SingleChoiceGenerator.add_method
def generate(self,
             texts, *,
             max_length=50,
             temperature=0.0):
    # Process the input texts
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]

    # Setup the loop inputs
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done, state, temperature)

        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)

        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)  # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens)
    return result


# @title
@SingleChoiceGenerator.add_method
def plot_attention(self, text, **kwargs):
    assert isinstance(text, str)
    output = self.generate([text], **kwargs)
    output = output[0].numpy().decode()

    attention = self.last_attention_weights[0]

    context = tf_lower_and_split_punct(text)
    context = context.numpy().decode().split()

    output = tf_lower_and_split_punct(output)
    output = output.numpy().decode().split()[1:]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap='viridis', vmin=0.0)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + output, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿äöüÄÖÜß]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


class Export(tf.Module, ABC):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def generate(self, inputs):
        return self.model.generate(inputs)


class TrainModel:
    path = Path('datenset/test.json')
    train_data = []
    UNITS = 256

    def __init__(self):
        self.target_text_processor = None
        self.context_text_processor = None

    def load_GermanQuAD(self):
        self.path = None
        p = Path('datenset/GermanQuAD_train.json')
        with p.open(mode='rt', encoding='utf-8') as f:
            data_temp = json.load(f)

        input = []
        output = []

        for paragraph in data_temp['data']:
            # print(paragraph['paragraphs'][0]['context'])
            sentence = paragraph['paragraphs'][0]['context']
            for qas in paragraph['paragraphs'][0]['qas']:
                answer = qas['answers'][0]['text']
                question = qas['question']
                inputSentence = sentence.replace(answer, "<A>") + " <A> " + answer
                input.append(inputSentence)
                output.append(question)

    def load_TestJson(self):
        with self.path.open(mode='rt', encoding='utf-8') as f:
            data_temp = json.load(f)

        input = []
        output = []

        for i in range(6):
            answer = data_temp['data'][i]["answer"]
            sentence = data_temp['data'][i]["sentence"].replace(answer, "<A>") + " <A> " + answer
            question = data_temp['data'][i]["question"]
            input.append(sentence)
            output.append(question)

    def prepare_training_data(self):
        #p = Path('datenset/GermanQuAD_train.json')
        #with p.open(mode='rt', encoding='utf-8') as f:
        #    data_temp = json.load(f)

        input = []
        output = []

        #for paragraph in data_temp['data']:
        #    # print(paragraph['paragraphs'][0]['context'])
        #    sentence = paragraph['paragraphs'][0]['context']
        #    if '===' in sentence:
        #        sentence = sentence.split('===')
        #        sentence = sentence[2]
        #    elif '==' in sentence:
        #        sentence = sentence.split('==')
        #        sentence = sentence[2]
        #
        #    for qas in paragraph['paragraphs'][0]['qas']:
        #        answer = qas['answers'][0]['text']
        #        # answer_start = qas['answers'][0]['answer_start']
        #        question = qas['question']
        #        answerIndex = sentence.find(answer)
        #        start = answerIndex
        #        end = answerIndex
        #        # find Start
        #        if answerIndex > len(answer):
        #            while sentence[start] != '.':
        #                start = start - 1
        #        start = start + 2
        #        # find End
        #        if answerIndex < len(sentence) - 1:
        #            while sentence[end] != '.':
        #                if end < len(sentence) - 1:
        #                    end = end + 1
        #                else:
        #                    break
        #        inputSentence = sentence[start:end].replace(answer, "<A>") + " <A> " + answer
        #        input.append(inputSentence)
        #        output.append(question)

        with self.path.open(mode='rt', encoding='utf-8') as f:
            data_temp1 = json.load(f)

        for i in range(7):
            answer1 = data_temp1['data'][i]["answer"]
            sentence1 = data_temp1['data'][i]["sentence"].replace(answer1, "<A>") + " <A> " + answer1
            question1 = data_temp1['data'][i]["question"]
            input.append(sentence1)
            output.append(question1)

        input = np.array([i for i in input])
        output = np.array([o for o in output])
        BUFFER_SIZE = len(output)
        BATCH_SIZE = 4
        print(BUFFER_SIZE)
        print(input)
        print(output)
        is_train = np.random.uniform(size=(len(input),)) < 0.8

        train_raw = (
            tf.data.Dataset
            .from_tensor_slices((input[is_train], output[is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE))
        val_raw = (
            tf.data.Dataset
            .from_tensor_slices((input[~is_train], output[~is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE))

        max_vocab_size = 5000

        self.context_text_processor = tf.keras.layers.TextVectorization(
            standardize=tf_lower_and_split_punct,
            max_tokens=max_vocab_size,
            ragged=True)

        self.context_text_processor.adapt(train_raw.map(lambda context, target: context))

        self.target_text_processor = tf.keras.layers.TextVectorization(
            standardize=tf_lower_and_split_punct,
            max_tokens=max_vocab_size,
            ragged=True)

        self.target_text_processor.adapt(train_raw.map(lambda context, target: target))

        train_ds = train_raw.map(self.process_text, tf.data.AUTOTUNE)
        val_ds = val_raw.map(self.process_text, tf.data.AUTOTUNE)

        print("Train: ", train_ds)
        print("Validation: ", val_ds)

        model = SingleChoiceGenerator(self.UNITS, self.context_text_processor, self.target_text_processor)
        model.compile(optimizer='adam',
                      loss=masked_loss,
                      metrics=[masked_acc, masked_loss])
        vocab_size = 1.0 * self.target_text_processor.vocabulary_size()

        print({"expected_loss": tf.math.log(vocab_size).numpy(),
               "expected_acc": 1 / vocab_size})
        print(model.evaluate(val_ds, steps=20, return_dict=True))
        history = model.fit(
            train_ds.repeat(),
            epochs=10,
            steps_per_epoch=10,
            validation_data=val_ds,
            validation_steps=20,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3)])
        inputs = [
            'Konvention ist, <A> als einfache Großbuchstaben anzugeben, hier T. <A> Typparameter',
            'Im Paket <A> befinden sich u. a. dynamische Datenstrukturen. <A> java.util',
            'Wie endliche Automaten lesen <A> eine Eingabe von einem Band und haben endlich viele Zustände. <A> Turing-Maschinen',
            'Java eine komfortable Möglichkeit'
        ]
        result = model.generate(inputs)  # Are you still home
        print("Result: ", result)
        print("Encoded Result: ", result[0].numpy().decode())
        export = Export(model)
        result = export.generate(tf.constant(inputs))
        #print("Result: ", result)
        tf.saved_model.save(export, 'generator',
                            signatures={'serving_default': export.generate})
        print("Model Saved!")
    def process_text(self, context, target):
        context = self.context_text_processor(context).to_tensor()
        target = self.target_text_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out


# @title
class ShapeChecker():
    def __init__(self):
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                                   mask_zero=True)

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(units,
                                      # Return the sequence and state
                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform'))

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()

        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   units, mask_zero=True)

        self.rnn = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.attention = CrossAttention(units)

        self.output_layer = tf.keras.layers.Dense(self.vocab_size)


@Decoder.add_method
def call(self,
         context, x,
         state=None,
         return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')

    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')

    if return_state:
        return logits, state
    else:
        return logits


@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature=0.0):
    logits, state = self(
        context, next_token,
        state=state,
        return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    done = done | (next_token == self.end_token)
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state
