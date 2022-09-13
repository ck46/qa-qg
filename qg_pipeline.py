import re
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from nltk import sent_tokenize

def has_figures_or_tables(text):
    return (bool(re.search('[fF]igure [0-9]+', text)) or
            bool(re.search('[Tt]able [0-9]+', text)))

# Answer Extraction Handler
class AEHandler:
  def __init__(self, model, tokenizer):
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)

  def __call__(self, context):
    answers = self.inference(self.preprocess(context))
    answers = [[a for a in ans if not has_figures_or_tables(a)] for ans in answers]
    return self.postprocess(answers)

  def preprocess(self, context):
    sents = sent_tokenize(context)

    inputs = []
    for i in range(len(sents)):
      source_text = "extract answers:"
      for j, sent in enumerate(sents):
        if i == j:
          sent = "<hl> %s <hl>" % sent
        source_text = "%s %s" % (source_text, sent)
        source_text = source_text.strip()
      source_text = source_text + " </s>"
      inputs.append(source_text)

    tokenized_inputs = self.tokenizer.batch_encode_plus(
        inputs, 
        max_length=256,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        pad_to_max_length=True,
        return_tensors="pt"
    )
    return tokenized_inputs

  def inference(self, inputs):
    outs = self.model.generate(
        input_ids=inputs['input_ids'].to(self.device),
        attention_mask=inputs['attention_mask'].to(self.device),
        num_beams=5,
        temperature=0.7,
        max_length=32)

    dec = [self.tokenizer.decode(ids, skip_special_tokens=False).replace('<pad> ', '').strip() for ids in outs]
    answers = [item.split('<sep>')[:-1] for item in dec]
    return answers

  def postprocess(self, outputs):
    return outputs


# Question Generation Handler
class QGHandler:
  def __init__(self, model, tokenizer, model_type='hl'):
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)
    self.model_type = model_type

  def __call__(self, answers, context):
    tokenized_inputs = self.preprocess(answers, context)
    return self.inference(tokenized_inputs)

  def preprocess(self, answers, context):
    # prepare inputs for question generation from answers
    sents = sent_tokenize(context)
    if self.model_type == 'hl':
      qg_inputs, qg_examples = self.prepare_inputs_hl(answers, sents)
    elif self.model_type == 'prefix':
      qg_inputs, qg_examples = self.prepare_inputs_prefix(answers, sents)
    else:
      qg_inputs, qg_examples = [], []

    tokenized_inputs = self.tokenizer.batch_encode_plus(
        qg_inputs, 
        max_length=512,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        pad_to_max_length=True,
        return_tensors="pt"
    )
    self.qg_examples = qg_examples
    return tokenized_inputs

  def inference(self, inputs):
    outs = self.model.generate(
        input_ids=inputs['input_ids'].to(self.device), 
        attention_mask=inputs['attention_mask'].to(self.device),
        max_length=32,
        num_beams=5,
        temperature=0.7,
      )

    questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    return questions

  def postprocess(self, questions):
    outputs = [{'question': que, 'answer': example['answer']} for example, que in zip(self.qg_examples, questions)]
    return outputs

  def prepare_inputs_hl(self, answers, sents):
    qg_examples = []
    for i, answer in enumerate(answers):
      if len(answer) == 0: continue
      for answer_text in answer:
        sent = sents[i]
        sents_copy = sents[:]

        answer_text = answer_text.strip()

        try:
          ans_start_idx = sent.index(answer_text)
        except:
          continue

        sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
        sents_copy[i] = sent

        source_text = " ".join(sents_copy)
        source_text = f"generate question: {source_text}" 
        #if self.model_type == "t5":
        source_text = source_text + " </s>"
        qg_examples.append({"answer": answer_text, "source_text": source_text})

    # question generation inputs
    qg_inputs = [example['source_text'] for example in qg_examples]
    return qg_inputs, qg_examples

  def prepare_inputs_prefix(self, answers, sents):
    qg_inputs = []
    qg_examples = []
    loaded = dict()
    for answer in answers:
      for answer_text in answer:
        if answer_text in loaded:
          continue
        else:
          loaded[answer_text] = 1
        for sent in sents:
          if answer_text in sent:
            source_text = f"context: {sent} answer: {answer_text} </s>"
            qg_inputs.append(source_text)
            qg_examples.append({'answer': answer_text, "source_text": source_text})
    return qg_inputs, qg_examples


# Question-Answer Generation Pipeline
class Pipeline:
  def __init__(self, q_model=None, q_tokenizer=None, a_model=None, a_tokenizer=None, q_model_type='hl'):
    self.q_model = q_model if q_model is not None else "ck46/t5-small-squad-qa-qg"
    self.q_tokenizer = q_tokenizer if q_tokenizer is not None else "ck46/t5-small-squad-qa-qg"
    self.a_model = a_model if a_model is not None else "ck46/t5-small-squad-qa-qg"
    self.a_tokenizer = a_tokenizer if a_tokenizer is not None else "ck46/t5-small-squad-qa-qg"
    self.q_model_type = q_model_type
    
    self.answer_extractor = AEHandler(self.a_model, self.a_tokenizer)
    self.question_generator = QGHandler(self.q_model, self.q_tokenizer, model_type=self.q_model_type)

  def __call__(self, context):
    answers = self.answer_extractor(context)
    questions = self.question_generator(answers, context)
    return self.question_generator.postprocess(questions)
