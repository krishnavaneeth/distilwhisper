import os
import torch
import jiwer
import datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (HfArgumentParser,
                          AutoConfig,
                          AutoFeatureExtractor,
                          AutoProcessor,
                          AutoModelForSpeechSeq2Seq,
                          AutoTokenizer,
                          AdamW)

from datasets import load_dataset , Features
# helper functions 

def save_model(output_dir, model, accelerate, feature_extractor, tokenizer,config,current_step,best_cer=None):
    """
    save the model, feature_extractor, tokenizer, config 

    arguments:
        --output_dir : output directory
        --model : WhisperForConditionalGeneration
        --accelerate : accelerator object 
        --feature_extractor : feature extractor obj
        --tokenizer : tokenizer obj
        --config : whisper model config 
        --best_cer : best cer
    """
    if best_cer:
        output_dir = os.path.join(output_dir,f"checkpoint-{current_step}-{best_cer}")
    else:
        output_dir = os.path.join(output_dir,f"checkpoint-{current_step}")
    accelerate.save_state(output_dir=output_dir)
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)

    
def load_arguments(tConfig,tArguments,distillArguments):
    """
    load the configurations for training 

    Arguments:
        --tconfig(distilwhisper.TrainingConfig) : TrainingConfig 
        --tArguments(distilwhisper.TrainingArguments) : TrainingArguments
        --distillArguments((distilwhisper.DsitilArguments) : DistilArguments
    return 
        respective configurtion for training 
    """
    parser = HfArgumentParser((tConfig,tArguments,distillArguments))
    tConfig, tArguments, distillArguments = parser.parse_args_into_dataclasses()
    return tConfig,tArguments,distillArguments

# load model 
def load_model(student_model:str,teacher_model:str,mixed_precesion:str="float32"):
    """
    load the model,tokenizer,preprocessor 

    arguments:
        --student_model (str) : name or path of student model
        --teacher_model (str) : name or path of teacher model 
        --mixed_precesion (str) : if user needs mixed prexesion training
    return:
        student_model, teacher_model, tokenizer, feature_extractor
    """
    config = AutoConfig.from_pretrained(student_model)
    featureExtractor = AutoFeatureExtractor.from_pretrained(student_model)
    tokenizer = AutoTokenizer.from_pretrained(student_model)
    student_model = AutoModelForSpeechSeq2Seq.from_pretrained(student_model)
    teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(teacher_model,torch_dtype=torch.bfloat16)
    return student_model,teacher_model,tokenizer,featureExtractor,config

def compute_metrics(pred,target,tokenizer):
    target[target == -100] = tokenizer.pad_token_id

    # convert into strings

    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
    ref = tokenizer.batch_decode(target, skip_special_tokens=True)
    return jiwer.cer(ref,pred) * 100



# helper classes 

#referenced from huggingface , whisper finetuning 
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class CustomDataset:
    """
    custom dataset class for training 
    """
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 feature_extractor: AutoFeatureExtractor,
                 test_size: float,
                 shuffle: bool,
                 language: str,
                 task: str,
                 audio_column: str,
                 text_column: str,
                 streaming: bool,
                 min_input_length: int,
                 max_input_length:int,
                 max_token:int) -> None:
        self.tokenizer = tokenizer # tokenizer object
        self.feature_extractor = feature_extractor # feature extractor object
        self.test_size = test_size #squeeze the test , based in the percent 
        self.shuffle = shuffle #shuff;e the data 
        self.language = language #target language used for tokenizer
        self.task = task  #task needs to be done (translate or transcribe)
        
        self.audio_column = audio_column #audio column name
      
        self.text_column = text_column # target text transcript name 
        self.streaming= streaming #whether to apply streaming or not 
        self.min_input_length = min_input_length #min input audio length
        self.max_input_length = max_input_length #max input audio length
        self.max_token = max_token #max token length
        
        # this is the format of our csv file , it requires for streaming , duration is to filer the data while pre processing 
        self.features = Features({"audio":datasets.Value(dtype='string',id=None),"transcript":datasets.Value(dtype='string',id=None),"duration":datasets.Value(dtype='float64',id=None)})
    
    def audio_filer(self,length):
        """
        for whisper audio length should be 30s and the target token should under 448 , outliers should be eliminated 
        """
        return length > self.min_input_length and length < self.max_input_length
    
    def token_filter(self,length):
        """
        for whisper audio length should be 30s and the target token should under 448 , outliers should be eliminated 
        """
        return length < self.max_token

    def get_features(self, batch):
        
        sample = batch[self.audio_column]
        inputs = self.feature_extractor(sample["array"], sampling_rate = sample["sampling_rate"])
        batch["input_features"] = inputs.get("input_features")[0]
        batch["input_length"] = len(sample["array"])
        input_str = batch[self.text_column]
        batch["labels"] = self.tokenizer(input_str).input_ids
        batch["transcript_length"] = len(batch["labels"])
        return batch
    
    # def get_features(self,batch):
    #     """
    #     preprocess , filter the data 

    #     arguments:
    #         --batch (dict) : audio array and text transcript 
    #     return:
    #         dict (audio array , text tokens)
    #     """
    #     mini_batch = batch[self.audio_column]
    #     mini_batch["input_features"] = self.feature_extractor(mini_batch["array"],sampling_rate=mini_batch["sampling_rate"]).input_features[0]
    #     mini_batch["labels"] = self.tokenizer(batch[self.text_column]).input_ids
        
    #     # these two are for filteration process , for whisper audio length should be 30s and the target token 
    #     # should under 448 , outliers are eliminated 
    #     mini_batch["input_length"] = len(mini_batch["array"])
    #     mini_batch["transcript_length"] = len(mini_batch["labels"])

    #     return mini_batch

    
    
    def load_dataset_from_csv(self,metadata_path):
        """
        load the csv file from the path
        """
        raw_ds = load_dataset("csv", data_files=metadata_path,streaming=self.streaming,features=self.features, split="train")
        raw_dataset = raw_ds.cast_column(self.audio_column, datasets.Audio(sampling_rate=16000))
        raw_dataset = raw_dataset.map(self.get_features)
        raw_dataset = raw_dataset.filter(self.audio_filer, input_columns = ["input_length"])
        # df = load_dataset("csv",data_files=metadata_path,streaming=self.streaming,features=self.features,split='train')
        # asr_dataset = df.cast_column(self.audio_column,datasets.features.Audio(sampling_rate=self.feature_extractor.sampling_rate))
        # asr_dataset = asr_dataset.map(self.get_features)
        # # filter the outliers
        # asr_dataset = asr_dataset.filter(self.audio_filer,input_columns=["input_length"])
        raw_dataset = raw_dataset.filter(self.token_filter,input_columns=["transcript_length"])
        return raw_dataset