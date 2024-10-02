"""
Knowledge distillation on whisper model 
"""
import os 
import pandas as pd 
import torch
import warnings 
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from dataclasses import dataclass , field
from typing import Any, Dict, List, Optional
from sklearn.utils import shuffle
from utils import (load_arguments,
                   load_model,
                   DataCollatorSpeechSeq2SeqWithPadding,
                   CustomDataset,
                   compute_metrics,
                   save_model)
import logging
from accelerate import Accelerator

from transformers import (AutoProcessor,
                          AdamW,
                          get_linear_schedule_with_warmup)

from tqdm import tqdm


# 1.configuration 

@dataclass
class TrainingConfig:

    model_name_or_path: str = "knowledge_distillation/whisper-small"
    teacher_model: str = "knowledge_distillation/whisper-small"
    train_data_path: str = "test.csv"
    eval_data_path: str = "test.csv"
    max_train_samples: Optional[float] = 0.3
    max_eval_samples: Optional[float] = 0.3
    audio_column_name: str = "audio"
    text_column_name: str = "transcript"
    max_duration: Optional[float] = 1.0 #in seconds 
    min_duration: Optional[float] = 30.0 #in seconds
    max_token: int = 448 #over 448 is not supported by whisper 
    freeze_encoder: bool = False #no recommended , maybe fine , when weights copied from teacher model 
    freeze_layer_count:int = 12 #number of encoder layer want to freeze
    gradient_checkpoint : bool = True
    train_split: str = "train"
    test_split: str = "test"
    language: str = "chinese"
    task: str = "transcribe"
    total_train_samples: int = None
    total_eval_samples: int = None
@dataclass
class TrainingArguments:

    output_dir: str = "knowledge_distillation/data"
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation: int = 1 #recomended when batch size is lower 
    mixed_precesion: str = "bf16" 
    log_with: str = "tensorboard"
    warmup_steps: int = 500
    num_epochs: int = 3
    log_steps: int = 5
    do_eval: bool = True
    eval_steps: int = 2000
    save_checkpoints: int = 2000
    learning_rate: float = 0.0001
    tracker: bool = True

@dataclass
class DistillationArguments:
    
    kl_weight: float = 2.5
    temperature: float = 3.0
    hidden_distributed_layers: int = 24

def kl_divergence(target_distribution,log_pred_distribution,labels):
    # refer : pytorch documentation
    kl_loss = torch.nn.KLDivloss(reduction="sum")
    # i dont want to ignore pad token , it's been handled at preprocessing stage , i replaced with -100
    return kl_loss(log_pred_distribution,target_distribution)

def eval_step(s_model, t_model, batch, kl_weight):
    """
    eval step , inference and return the ,metrics 

    Arguments:
        --s_model : student model
        --t_model : teacher model 
        --batch (dict) : input and the respective transcript
        --kl_wieght : the percentage you want to take KL loss
    """
    s_model = s_model.eval()
    t_model = t_model.eval()

    with torch.no_grad():
        student_output = s_model(**batch)
        teacher_output = t_model(**batch)
    
    student_loss = student_output.loss
    # log softmax / softmax for numerical stability
    student_distribution = torch.nn.functional.log_softmax(student_output.logits, dim=-1)
    teacher_distribution = torch.nn.functional.softmax(teacher_output.logits, dim=-1)
    # temperature is always 1 for eval
    kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"])
    # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
    loss = 0.8 * student_loss + kl_weight * kl_loss
    return {"loss": loss, "student_loss": student_loss, "kl_loss": kl_loss,"student_logits":student_loss.logits.argmax(dim=-1)}


def train_step(accelerate,s_model,t_model,batch,distil_args,tokenizer,layer_wise_sharing):
    """
    Arguments:
        --accelerate : accelrator object
        --s_model : student model 
        --t_model : teacher model
        --batchj(dict) : input information 
        --temperature : hyper paramter to control the teacher knowledge to student 
        --tokenizer : tokenizer , for compute metrics
        --layer_wise_sharing : do we want to share the internal state repersentation from teacher to student
    return:
        loss,metric
    """
    s_model.train()
    # we dont want teacher model to be in train 
    t_model.eval()

    student_output = s_model(**batch)
    with torch.no_grad():
        teacher_output = t_model(**batch,)

    # student model loss
    student_loss = student_output.loss
    # now find the distribution , basically we are rescaling the distribution based on the temperature , go the readme.md , on topic , temperature scaling 
    teacher_distribution = torch.nn.functional.softmax(teacher_output.logits / distil_args.temperature ,dim = -1)
    # Importtant , you need to use log_softmax
    #refer :https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
    student_distribution = torch.nn.functional.log_softmax(student_output.logits / distil_args.temperature ,dim = -1)

    # apply KL loss 
    kl_loss = kl_divergence(teacher_distribution,student_distribution,batch["labels"]) * distil_args.temperature**2

    # refer : https://arxiv.org/abs/2311.00430
    # basically i took 50 percent from my student model and take 50 percent from teacher , student distribution  , combine that 
    loss = 0.5 * student_loss + distil_args.kl_wieght * kl_loss

    return {"student_logits":student_output.logits.argmax(dim=-1),"student_loss":student_loss,"kl_loss":kl_loss,"loss":loss}

def eval(accelerate, s_model, t_model, eval_dataloader,train_args,distil_args,tokenizer,feature_extractor,config,current_step):

    # unwrap the model for evaluation 
    total_eval_loss, student_loss ,total_eval_cer, kl_loss = 0, 0, 0, 0
    for batch in tqdm(eval_dataloader):
        metric = eval_step(s_model, t_model, batch,distil_args.kl_weight)
        total_eval_loss += metric["loss"]
        kl_loss += metric["kl_loss"]
        student_loss += metric["student_loss"]
        predictions = metric["student_logits"]
        predictions , targets = accelerate.gather_for_metrics((predictions,batch["labels"]))
        total_eval_cer += compute_metrics(predictions, targets,tokenizer)

    logs = {"loss_eval":total_eval_loss/train_args.batch_size,
            "kl_loss":kl_loss/train_args.batch_size,
            "eval_cer":total_eval_cer/train_args.batch_size}
    
    if (total_eval_cer/train_args.batch_size) <= best_cer:
        best_cer = total_eval_cer/train_args.batch_size
        save_model(train_args.output_dir, s_model, accelerate, feature_extractor, tokenizer,config,current_step,best_cer)
    else:
        logs = {
            "eval/eval_cer":metric["eval_cer"],
            "eval/eval_loss":metric["eval_loss"],
            "eval/kl_loss":metric["kl_loss"]
        }
        accelerate.logs(logs,step=current_step)

        


def train(accelerate,
          s_model,
          t_model,
          optimizer,
          lr_scheduler,
          train_dataloader,
          eval_dataloader,
          train_args,
          train_cfg,
          distil_args,
          tokenizer,
          feature_extractor,
          config):
    """
    training

    arguments:
        --accelerate: accelerator obj 
        --s_model : student model 
        --t_model : teacher mdel
        --optimizer : AdamW optimizer 
        --lr_scheduler : learning rate scheduling and warmup steps information 
        --train_dataloader : train set 
        --eval_dataloader : eval set
        --train_args : training arguments 
        --train_cfg : training confifurations 
        --distil_arguments : distillation arguments
        --tokenizer : model tokenizer , for compute metrics
        --feature_extractor : feature extractor 
        --config : configs that used in model
    """
    print("training started::")
    current_step,  train_loss, train_cer , total_kl_loss =  0, 0, 0, 0
    total_steps = train_args.total_train_samples // train_args.train_batch_size
    best_cer = float("-inf")
    for epoch in range(train_args.num_epochs):
        print("epoch ::",epoch)
        for batch in tqdm(train_dataloader):
            with accelerate.accumulate(s_model):
                metric = train_step(accelerate,s_model,t_model,batch,distil_args,None)
                train_loss += metric["loss"].detach().item()
                total_kl_loss += metric["kl_loss"].detach().item()
                accelerate.backward(metric["loss"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                predictions = metric["student_logits"]
                predictions , targets = accelerate.gather_for_metrics((predictions,batch["labels"]))
                train_cer += compute_metrics(predictions, targets,tokenizer)
            current_step += 1
            global_step += 1
            # logging
            if current_step % train_args.log_steps == 0:
                train_cer = train_cer / (train_args.log_steps * train_args.train_batch_size)
                train_loss = train_loss / (train_args.log_steps * train_args.train_batch_size)
                total_kl_loss = total_kl_loss / (train_args.log_steps * train_args.train_batch_size)
                logs = {"train/loss_train":train_loss,
                        "train/kl_loss":total_kl_loss}
                train_cer = 0
                train_loss = 0
                total_kl_loss = 0

                # logging in tensorboard
                accelerate.log(logs,step=current_step)
                #stdout for training #write a logger her , with all the steps and loss information 

            if current_step % train_args.save_checkpoints == 0:
                save_model(train_args.output_dir, s_model, accelerate, feature_extractor, tokenizer,config,current_step)
                accelerate.wait_for_everyone()
            if current_step % train_args.eval_step == 0 and train_args.do_eval:
                # write logging here!!
                eval(accelerate,s_model,eval_dataloader,train_args,tokenizer,feature_extractor,config,current_step)
    
def main():
    """
    load the configuration and the data
    """

    # load configurations 
    train_cfg, train_args, distil_args = load_arguments(TrainingConfig,TrainingArguments,DistillationArguments)
    # shufffle the train set
    train_df = pd.read_csv(train_cfg.train_data_path)
    eval_df = pd.read_csv(train_cfg.eval_data_path)
    train_args.total_train_samples = train_df.shape[0]
    train_args.total_eval_samples = eval_df.shape[0]
    # log here the length of the training set
    print("lngth of df",train_df.shape[0],eval_df.shape[0])
    # initilazie acclerate and the respetive 
    accelerate = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation,
        mixed_precision=train_args.mixed_precesion,
        log_with=train_args.log_with,
        project_dir=train_args.output_dir
    )

    # accelerate init trackers 
    accelerate.init_trackers("distil-whisper")

    # load model #student model , #teacher model 
    s_model, t_model, tokenizer, featureExtractor,config = load_model(train_cfg.model_name_or_path,
                                                                      train_cfg.teacher_model,
                                                                      train_args.mixed_precesion)
    print("Model loaded successfully ::")
    if train_cfg.gradient_checkpoint:
        s_model.config.use_cache = True
        s_model.gradient_checkpointing_enable()
    
    #freeze the student encoder
    if train_cfg.freeze_encoder:
        # assert len(config.encoder_layers) < train_args.freeze_layer_count , f"requested freezing encoder layer is greater than the actual 
        #                                                                     encoder layer in student model: {config.encoder_layer} requested :: {train_args.freeze_layer_count}!!"
        for _,param in s_model.model.encoder():
            param.requires_grad = False
        s_model.model.encoder.gradient_checkpointing = False

    # save the models and the accelerator running in min process 
    # with accelerate.main_process_first():
    #     featureExtractor.save_pretrained(train_args.output_dir,"checkpoint")
    #     tokenizer.save_pretrained(train_args.output_dir,"checkpoint")
    #     config.save_pretrained(train_args.output_dir,"checkpoint")
    
    processor = AutoProcessor.from_pretrained(train_cfg.model_name_or_path)

    # load the dataset
    datacollator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=s_model.config.decoder_start_token_id,
    )
    # print("ta",train_cfg.audio_column_name)
    customDataset = CustomDataset(
        tokenizer = tokenizer,
        feature_extractor = featureExtractor,
        test_size = 0.2,
        shuffle = True,
        language = train_cfg.language,
        task = train_cfg.task,
        audio_column = train_cfg.audio_column_name,
        text_column = train_cfg.text_column_name,
        streaming=True,
        min_input_length = int(train_cfg.min_duration * featureExtractor.sampling_rate),
        max_input_length = int(train_cfg.max_duration * featureExtractor.sampling_rate),
        max_token = train_cfg.max_token

    )
    stream_train = customDataset.load_dataset_from_csv(train_cfg.train_data_path)
    stream_eval = customDataset.load_dataset_from_csv(train_cfg.eval_data_path)

    train_dataloader = DataLoader(stream_train,
                                  collate_fn=datacollator,
                                  batch_size=train_args.train_batch_size,
                                  drop_last = True)
    eval_dataloader = DataLoader(stream_eval,
                                  collate_fn=datacollator,
                                  batch_size=train_args.eval_batch_size,
                                  drop_last = True)
    
    optimizer = AdamW(params=s_model.parameters(),lr=train_args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=train_args.warmup_steps,
                                                   num_training_steps=(train_args.total_train_samples * train_args.num_epochs))
    
    # unpack the models ,dataloader , optimizer , lr_scheduler
    s_model, t_model, optimizer, lr_scheduler = accelerate.prepare(
        s_model,
        t_model,
        optimizer,
        lr_scheduler
    )

    train(accelerate,
          s_model,
          t_model,
          optimizer,
          lr_scheduler,
          train_dataloader,
          eval_dataloader,
          train_args,
          train_cfg,
          distil_args,
          tokenizer,
          featureExtractor,
          config)   

if __name__ == "__main__":
    main()
