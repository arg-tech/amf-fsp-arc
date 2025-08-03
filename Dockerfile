FROM python:3.11-slim

WORKDIR /home/FSP


COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gunicorn flask-compress


ENV HF_HOME=/root/.cache/huggingface



RUN python3 -c "\
from transformers import BartTokenizer, BartForConditionalGeneration, RobertaTokenizer, RobertaModel; \
BartTokenizer.from_pretrained('facebook/bart-large'); \
BartForConditionalGeneration.from_pretrained('facebook/bart-large'); \
RobertaTokenizer.from_pretrained('roberta-base'); \
RobertaModel.from_pretrained('roberta-base')"


COPY app/ app/
COPY boot.sh ./

RUN chmod +x boot.sh


EXPOSE 5050

ENTRYPOINT ["./boot.sh"]
