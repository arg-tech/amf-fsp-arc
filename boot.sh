#!/bin/sh


export HF_HOME=/root/.cache/huggingface


python3 -c "from frame_semantic_transformer import FrameSemanticTransformer; f = FrameSemanticTransformer(); f.setup()"


exec gunicorn -w 2 -k gthread -b 0.0.0.0:5050 --timeout 600 --access-logfile - --error-logfile - app.routes:app
