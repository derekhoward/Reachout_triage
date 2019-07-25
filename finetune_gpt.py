import pandas as pd
from pathlib import Path
import argparse
from finetune import Classifier
#import config


DATA_PATH = Path('./data')
MODELS_PATH = Path('./models')
MODELS_PATH.mkdir(exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', default=147618, type=int,
                        help='Define number of posts to be used to perform unsupervised finetuning of language model, defaults to all posts available (147618)')
    parser.add_argument('--name', type=str, 
                        help='Name of model to be saved in ./models directory')
    parser.add_argument('--labeled', action='store_true',
                        help='Use only labeled posts for finetuning')
    args = parser.parse_args()

    # read in data and select sample based on CLI args
    posts_df = pd.read_csv(DATA_PATH/'processed'/'all_posts_data.csv', usecols=['post_id', 'cleaned_body', 'label', 'predict_me'])

    if args.labeled:
        posts_sample = posts_df[(posts_df.label.notnull()) | posts_df.predict_me]
    else:
        posts_sample = posts_df.sample(n=args.nrows, random_state=42)     

    texts = list(posts_sample.cleaned_body.astype(str))
    print(f'{len(texts)} posts will be used to finetune the GPT language model')

    model = Classifier(batch_size=8)
    model.fit(texts)

    model.save(MODELS_PATH / args.name)
