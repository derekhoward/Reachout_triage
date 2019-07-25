from pathlib import Path
import pandas as pd
from finetune import Classifier


PROCESSED_PATH = Path('./data/processed')
FEATURES_PATH = Path('./data/features')
FEATURES_PATH.mkdir(exist_ok=True)


def flatten_cols(df):
    df.columns = [
        '-'.join(tuple(map(str, t))).rstrip('_') 
        for t in df.columns.values
        ]
    return df


def generate_GPT_feats(model_path, post_level=True):  
    if post_level:
        df = pd.read_csv(PROCESSED_PATH / 'all_posts_data.csv')
        df = df[df.predict_me | (df.label.notnull())].loc[:, ['post_id', 'cleaned_body']]
    else:
        df = pd.read_csv(PROCESSED_PATH / 'sentences.csv')
        df = df.rename(columns={'body': 'cleaned_body'})
    
    model = Classifier.load(model_path)
    texts_to_featurize = list(df.cleaned_body.astype(str))
    features = model.featurize(texts_to_featurize)
    
    # generate a df with features as cols, with index as post_id
    GPT_embeddings = pd.DataFrame(features)
    GPT_embeddings.index = df.post_id
    
    if post_level:
        GPT_embeddings = GPT_embeddings.add_prefix('post_lvl-')
    else:
        GPT_embeddings = GPT_embeddings.add_prefix('sentence_lvl-')
        GPT_embeddings = flatten_cols(GPT_embeddings.groupby('post_id').agg(['mean', 'max', 'min']))
    
    return GPT_embeddings


if __name__ == '__main__':
    sfeatures = generate_GPT_feats(model_path='./models/GPT-finetuned_all_posts.model', post_level=False)
    sfeatures.to_csv(FEATURES_PATH / 'GPT_ft_all-sent_lvl.csv')