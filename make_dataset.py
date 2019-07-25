import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
#import config
import unicodedata
from bs4 import BeautifulSoup
import spacy



DATA_DIR = Path('./data')
PROCESSED_PATH = DATA_DIR / 'processed'
PROCESSED_PATH.mkdir(exist_ok=True)


def xmlpost_to_dict(post):
    """
    Transforms a forum post from xml format into a dict
    Input: single xml forum post
    output: dict of data contained in forum post
    """

    tree = ET.parse(post)
    root = tree.getroot()
    msg = root.find('message')

    post_data = {}

    board_id = msg.find('board_id')
    post_data['board_id'] = int(board_id.text)

    root_post = msg.find('root').attrib['href']
    post_data['root_post'] = root_post.split('/')[-1]

    kudos = msg.find('kudos')
    count = kudos.find('count')
    post_data['kudos_count'] = int(count.text)

    edit_author_id = msg.find('last_edit_author').attrib['href']
    post_data['edit_author_id'] = int(edit_author_id.split('/')[-1])

    post_time = msg.find('post_time')
    post_data['post_time'] = post_time.text

    last_edit_time = msg.find('last_edit_time')
    post_data['last_edit_time'] = last_edit_time.text

    body = msg.find('body')
    post_data['body'] = body.text

    thread = msg.find('thread').attrib['href']
    post_data['thread'] = int(thread.split('/')[-1])

    board = msg.find('board').attrib['href']
    post_data['board'] = board.split('/')[-1]

    try:
        parent_post = msg.find('parent').attrib['href']
        post_data['parent_post'] = int(parent_post.split('/')[-1])
    except KeyError:
        post_data['parent_post'] = None

    views = msg.find('views')
    post_data['views'] = int(views.find('count').text)

    subject = msg.find('subject')
    post_data['subject'] = subject.text

    post_id = msg.find('id')
    post_data['post_id'] = int(post_id.text)

    author_id = msg.find('author').attrib['href']
    post_data['author_id'] = int(author_id.split('/')[-1])

    return post_data


def create_posts_df(post_filenames):
    """
    Takes a list of xml filenames and processes them to create a dataframe
    Input: list of post filenames to be processed
    Output: dataframe where each row is a post and columns are for the
            extracted information field
    """
    posts_list = []
    n = 1
    for post in post_filenames:
        try:
            processed_post = xmlpost_to_dict(post)
            posts_list.append(processed_post)
        except AttributeError:
            print('Error parsing post:', post)
            n += 1

    print("Posts with trouble parsing (possibly missing messages):" + str(n))
    df = pd.DataFrame(posts_list)
    df.post_time = pd.to_datetime(df.post_time)
    df.last_edit_time = pd.to_datetime(df.last_edit_time)
    # df.set_index(['post_id'])

    return df


def process_body(text):
    """
    Helper function to process/clean the post body.
    """
    # if text != None:
    if text is not None:
        soup = BeautifulSoup(str(text), 'html.parser')
        try:
            soup.find('blockquote').decompose()
            contained_quote = True

        except AttributeError:
            contained_quote = False

        cleaned = soup.get_text()
        cleaned = unicodedata.normalize("NFKD", cleaned)

        return cleaned, contained_quote
    else:
        cleaned = float("nan")
        contained_quote = float("nan")
        return cleaned, contained_quote


def process_images(text):
    """
    Helper used to extract images from post body for a image feature column.
    """
    # if text != None:
    if text is not None:
        soup = BeautifulSoup(str(text), 'html.parser')
        img = soup.img
        try:
            image = img['title']
            return image
        except (TypeError, KeyError):
            # print(img)
            pass


def merge_post_labels(posts_df, labels_file):
    """
    Merges triage labels from training data with the posts dataframe
    Input: posts dataframe
    Output: posts dataframe with associated labels
    """

    labels = pd.read_table(labels_file, header=None,
                           names=['post_id', 'label', 'granular_label'])
    labeled_df = posts_df.merge(labels, how='left', on='post_id')

    assert(labeled_df.shape[0] == posts_df.shape[0])

    return labeled_df


def merge_author_ranks(posts_df):
    """
    Merges author rankings with the posts dataframe
    Input: posts dataframe
    Output: posts dataframe with associated author rankings
    """
    # author rankings has duplicate rows for author_id == 3727
    # author_ranks = pd.concat([author_ranks_training, author_ranks_testing,
    #                           new_author_ranks]).drop_duplicates()

    # def merge_author_ranks(posts_df, authors_file, authors_summary_file):
    # authors summary file is inconsistently formated, requires fixing to ease read in

    author_ranks_train2017 = pd.read_csv(os.path.join(DATA_DIR,
                                                      'raw',
                                                      'clpsych17-data',
                                                      'data',
                                                      'training',
                                                      'author_rankings.tsv'),
                                         header=None,
                                         names=['author_id', 'author_rank'], sep='\t')

    author_ranks_test2017 = pd.read_csv(os.path.join(DATA_DIR,
                                                     'raw',
                                                     'clpsych17-test',
                                                     'user-rankings.tsv'),
                                        header=None,
                                        names=['author_id', 'author_rank'], sep='\t')

    author_ranks = pd.concat([author_ranks_train2017, author_ranks_test2017]).drop_duplicates(
        subset='author_id', keep='last')

    # need to open file because it has bad delimiter setup
    with open(os.path.join(DATA_DIR,
                           'raw',
                           'clpsych17-data',
                           'data',
                           'author_rankings_summary.tsv')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    staffRanks = list()
    for line in content:
        if line.endswith(" 1"):
            line = line[:-1].strip()
            staffRanks.append(line)
    author_ranks['is_staff'] = [
        True if x in staffRanks else False for x in author_ranks['author_rank']]

    df = posts_df.reset_index().merge(author_ranks, how='left', on='author_id')
    df = df.set_index('post_id')

    assert(df.shape[0] == posts_df.shape[0])

    return df


def merge_ground_truth(posts_df, label_file):
    label_table = pd.read_csv(label_file, sep='\t', header=None)
    label_table = label_table.iloc[:, :3]
    label_table.columns = ['post_id', 'temp_label', 'temp_granular_label']

    posts_df = posts_df.merge(label_table, how='left')
    # add new labels then drop the temp columns
    posts_df['label'] = np.where(posts_df.label.isnull(),
                                 posts_df.temp_label,
                                 posts_df.label)
    posts_df['granular_label'] = np.where(posts_df.granular_label.isnull(),
                                          posts_df.temp_granular_label,
                                          posts_df.granular_label)

    posts_df.drop(['temp_label', 'temp_granular_label'], axis=1, inplace=True)

    return posts_df


def merge_test_ids(posts_df, test_ids_location):
    test_ids = pd.read_table(test_ids_location, header=None)
    test_ids = test_ids.iloc[:, 0]
    assert(isinstance(test_ids, pd.Series))
    posts_df['predict_me'] = posts_df.post_id.isin(test_ids)

    return posts_df


def split_to_sentences(df):
    nlp = spacy.load('en_core_web_sm')

    # select rows of interest
    df = df[(df.label.notnull()) | (df.predict_me == True)]
    df['parsed'] = df['cleaned_body'].astype(str).apply(nlp)

    corpus_dict = {}
    for doc in df.iterrows():
        document_dict = {}
        post_id = doc[1].post_id
        parsed_doc = doc[1].parsed
        for i, sentence in enumerate(parsed_doc.sents):
            document_dict[i] = str(sentence)
        corpus_dict[post_id] = document_dict

    sentences = pd.DataFrame(corpus_dict).T.reset_index().rename(columns={'index': 'post_id'})
    sentences = pd.melt(sentences, id_vars='post_id', var_name='sentence_num', value_name='body')
    sentences.body = sentences.body.fillna(' ')
    sentences = sentences.dropna().sort_values(['post_id', 'sentence_num'])
    sentences = sentences.drop_duplicates(subset=['post_id', 'body'])

    output_path = PROCESSED_PATH / 'sentences.csv'
    print('--Writing sentences data to {}--'.format(output_path))
    sentences.to_csv(output_path, index=None)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    if os.path.exists(os.path.join(PROCESSED_PATH,
                                   'all_posts_data.csv')):
        print("-- all_posts_data.csv found locally - delete interm files if rerun needed")
        total_df = pd.read_csv(PROCESSED_PATH / 'all_posts_data.csv')
    else:
        training_post_filenames = glob.glob(os.path.join(DATA_DIR,
                                                         'raw',
                                                         'clpsych16-data',
                                                         'data',
                                                         'training',
                                                         'posts', '*.xml'))
        dev_post_filenames = glob.glob(os.path.join(DATA_DIR,
                                                    'raw',
                                                    'clpsych16-data',
                                                    'data',
                                                    'testing',
                                                    'posts', '*.xml'))

        new_posts2017 = glob.glob(os.path.join(DATA_DIR,
                                               'raw',
                                               'clpsych17-test',
                                               'posts', '*.xml'))

        training_labels = os.path.join(DATA_DIR,
                                       'raw',
                                       'clpsych16-data',
                                       'data',
                                       'training',
                                       'labels.tsv')
        dev_labels = os.path.join(DATA_DIR,
                                  'raw',
                                  'clpsych16-data',
                                  'data',
                                  'testing',
                                  'labels.tsv')

        training_df = create_posts_df(training_post_filenames)
        dev_df = create_posts_df(dev_post_filenames)
        new_df = create_posts_df(new_posts2017)

        training_df['corpus_source'] = '2016train_2017train'
        dev_df['corpus_source'] = '2016test_2017train'
        new_df['corpus_source'] = '2017test'

        training_df = merge_post_labels(training_df, training_labels)
        dev_df = merge_post_labels(dev_df, dev_labels)

        training_df = merge_author_ranks(training_df)
        dev_df = merge_author_ranks(dev_df)
        new_df = merge_author_ranks(new_df)

        total_df = pd.concat([training_df, dev_df, new_df])

        test_labels = os.path.join(DATA_DIR,
                                   'raw',
                                   'clpsych17-test',
                                   'test_ids.tsv')
        total_df.reset_index(inplace=True)
        total_df = merge_test_ids(total_df, test_labels)
        label_file = os.path.join(DATA_DIR,
                                  'raw',
                                  'clpsych17-test-labels.tsv')
        merge_ground_truth(total_df, label_file)
        output_path = PROCESSED_PATH / 'all_posts_data.csv'
        

        # clean body of text
        total_df['cleaned_body'], total_df['contained_quote'] = zip(*total_df['body'].apply(process_body))
        total_df['images'] = total_df['body'].apply(process_images)

        print('--Writing data to {}--'.format(output_path))
        total_df.to_csv(output_path, index=False)

        sentences_df = total_df.loc[:, ['post_id', 'cleaned_body', 'label', 'predict_me']]
        # the following will split posts into sentences and write out to a separate csv
        split_to_sentences(sentences_df)


if __name__ == '__main__':
    main()
