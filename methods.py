# import statements required for the program
import glob
import json
import re
from langdetect import detect
import pprint as pp
from nltk.corpus import stopwords
import pandas as pd

# obtain a unique list of english stopwords
stop = set(stopwords.words("english"))

# method to clean the text
def clean_text(row):

    # declare a list to hold each sentence
    sentence = []

    # for each term in the row
    for term in row.split():

        # replace all nonletters with blank space and make all letters lowercase
        term = re.sub('[^a-zA-Z]', "", term.lower())

        # remove all single characters
        term = re.sub(r'\s+[a-zA-Z]\s+', '', term)

        # additionall preprocessing step to remove empty spaces
        term = re.sub(r'\s+','', term)

        # additional preprocessing step to remove puncation
        term = re.sub(r'\W','', term)

        # if the term has greater than 3 characters
        if len(term) >= 3:

            # append the term to the sentence
            sentence.append(term)

    # get rid of stop words in the sentence
    sentence = [word for word in sentence if word not in stop]

    # return a string with a space between them
    return " ".join(sentence)

# method to extract the files
def extract_files(file_number):

    # declare the root folder
    root = r"/Users/martinhanna/Desktop/document_parses"

    # declare a variable to hold the langauges
    language_count = 0

    # declare a variable to hold langauges unable to be detected
    no_language_count = 0

    # use glob to get to the files from pmc
    json_files = glob.glob(f'{root}//pmc_json/*.json')

    # initalise a list for the body_text
    body_text_list = []

    # initalise a list for the paper_id
    paper_id = []

    # for the first file within the folder
    for file_path in json_files[:file_number]:

        # open the file
        with open(file_path) as file:

            # set the articles
            articles = json.load(file)

            # check if there is body text is present within the content
            if articles['body_text']:

                # for each article that has a body of text
                for article in articles['body_text']:

                    # look for the first instance of text within the body
                    text = article['text']

                    # checking for foreign langauges
                    try:

                        # detect the language from the text
                        language = detect(text)

                        # if the langauge is english
                        if language == "en":

                            # append to the body_text list
                            body_text_list.append(text)

                            # append to the paper id list
                            paper_id.append(articles['paper_id'])

                        # otherwise
                        else:

                            # increase foreign language count
                            language_count += 1

                    # if the language was not able to be detected
                    except:

                        # increase no langauge detected count
                        no_language_count += 1


    # create a dataframe using the abstract list
    df = pd.DataFrame({"body_text" : body_text_list, "id" : paper_id})

    # call helper method to format the data
    format_data(df)

    # return the df to the calling function
    return df


# helper method to format the data using metadata.csv file
def format_data(df):

    # read off id and published time from metadata.csv
    col_list = ["pmcid", "publish_time"]

    # read the data from the metadata.csv file
    df1 = pd.read_csv(r"/Users/martinhanna/Downloads/metadata.csv", usecols = col_list, low_memory = False)

    # merge dfs on the pmcid
    df = df.merge(df1[['publish_time', 'pmcid']], how = 'left', left_on = "id", right_on = "pmcid")

    # drop all data that does not have a publication date
    df.dropna(inplace = True)

    # add the year to the dataframe
    df['year'] = pd.to_datetime(df['publish_time'], errors = 'coerce').dt.year

    # sort the dataframe by year
    df.sort_values(by = ['year'], inplace = True)

    # delete the id column
    del df['id']

    # delete the pmcid column
    del df['pmcid']

    # delete the publish time column
    del df['publish_time']

    # return the dataframe to
    return df
