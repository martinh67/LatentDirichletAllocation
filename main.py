# declare imports required for the program
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from methods import clean_text, extract_files

# start the timer for the program
start = time.time()

# declare the main method
def main():

    # declare a dataframe for the extracted files
    df = extract_files(10)

    # clean the data and add to a new column in the dataframe
    df['vsm'] = df.body_text.apply(clean_text)

    # convert a collection of text documents to a matrix of token counts
    # ignore the terms that appear in more than 90% of the documents
    # ignore the terms that appear in less than 10% of the documents"
    vectoriser = CountVectorizer(max_df = 0.95, min_df = 0.05)

    # this did not produce good results
    # vectoriser = CountVectorizer(max_features = 1000)

    # vectorse the dataframe
    X = vectoriser.fit_transform(df.vsm)

    # declare the Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components = 10, max_iter = 10,
    learning_method = "online", verbose = True, random_state = 42)

    # fit the LDA data
    lda.fit_transform(X)

    # get the topic value terms from the LDA components
    topic_value_terms = lda.components_

    # get the feature names
    topic_names = vectoriser.get_feature_names()

    # print a space to clearly see the results
    print()

    # for each topic within the lda analysis
    for topic_id, topic in enumerate(lda.components_):

        # print the topic id usin f string
        print(f"Topic {topic_id}")

        # print the topic terms
        print(" ".join(topic_names[i] for i in topic.argsort()[:10]))

        # print a space
        print()

    # assign the document topic
    document_topic = lda.fit_transform(X)


# magic method to run the main function
if __name__ == "__main__":
    main()

# time of the program
print("\n" + 50 * "#")
print(time.time() - start)
print(50 * "#" + "\n")
