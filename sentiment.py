from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def sentiment_score(headlines):

    scores=[]

    for headline in headlines:

        result=classifier(headline)[0]

        if result['label']=="positive":
            scores.append(result['score'])

        elif result['label']=="negative":
            scores.append(-result['score'])

        else:
            scores.append(0)

    return sum(scores)/len(scores)