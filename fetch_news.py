from newsapi import NewsApiClient

api = NewsApiClient(api_key='c8d2a485b55e47e1a412a04b55d87fcd')

articles = api.get_everything(
    q='Tesla',
    language='en',
    sort_by='publishedAt'
)

print("Total Results:", articles['totalResults'])

for article in articles['articles']:
    print(article['title'])