import os
import pickle
from flask import Flask
from flask_restful import Api, Resource, reqparse
from pythainlp.tokenize import word_tokenize
import pandas as pd
import boto3

BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# intitialize
bm25_scorer = None
itoid = None

def create_app():
    app = Flask(__name__)
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(FetchClassifier, '/fetch_classifier')
    api.add_resource(IntentClassifier, '/intent_classifier')

app = create_app()
create_api(app)

class IntentClassifier(Resource):
    
    parser = reqparse.RequestParser()
    parser.add_argument('value',
        type=str,
        required=True,
        help="Sentence to send to chatbot agent cannot be empty."
        )

    @staticmethod
    def get_intent(sentence):
        tokenized_sent = word_tokenize(sentence)
        scores = bm25_scorer.get_scores(tokenized_sent)
        return int(pd.DataFrame({'scores':scores, 'intent_id': itoid}).groupby('intent_id').sum().idxmax()['scores'])

    def get(self):
        if bm25_scorer:
            payload = self.__class__.parser.parse_args()
            # 1. get intent
            intent_id = self.get_intent(payload['value'])

            return {'intent_id':intent_id}
        else:
            return {'message':'Please fecth classifier weight from S3 first.'}

class FetchClassifier(Resource):

    def get(self):
        global bm25_scorer, itoid
        s3_resource = boto3.resource('s3')
        print('Downloading from S3...')
        s3_resource.Object(BUCKET_NAME, 'bm25_scorer.pkl').download_file(f'bm25_scorer.pkl')
        s3_resource.Object(BUCKET_NAME, 'itoid.pkl').download_file(f'itoid.pkl')
        # load from local
        bm25_scorer = pickle.load(open('bm25_scorer.pkl','rb'))
        itoid = pickle.load(open('itoid.pkl','rb'))
        return {'message': "fecth result from S3 successfully!"}

if __name__ == "__main__":
    app.run(port=5000, debug=True)