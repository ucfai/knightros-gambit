#!/usr/bin/env python
from email.mime import base
import os
import hashlib
import json
import os

from urllib.request import urlretrieve
import torch
import requests
from requests.exceptions import HTTPError

class FigshareApi:

    CHUNK_SIZE = 1048576
    BASE_URL = 'https://api.figshare.com/v2/{endpoint}'

    def __init__(self):
        # get the api key which is stored as an environment variable
        self.api_key = os.getenv('FIGSHARE_KEY')
            
    def get_figshare_dataset(self,store_path,file_name):
        '''
        get a dataset from figshare using the path provided
        '''
        endpoint = "account/articles/{article_id}/files"
        results = self.get_articles()
        download_url = None

        # find the file we want to download
        for item in results:
            if file_name == item['title']:
                article_id = item['id']
                file_data = self.issue_request('GET', endpoint.format(article_id=article_id))
                download_url = file_data[0]["download_url"]
        if not download_url:
            print("file not found")
        else:
            urlretrieve(download_url, store_path)

        return torch.load(store_path)

    def raw_issue_request(self,method, url, data=None, binary=False):
        headers = {'Authorization': 'token ' + self.api_key}
        if data is not None and not binary:
            data = json.dumps(data)
        response = requests.request(method, url, headers=headers, data=data)
        try:
            response.raise_for_status()
            try:
                data = json.loads(response.content)
            except ValueError:
                data = response.content
        except HTTPError as error:
            print ("Caught an HTTPError: {}".format(error.message))
            print ('Body:\n', response.content)
            raise

        return data

    def issue_request(self,method, endpoint, *args, **kwargs):
        return self.raw_issue_request(method, self.BASE_URL.format(endpoint=endpoint), *args, **kwargs)

    # create the article within figshare
    def create_article(self,title):
        # NOTE: this data is just place holder , however required for publishing
        data = {
            'title': title,
            "description": "Test description of article",
            "keywords": ["TEST","TEST2"],
            "categories": [ 1,10,11]
        }
        result = self.issue_request('POST', 'account/articles', data=data)
        print ('Created article:', result['location'], '\n')

        result = self.raw_issue_request('GET', result['location'])
        return result['id']


    # ensure file is good to be uploaded
    def get_file_check_data(self,file_name):
        with open(file_name, 'rb') as fin:
            md5 = hashlib.md5()
            size = 0
            data = fin.read(self.CHUNK_SIZE)
            while data:
                size += len(data)
                md5.update(data)
                data = fin.read(self.CHUNK_SIZE)

            return md5.hexdigest(), size


    # upload the file to figshare
    def initiate_new_upload(self,article_id, file_name):
        endpoint = 'account/articles/{}/files'
        endpoint = endpoint.format(article_id)

        md5, size = self.get_file_check_data(file_name)
        data = {'name': os.path.basename(file_name),
                'md5': md5,
                'size': size,
                }

        result = self.issue_request('POST', endpoint, data=data)
        print ('Initiated file upload:', result['location'], '\n')

        result = self.raw_issue_request('GET', result['location'])

        return result

    def get_articles(self):
        result = self.issue_request('GET', 'account/articles')
        return result

    def complete_upload(self,article_id, file_id):
        self.issue_request('POST', 'account/articles/{}/files/{}'.format(article_id, file_id))

    def upload_parts(self,file_info,path):
        url = '{upload_url}'.format(**file_info)
        result = self.raw_issue_request('GET', url)


        print('Uploading parts:')
        with open(path, 'rb') as fin:
            for part in result['parts']:
                self.upload_part(file_info, fin, part)
        print

    def upload_part(self,file_info, stream, part):
        udata = file_info.copy()
        udata.update(part)
        url = '{upload_url}/{partNo}'.format(**udata)

        stream.seek(part['startOffset'])
        data = stream.read(part['endOffset'] - part['startOffset'] + 1)

        self.raw_issue_request('PUT', url, data=data, binary=True)
        print ('  Uploaded part {partNo} from {startOffset} to {endOffset}'.format(**part))


    def upload(self,title,path):

        # create and upload the article
        article_id = self.create_article(title)
        file_info = self.initiate_new_upload(article_id, path)
        self.upload_parts(file_info,path)

        # We return to the figshare API to complete the file upload process.
        self.complete_upload(article_id, file_info['id'])

        # article must be published to be downloaded
        self.issue_request('POST', 'account/articles/{}/publish'.format(article_id))


       

