#!/usr/bin/env python
import os
import hashlib
import json
import os

import requests
from requests.exceptions import HTTPError

BASE_URL = 'https://api.figshare.com/v2/{endpoint}'

# 10MB chunks of data
CHUNK_SIZE = 1048576

# get the token that is an environment variable
TOKEN = os.getenv('FIGSHARE_KEY')

# specify file that should be uploaded
FILE_PATH = '../models.pt'
TITLE = 'Model data'


def raw_issue_request(method, url, data=None, binary=False):
    headers = {'Authorization': 'token ' + TOKEN}
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

def issue_request(method, endpoint, *args, **kwargs):
    return raw_issue_request(method, BASE_URL.format(endpoint=endpoint), *args, **kwargs)


# create the article within figshare
def create_article(title):
    data = {
        'title': title 
    }
    result = issue_request('POST', 'account/articles', data=data)
    print ('Created article:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])
    return result['id']


# ensure file is good to be uploaded
def get_file_check_data(file_name):
    with open(file_name, 'rb') as fin:
        md5 = hashlib.md5()
        size = 0
        data = fin.read(CHUNK_SIZE)
        while data:
            size += len(data)
            md5.update(data)
            data = fin.read(CHUNK_SIZE)
        return md5.hexdigest(), size


# upload the file to figshare
def initiate_new_upload(article_id, file_name):
    endpoint = 'account/articles/{}/files'
    endpoint = endpoint.format(article_id)

    md5, size = get_file_check_data(file_name)
    data = {'name': os.path.basename(file_name),
            'md5': md5,
            'size': size}

    result = issue_request('POST', endpoint, data=data)
    print ('Initiated file upload:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])

    return result


def complete_upload(article_id, file_id):
    issue_request('POST', 'account/articles/{}/files/{}'.format(article_id, file_id))

def upload_parts(file_info):
    url = '{upload_url}'.format(**file_info)
    result = raw_issue_request('GET', url)

    print('Uploading parts:')
    with open(FILE_PATH, 'rb') as fin:
        for part in result['parts']:
            upload_part(file_info, fin, part)
    print


def upload_part(file_info, stream, part):
    udata = file_info.copy()
    udata.update(part)
    url = '{upload_url}/{partNo}'.format(**udata)

    stream.seek(part['startOffset'])
    data = stream.read(part['endOffset'] - part['startOffset'] + 1)

    raw_issue_request('PUT', url, data=data, binary=True)
    print ('  Uploaded part {partNo} from {startOffset} to {endOffset}'.format(**part))


def main():
    
    # create the article
    article_id = create_article(TITLE)

    # Then we upload the file.
    file_info = initiate_new_upload(article_id, FILE_PATH)

    upload_parts(file_info)

    # We return to the figshare API to complete the file upload process.
    complete_upload(article_id, file_info['id'])

if __name__ == '__main__':
    main()